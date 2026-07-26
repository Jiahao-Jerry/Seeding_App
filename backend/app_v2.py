"""
"New architecture" seeding app -- fully separate from backend/app.py, run on
its own port so both can be tested side by side.

What's different from the main app:
  - NO LLM profile update anywhere (backend/session_v2.py: topics via simple
    frequency counting, style axes via the Bradley-Terry model exclusively).
  - NO LLM substance-check gate. Verification is classifier (style-axis
    shift) + BERTScore (content preservation) only (backend/transform_v2.py).
  - Rewrites that fail the gate are REGENERATED (up to 3 attempts) instead of
    silently falling back to the original on the first failure.
  - NO SAE anywhere.

Reuses (unmodified, imported directly) from backend.app: get_data(),
post_to_api_dict(), get_shelf_posts(), get_pair_posts() -- pure/duck-typed
helpers that don't care which SessionState variant is passed in. Reuses
decide_next_action()/check_stop_condition()/compute_engagement_centroid()
from backend.session the same way.
"""
import asyncio
import json
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import AuthenticationError
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROFILES_DIR

from config.settings import MIN_TOPICS_CONFIDENT, LLM_MODEL_PROFILE

from backend.app import get_data, post_to_api_dict, get_shelf_posts, get_pair_posts
from backend.session import decide_next_action, check_stop_condition, compute_engagement_centroid
from backend.session_v2 import SessionStateV2, update_profile_v2, TOPIC_CONFIDENCE_SATURATION
from backend.transform import compute_anti_preference_deltas
from backend.transform_v2 import transform_post_v2
from backend.llm_helper import llm_json
from fastapi.exceptions import RequestValidationError

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="Adaptive Delivery Seeding — v2 (no-LLM-profile architecture)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Without this, any request that fails Pydantic validation (missing
    # field, wrong type, wrong route signature, etc.) surfaces as a raw
    # error array like {"detail":[{"type":"missing","loc":["body"],...}]}
    # directly on screen. That's never something a user should see -- show a
    # plain retry-worthy message instead, app-wide, for any endpoint. Note
    # this only masks the display: it does NOT make the underlying request
    # succeed, so a deterministic validation bug (wrong signature, route
    # decorator on the wrong function, etc.) will still fail every time --
    # fix the actual cause, don't rely on this handler alone.
    return JSONResponse(
        status_code=422,
        content={"detail": "Something went wrong sending your request. Please try again."},
    )


@app.exception_handler(AuthenticationError)
async def openai_auth_error_handler(request: Request, exc: AuthenticationError):
    # Without this, an invalid/expired OPENAI_API_KEY_UMICH_DYIMOD surfaces
    # as a raw, unhandled "Internal Server Error" with a stack trace instead
    # of a clear message -- this catches it app-wide, regardless of which
    # endpoint's LLM call triggered it.
    return JSONResponse(
        status_code=503,
        content={"detail": "The AI rewriting service is unavailable right now — the API key is invalid or expired. Contact the app admin."},
    )


@app.on_event("startup")
async def _warmup_style_verifier():
    """Same warmup as the main app -- load the classifier and force
    BERTScore's model to load once at startup."""
    import threading
    def _load():
        try:
            from backend.verify_classifier import load_classifier, verify_rewrite
            load_classifier()
            print("[startup v2] Style classifier loaded.")
            verify_rewrite("warm up text one", "warm up text two", ["tone"])
            print("[startup v2] BERTScore model ready.")
        except Exception as e:
            print(f"[startup v2] Style verifier warm-up failed: {e}")
    threading.Thread(target=_load, daemon=True).start()


_sessions_v2: dict[str, SessionStateV2] = {}


# ── API models ───────────────────────────────────────────────────
class StartSessionResponse(BaseModel):
    session_id: str
    needs_topic_selection: bool = False
    available_topics: list[str] | None = None
    action: dict | None = None
    posts: list[dict] | None = None


class ChooseTopicsRequest(BaseModel):
    session_id: str
    topics: list[str]


class InteractionRequest(BaseModel):
    session_id: str
    chosen_post_ids: list[str]
    shown_post_ids: list[str]


class InteractionResponse(BaseModel):
    session_id: str
    is_complete: bool
    confidence: dict
    style_prefs: dict
    all_style_prefs: dict = {}  # unfiltered preferred_value for every axis -- debug/dev display only
    step: int
    action: dict | None = None
    posts: list[dict] | None = None


class ProfileResponse(BaseModel):
    session_id: str
    confidence: dict
    style_prefs: dict
    all_style_prefs: dict = {}  # unfiltered preferred_value for every axis -- debug/dev display only
    liked_post_ids: list[str]
    engagement_centroid: list[float]
    n_interactions: int


# "short" variant (used by the Chinese demo flow, index_zh.html): 6 real
# pair-rounds after topic selection instead of 11 (max_steps = 1 for topic
# selection + 6), split 3 explore / 3 exploit instead of 5/6, and a 6-trial,
# all-standard eval (no anti_preference/rescue trials attempted at all).
# Purely a per-session config choice -- the English flow (index.html) never
# passes ?variant=short, so it's completely unaffected.
SHORT_VARIANT_MAX_STEPS = 7
SHORT_VARIANT_EXPLORE_ROUNDS = 3
SHORT_VARIANT_EVAL_TRIAL_COUNT = 6
# Lowered from bt_profile.MIN_CONFIDENCE_FOR_PREFS (0.3): only 6 pairwise
# comparisons total means per-axis confidence is naturally lower across the
# board, so the normal bar can leave style_prefs with just a single
# qualifying axis -- confirmed live, one such session got permanently stuck
# on "no posts available" because every eval trial was forced onto that one
# axis with no alternative. Halving the bar lets 2-3 axes typically qualify
# instead, spreading eval trials across more targets.
SHORT_VARIANT_MIN_CONFIDENCE = 0.15


# ── Endpoints ────────────────────────────────────────────────────
@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session(variant: str = "full"):
    """
    v2 has no post-based topic-discovery shelf: topic confidence here is just
    a frequency count over chosen posts, so rather than inferring interest
    indirectly (show posts, count which topics get picked), we ask directly.
    The frontend shows available_topics as a picker; the actual first round
    of posts only gets generated after /api/session/choose-topics.

    variant="short" configures a shortened session (see SHORT_VARIANT_*
    above) for the whole rest of its lifetime -- everything downstream
    (decide_next_action, check_stop_condition, the eval endpoints,
    _generate_next_trial) reads these per-session fields off state instead
    of the module-level constants, so this doesn't touch the standard flow.
    """
    corpus, _, _, _ = get_data()

    state = SessionStateV2()
    if variant == "short":
        state.max_steps = SHORT_VARIANT_MAX_STEPS
        state.explore_rounds = SHORT_VARIANT_EXPLORE_ROUNDS
        state.eval_trial_count = SHORT_VARIANT_EVAL_TRIAL_COUNT
        state.eval_all_standard = True
        state.style_prefs_min_confidence = SHORT_VARIANT_MIN_CONFIDENCE
    _sessions_v2[state.session_id] = state

    topics = sorted(corpus["topic_name"].unique().tolist())
    return StartSessionResponse(
        session_id=state.session_id,
        needs_topic_selection=True,
        available_topics=topics,
    )


def _scope_to_engaged_topics(corpus, pairs, cross_pairs, state):
    """
    get_pair_posts() (shared with the main app) picks pairs by target_axis
    alone -- its cross-topic-pair and final random-fallback branches ignore
    the topic field entirely, so Mode B could surface any topic in the
    corpus regardless of what the user actually said they're interested in.
    Restrict the corpus + pair tables to the user's engaged topics *before*
    calling it, so every branch (including the fallbacks) stays scoped.
    """
    engaged = {t for t, c in state.confidence.get("topics", {}).items() if c > 0.25}
    if not engaged:
        return corpus, pairs, cross_pairs

    scoped_corpus = corpus[corpus["topic_name"].isin(engaged)]
    valid_ids = set(scoped_corpus["post_id"])

    def _both_in_scope(df):
        if df is None or df.empty:
            return df
        mask = df["high_post_id"].isin(valid_ids) & df["low_post_id"].isin(valid_ids)
        return df[mask]

    return scoped_corpus, _both_in_scope(pairs), _both_in_scope(cross_pairs)


@app.post("/api/session/choose-topics", response_model=InteractionResponse)
async def choose_topics(req: ChooseTopicsRequest):
    """
    Explicit topic pick: sets those topics' confidence straight to 1.0 (full
    confidence, since the user told us directly) instead of building it up
    via update_profile_v2's frequency counting over several shelf rounds.
    """
    corpus, pairs, cross_pairs, _ = get_data()
    state = _sessions_v2.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    valid_topics = set(corpus["topic_name"].unique())
    chosen = [t for t in dict.fromkeys(req.topics) if t in valid_topics]  # de-dupe, preserve order
    if len(chosen) < MIN_TOPICS_CONFIDENT:
        raise HTTPException(400, f"Pick at least {MIN_TOPICS_CONFIDENT} topics.")

    for t in chosen:
        state.topic_counts[t] = TOPIC_CONFIDENCE_SATURATION
        state.confidence["topics"][t] = 1.0

    state.history.append({
        "step": state.step_count,
        "topic_selection": True,
        "topics_chosen": chosen,
    })
    # decide_next_action() forces a shelf round when step_count == 0
    # regardless of topic confidence -- bump past that since topic selection
    # already covers what that first shelf round would have discovered.
    state.step_count = 1

    all_topics = corpus["topic_name"].unique().tolist()
    action = decide_next_action(state, all_topics)
    state.last_action = action

    if action["mode"] == "shelf":
        posts = get_shelf_posts(corpus, action, state)
    else:
        scoped_corpus, scoped_pairs, scoped_cross_pairs = _scope_to_engaged_topics(
            corpus, pairs, cross_pairs, state)
        posts = get_pair_posts(scoped_corpus, scoped_pairs, action, state,
                                cross_pairs=scoped_cross_pairs)

    return InteractionResponse(
        session_id=state.session_id, is_complete=False,
        confidence=state.confidence, style_prefs=state.style_prefs,
        all_style_prefs=state.all_style_prefs, step=state.step_count,
        action=action, posts=posts,
    )


@app.post("/api/session/interact", response_model=InteractionResponse)
async def interact(req: InteractionRequest, background_tasks: BackgroundTasks):
    corpus, pairs, cross_pairs, embeddings = get_data()

    state = _sessions_v2.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    shown_mask = corpus["post_id"].isin(req.shown_post_ids)
    shown_posts = [post_to_api_dict(row) for _, row in corpus[shown_mask].iterrows()]

    state = update_profile_v2(state, shown_posts, req.chosen_post_ids, action=state.last_action)

    if state.is_complete:
        await _save_profile(state, corpus, embeddings)
        # Start filling the eval queue the moment the profile is ready --
        # rather than waiting for the "Start Evaluation" click -- so the
        # background work gets a head start during whatever time the user
        # spends on the "profile ready" screen before navigating to eval.
        background_tasks.add_task(_fill_eval_queue, state.session_id)
        return InteractionResponse(
            session_id=state.session_id, is_complete=True,
            confidence=state.confidence, style_prefs=state.style_prefs,
            all_style_prefs=state.all_style_prefs, step=state.step_count,
        )

    topics = corpus["topic_name"].unique().tolist()
    action = decide_next_action(state, topics)
    state.last_action = action

    if action["mode"] == "shelf":
        posts = get_shelf_posts(corpus, action, state)
    else:
        valid_topics = set(corpus["topic_name"].unique())
        if action.get("topic") not in valid_topics:
            action["topic"] = next(
                (t for t in state.confidence.get("topics", {}) if t in valid_topics),
                list(valid_topics)[0]
            )
        scoped_corpus, scoped_pairs, scoped_cross_pairs = _scope_to_engaged_topics(
            corpus, pairs, cross_pairs, state)
        posts = get_pair_posts(scoped_corpus, scoped_pairs, action, state,
                                cross_pairs=scoped_cross_pairs)

    return InteractionResponse(
        session_id=state.session_id, is_complete=False,
        confidence=state.confidence, style_prefs=state.style_prefs,
        all_style_prefs=state.all_style_prefs, step=state.step_count,
        action=action, posts=posts,
    )


@app.get("/api/session/{session_id}/profile", response_model=ProfileResponse)
async def get_profile(session_id: str):
    corpus, _, _, embeddings = get_data()
    state = _sessions_v2.get(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    pid_to_emb = {}
    if embeddings is not None:
        for i, row in corpus.iterrows():
            pid_to_emb[str(row["post_id"])] = embeddings[i]
    centroid = compute_engagement_centroid(state.liked_post_ids, pid_to_emb)

    return ProfileResponse(
        session_id=state.session_id, confidence=state.confidence, style_prefs=state.style_prefs,
        all_style_prefs=state.all_style_prefs,
        liked_post_ids=state.liked_post_ids, engagement_centroid=centroid, n_interactions=state.step_count,
    )


async def _save_profile(state: SessionStateV2, corpus: pd.DataFrame, embeddings):
    profiles_dir = Path(__file__).parent.parent / PROFILES_DIR
    profiles_dir.mkdir(parents=True, exist_ok=True)

    pid_to_emb = {}
    if embeddings is not None:
        for i, row in corpus.iterrows():
            pid_to_emb[str(row["post_id"])] = embeddings[i]
    centroid = compute_engagement_centroid(state.liked_post_ids, pid_to_emb)

    profile_data = {
        "session_id": state.session_id,
        "confidence": state.confidence,
        "style_prefs": state.style_prefs,
        "liked_post_ids": state.liked_post_ids,
        "engagement_centroid": centroid,
        "n_interactions": state.step_count,
        "history": state.history,
    }
    with open(profiles_dir / f"{state.session_id}_v2.json", "w") as f:
        json.dump(profile_data, f, indent=2)


# ── Candidate-pool selection (used by the eval endpoints) ──────────
def _select_candidate_pool(corpus: pd.DataFrame, state: SessionStateV2,
                            min_needed: int = 1, extra_seen: set | None = None):
    """
    Same selection logic used everywhere a post needs to be picked for
    transformation: engaged topics (confidence > 0.3) with a fallback to all
    topics, excluding anything already seen (liked, shown during seeding, or
    passed in via extra_seen -- e.g. posts already used in an eval trial).
    Falls back to the whole unseen corpus if the engaged-topic pool has
    fewer than min_needed posts.
    """
    topic_confs = state.confidence.get("topics", {})
    engaged = [t for t, c in topic_confs.items() if c > 0.3]
    if not engaged:
        engaged = corpus["topic_name"].unique().tolist()

    seen = set(state.liked_post_ids + [
        pid for h in state.history for pid in h.get("shown", [])
    ])
    if extra_seen:
        seen |= extra_seen

    candidates = corpus[
        (corpus["topic_name"].isin(engaged)) &
        (~corpus["post_id"].isin(seen))
    ]
    if len(candidates) < min_needed:
        candidates = corpus[~corpus["post_id"].isin(seen)]
    return candidates


# ── Eval endpoint: blind A/B of one post at a time, same selection/direction
# logic as /api/feed, dev-mode reveals which side is the rewrite + full
# verification detail. ──────────────────────────────────────────────
class EvalNextRequest(BaseModel):
    session_id: str


class EvalChoiceRequest(BaseModel):
    session_id: str
    post_id: str
    chosen_side: str  # "left" or "right"


# Fixed number of A/B trials per evaluation session -- after this many the
# session ends and /api/eval/next returns a "done" signal instead of a trial.
# This is the default; a specific session can override it via
# state.eval_trial_count (see SHORT_VARIANT_EVAL_TRIAL_COUNT above) -- every
# site below reads state.eval_trial_count, not this constant, so this exists
# purely as the value SessionStateV2.eval_trial_count defaults to.
EVAL_TRIAL_COUNT = 10

# How many ready-to-serve trials we try to keep pre-generated per session, so
# most /api/eval/next calls just pop from the queue instead of waiting live.
EVAL_QUEUE_TARGET_SIZE = 4

# How many distinct candidate posts to attempt concurrently per batch. Each
# post itself fires up to max_retries (2, passed explicitly at the eval
# trial call sites -- see _generate_one_eval_trial et al.) concurrent LLM
# calls inside transform_post_v2, so a batch of 3 means up to 6 concurrent
# LLM calls -- enough to meaningfully cut latency on a bad run of posts
# without piling on too much concurrent API load. There's no longer a cap
# on how many batches get tried in total (see _generate_one_eval_trial's
# docstring) -- only on how many run concurrently at once.
CONCURRENT_POSTS_PER_BATCH = 3

# Bidirectional manipulation check (see bt_preference_model memory): rather
# than only testing "does moving toward preference win," a couple of trials
# instead take a post the user already liked and deliberately degrade it
# AWAY from their preference, to see if they now prefer the untouched
# original. If personalization is real, these should flip the usual result.
ANTI_PREFERENCE_TARGET_COUNT = 2
# Prefer to attempt one around the middle and one near the end, spread out
# rather than clustered -- but this is just when we'd *like* to try, not a
# hard requirement (see _generate_next_trial: it keeps retrying on later
# calls if an attempt here fails, and forces it once slots start running out).
ANTI_PREFERENCE_PREFERRED_ATTEMPT_POINTS = {4, 8}

# Mirror check: a couple of trials instead take a post the user explicitly
# passed over during seeding (shown but not chosen) and push it TOWARD their
# preference -- the normal direction -- to see whether the rewrite can
# "rescue" a rejected post: do they now prefer the improved version over the
# one they originally skipped? Unlike the anti-preference direction, this
# uses the exact same generation direction as regular trials, so the normal
# gate (axis_ratio=1.5) applies with no relaxation.
RESCUE_TARGET_COUNT = 2
# Spread apart from ANTI_PREFERENCE_PREFERRED_ATTEMPT_POINTS so the 10 slots
# roughly interleave: 1,2 normal, 3 rescue, 4 normal, 5 anti, 6 normal,
# 7 rescue, 8 normal, 9 anti, 10 normal.
RESCUE_PREFERRED_ATTEMPT_POINTS = {2, 6}


def _build_trial(state, row, result, trial_kind: str = "toward_preference") -> dict:
    rewrite_is_left = random.random() < 0.5
    original_text = row["text"]
    rewritten_text = result["rewritten_text"]
    return {
        "session_id": state.session_id,
        "post_id": str(row["post_id"]),
        "topic_name": row["topic_name"],
        "trial_kind": trial_kind,  # "toward_preference" (default), "anti_preference", or "rescue"
        "left_text": rewritten_text if rewrite_is_left else original_text,
        "right_text": original_text if rewrite_is_left else rewritten_text,
        # dev-mode fields -- frontend decides whether to display these
        "rewrite_side": "left" if rewrite_is_left else "right",
        "rewrite_succeeded": not result["used_original"],
        "deltas": result["deltas"],
        "changes_made": result["changes_made"],
        "style_verification": result.get("style_verification"),
        "attempts": result.get("attempts"),
        "attempts_log": result.get("attempts_log"),
    }


async def _generate_one_eval_trial(corpus, state) -> dict | None:
    """
    Try candidate posts in concurrent batches (CONCURRENT_POSTS_PER_BATCH at a
    time) rather than one at a time, so a post that ends up getting rejected
    doesn't force the next post to wait behind it. NO fixed cap on how many
    posts get tried -- keeps pulling fresh batches until the real candidate
    pool (topic-scoped, or the whole corpus via _select_candidate_pool's own
    fallback) is genuinely exhausted. A fixed try-count budget was tried
    first and repeatedly proved too small under real verification failure
    rates (~50%+ per post under some preference profiles), surfacing "no
    posts available" to real users even after being raised from 12 to 18 --
    the actual candidate pool is almost always hundreds of posts, so looping
    until it's truly empty is both more correct and, in practice, hits the
    genuine-exhaustion case far less often than any fixed number could.

    Each post gets 2 rewrite attempts (transform_post_v2's max_retries=2,
    down from the default 3) before being marked failed for THIS call and
    moving on -- tried_this_call (local, not session state) makes sure a
    failed post isn't resampled again within the same call, which is what
    guarantees this loop actually terminates once the pool is exhausted
    rather than spinning on the same few posts forever. Like before, a
    failed post is only excluded for this call (via tried_this_call here,
    same idea as eval_in_flight_post_ids), not permanently -- it goes back
    into the pool for a future call, since it may still succeed for a
    different axis or on a different attempt. Only an actual success goes
    into the permanent eval_seen_post_ids.
    """
    tried_this_call = set()
    while True:
        candidates = _select_candidate_pool(
            corpus, state, min_needed=1,
            extra_seen=set(state.eval_seen_post_ids) | state.eval_in_flight_post_ids | tried_this_call)
        if candidates.empty:
            return None  # genuinely exhausted -- topic pool AND corpus-wide fallback both tried

        batch_size = min(CONCURRENT_POSTS_PER_BATCH, len(candidates))
        rows = [row for _, row in candidates.sample(batch_size).iterrows()]
        # Reserve these posts *before* the (slow) transform calls so a
        # parallel background-fill task never samples the same rows.
        for row in rows:
            pid = str(row["post_id"])
            state.eval_in_flight_post_ids.add(pid)
            tried_this_call.add(pid)

        async def _attempt(row):
            post_axes = json.loads(row["axes_json"]) if pd.notna(row.get("axes_json")) else {}
            result = await transform_post_v2(row["text"], post_axes, state.style_prefs,
                                              orig_post_id=str(row["post_id"]),
                                              axis_use_counts=state.axis_use_counts,
                                              max_retries=2)
            return row, result

        batch_results = await asyncio.gather(*[_attempt(row) for row in rows])
        # Resolve every post in the batch (release from in-flight) BEFORE
        # deciding what to return -- returning early partway through this
        # loop would leave any later post in the batch stuck in
        # eval_in_flight_post_ids forever, permanently excluding it even
        # though its attempt never actually failed or succeeded, it just
        # never got resolved.
        chosen = None
        for row, result in batch_results:
            state.eval_in_flight_post_ids.discard(str(row["post_id"]))
            if not result["used_original"] and chosen is None:
                state.eval_seen_post_ids.append(str(row["post_id"]))
                chosen = _build_trial(state, row, result)
        if chosen is not None:
            return chosen
        # Whole batch rejected (or no qualifying axis gap) -- tried_this_call
        # keeps them out of the NEXT sample within this call, so the loop
        # makes real progress toward the pool actually running out.


async def _generate_one_rescue_trial(corpus, state) -> dict | None:
    """
    Rescue check: mirror of the anti-preference trial with the opposite
    source pool and direction. Takes a post the user explicitly passed over
    during seeding (shown but not chosen, state.disliked_post_ids) and
    rewrites it TOWARD their preference -- the normal direction -- to test
    whether the rewrite can flip their judgment: do they now prefer the
    improved version over the one they originally skipped? Since this is
    the same generation direction as regular trials, the normal gate
    (axis_ratio=1.5, the transform_post_v2 default) applies unmodified --
    no relaxation needed here, unlike the anti-preference direction.
    """
    disliked_ids = list(dict.fromkeys(state.disliked_post_ids))  # de-dupe, keep order
    disliked_ids = [pid for pid in disliked_ids
                    if pid not in state.eval_seen_post_ids and pid not in state.eval_in_flight_post_ids]
    if not disliked_ids:
        return None
    random.shuffle(disliked_ids)

    # No fixed try-count cap -- keeps consuming disliked_ids until the whole
    # list runs out (see _generate_one_eval_trial's docstring for why a
    # fixed budget proved unreliable). This list is naturally finite and
    # shrinks every iteration via slicing below, so the loop always
    # terminates once every disliked post has been tried.
    while disliked_ids:
        batch_ids = disliked_ids[:CONCURRENT_POSTS_PER_BATCH]
        disliked_ids = disliked_ids[CONCURRENT_POSTS_PER_BATCH:]

        rows = []
        for pid in batch_ids:
            match = corpus[corpus["post_id"] == str(pid)]
            if not match.empty:
                rows.append(match.iloc[0])
        if not rows:
            continue
        # Reserved in-flight for this attempt only -- see
        # _generate_one_eval_trial's docstring for why a failed post isn't
        # permanently excluded here.
        for row in rows:
            state.eval_in_flight_post_ids.add(str(row["post_id"]))

        async def _attempt(row):
            post_axes = json.loads(row["axes_json"]) if pd.notna(row.get("axes_json")) else {}
            result = await transform_post_v2(row["text"], post_axes, state.style_prefs,
                                              orig_post_id=str(row["post_id"]),
                                              axis_use_counts=state.axis_use_counts,
                                              max_retries=2)
            return row, result

        batch_results = await asyncio.gather(*[_attempt(row) for row in rows])
        chosen = None
        for row, result in batch_results:
            state.eval_in_flight_post_ids.discard(str(row["post_id"]))
            if not result["used_original"] and chosen is None:
                state.eval_seen_post_ids.append(str(row["post_id"]))
                chosen = _build_trial(state, row, result, trial_kind="rescue")
        if chosen is not None:
            return chosen
        # Whole batch rejected -- released back above, so the next batch of
        # fresh disliked posts (or a later call) can include them again.
    return None


# Verification threshold for anti-preference trials specifically -- lower
# than the main product's 1.2 (see verify_classifier.verify_rewrite's
# axis_ratio docstring). Empirically validated: asking the LLM to push an
# already-liked post AWAY from preference produces much larger collateral
# shifts than a normal toward-preference rewrite on the same corpus (median
# best-case ratio ~0.3-0.4 vs routinely >1.2 for toward-preference), so
# holding these to the same 1.2 bar made them pass on <10% of liked posts --
# not enough to reliably clear ANTI_PREFERENCE_TARGET_COUNT out of a liked-
# post pool that's typically only ~10-15 posts. 0.4 still requires the
# target axis to be a real, substantial contributor (not negligible), just
# not the dominant one the main product's rewrites are held to.
ANTI_PREFERENCE_AXIS_RATIO = 0.4

# How many candidate away-axes to try per liked post (not just the single
# biggest gap) -- different axes on the same post fail somewhat
# independently, so trying a few meaningfully increases the chance one
# clears the gate without needing more liked posts than are actually available.
ANTI_PREFERENCE_AXES_PER_POST = 3


async def _generate_one_anti_preference_trial(corpus, state) -> dict | None:
    """
    Bidirectional check: pick a post the user already liked during seeding
    (not yet used in an eval trial), push it AWAY from their learned
    preference instead of toward it (compute_anti_preference_deltas), and
    see if they now prefer the untouched original over the degraded
    rewrite. Tries every unused liked post (no fixed cap -- see
    _generate_one_eval_trial's docstring) and a few candidate axes per
    post, since the away direction passes the gate far less often than the
    normal toward direction -- see ANTI_PREFERENCE_AXIS_RATIO. Returns None
    if there are no unused liked posts left, or if nothing clears the gate
    across all of them -- callers should fall back to a normal trial in
    that case, not surface an error.
    """
    liked_ids = list(dict.fromkeys(state.liked_post_ids))  # de-dupe, keep order
    liked_ids = [pid for pid in liked_ids
                 if pid not in state.eval_seen_post_ids and pid not in state.eval_in_flight_post_ids]
    if not liked_ids:
        return None
    random.shuffle(liked_ids)

    # No fixed try-count cap -- keeps consuming liked_ids until the whole
    # list runs out. Failed posts are released back (eval_in_flight_post_ids,
    # not eval_seen_post_ids -- see _generate_one_eval_trial's docstring)
    # rather than permanently consumed, so a post that doesn't clear the
    # gate for one away-axis can still be tried again later for a different
    # one -- that fix is also what makes being exhaustive here safe (an
    # earlier version capped this specifically to avoid one failed call
    # burning the whole liked-post pool, back when failures WERE permanent).
    while liked_ids:
        batch_ids = liked_ids[:CONCURRENT_POSTS_PER_BATCH]
        liked_ids = liked_ids[CONCURRENT_POSTS_PER_BATCH:]

        rows = []
        for pid in batch_ids:
            match = corpus[corpus["post_id"] == str(pid)]
            if not match.empty:
                rows.append(match.iloc[0])
        if not rows:
            continue
        for row in rows:
            state.eval_in_flight_post_ids.add(str(row["post_id"]))

        async def _attempt(row):
            post_axes = json.loads(row["axes_json"]) if pd.notna(row.get("axes_json")) else {}
            candidates = compute_anti_preference_deltas(
                post_axes, state.style_prefs, max_axes=ANTI_PREFERENCE_AXES_PER_POST,
                axis_use_counts=state.axis_use_counts,
            )
            for axis_name, delta in candidates.items():
                result = await transform_post_v2(
                    row["text"], post_axes, state.style_prefs,
                    orig_post_id=str(row["post_id"]), deltas_override={axis_name: delta},
                    axis_ratio=ANTI_PREFERENCE_AXIS_RATIO, max_retries=2,
                )
                if not result["used_original"]:
                    return row, result
            return row, {"used_original": True, "deltas": {}, "changes_made": "No candidate axis cleared the gate."}

        batch_results = await asyncio.gather(*[_attempt(row) for row in rows])
        chosen = None
        for row, result in batch_results:
            state.eval_in_flight_post_ids.discard(str(row["post_id"]))
            if not result["used_original"] and chosen is None:
                state.eval_seen_post_ids.append(str(row["post_id"]))
                chosen = _build_trial(state, row, result, trial_kind="anti_preference")
        if chosen is not None:
            return chosen
    return None


async def _generate_next_trial(corpus, state) -> dict | None:
    """
    Decides whether the next trial to generate should be one of the two
    manipulation-check trial types (anti-preference or rescue) or a normal
    toward-preference trial.

    Tries each at a preferred spread-out point first. If an attempt there
    fails (e.g. no liked/disliked posts qualified yet, or the rewrite never
    cleared the gate), that failure is remembered (state.
    anti_preference_attempt_failed / rescue_attempt_failed) and the type is
    retried on EVERY subsequent trial generation from then on -- not just
    once remaining slots get critically low. A slot-count-based "only force
    it right at the end" version of this was tried first and empirically
    still missed the guaranteed count in ~40% of a 5-persona synthetic test
    (see scratchpad/synthetic_persona_test.py) -- retrying immediately after
    any failure, rather than waiting, is what actually closes the gap.

    If state.eval_all_standard is set (the shortened Chinese demo variant --
    see SHORT_VARIANT_* / /api/session/start), anti-preference and rescue
    are never attempted at all: every trial generated is a normal
    toward-preference trial, straight through to _generate_one_eval_trial
    below.
    """
    if state.eval_all_standard:
        trial = await _generate_one_eval_trial(corpus, state)
        if trial is not None:
            state.eval_trials_generated += 1
        return trial

    anti_needed = ANTI_PREFERENCE_TARGET_COUNT - state.anti_preference_generated
    rescue_needed = RESCUE_TARGET_COUNT - state.rescue_generated

    attempt_anti = anti_needed > 0 and (
        state.eval_trials_generated in ANTI_PREFERENCE_PREFERRED_ATTEMPT_POINTS
        or state.anti_preference_attempt_failed
    )
    attempt_rescue = rescue_needed > 0 and (
        state.eval_trials_generated in RESCUE_PREFERRED_ATTEMPT_POINTS
        or state.rescue_attempt_failed
    )

    attempts = []
    if attempt_anti:
        attempts.append(("anti", anti_needed, _generate_one_anti_preference_trial))
    if attempt_rescue:
        attempts.append(("rescue", rescue_needed, _generate_one_rescue_trial))
    # If both are due this call, try whichever has the larger remaining need first.
    attempts.sort(key=lambda a: -a[1])

    for kind, _, generate_fn in attempts:
        trial = await generate_fn(corpus, state)
        if trial is not None:
            if kind == "anti":
                state.anti_preference_generated += 1
                state.anti_preference_attempt_failed = False
            else:
                state.rescue_generated += 1
                state.rescue_attempt_failed = False
            state.eval_trials_generated += 1
            return trial
        # Didn't qualify this time -- mark it so every future call retries,
        # then fall through to the next attempt (or a normal trial) this call.
        if kind == "anti":
            state.anti_preference_attempt_failed = True
        else:
            state.rescue_attempt_failed = True

    trial = await _generate_one_eval_trial(corpus, state)
    if trial is not None:
        state.eval_trials_generated += 1
    return trial


async def _fill_eval_queue(session_id: str):
    """Background task: top the session's eval_queue back up to target size."""
    state = _sessions_v2.get(session_id)
    if not state or not state.style_prefs or state.eval_queue_filling:
        return
    state.eval_queue_filling = True
    try:
        corpus, _, _, _ = get_data()
        while True:
            # Don't generate more trials than the session could ever show --
            # avoids burning real LLM calls on trials past the fixed budget.
            remaining_budget = state.eval_trial_count - len(state.eval_responses) - len(state.eval_queue)
            if remaining_budget <= 0 or len(state.eval_queue) >= EVAL_QUEUE_TARGET_SIZE:
                break
            trial = await _generate_next_trial(corpus, state)
            if trial is None:
                break  # corpus exhausted or a bad run -- stop, don't loop forever
            state.eval_queue.append(trial)
    finally:
        state.eval_queue_filling = False


@app.post("/api/eval/prewarm")
async def eval_prewarm(req: EvalNextRequest, background_tasks: BackgroundTasks):
    """
    Kick off queue generation in the background -- called right when the user
    clicks "Start Evaluation", so by the time eval_v2.html loads, a trial (or
    two) is often already sitting in the queue instead of making them wait.
    """
    state = _sessions_v2.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    if not state.style_prefs:
        raise HTTPException(400, "Profile not ready — complete seeding first")
    background_tasks.add_task(_fill_eval_queue, req.session_id)
    return {"ok": True}


@app.post("/api/eval/next")
async def eval_next(req: EvalNextRequest, background_tasks: BackgroundTasks):
    state = _sessions_v2.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")
    if not state.style_prefs:
        raise HTTPException(400, "Profile not ready — complete seeding first")

    if len(state.eval_responses) >= state.eval_trial_count:
        return {
            "done": True,
            "trials_completed": len(state.eval_responses),
            "trials_total": state.eval_trial_count,
        }

    if state.eval_queue:
        trial = state.eval_queue.pop(0)
    else:
        corpus, _, _, _ = get_data()
        # A single call already represents an exhaustive search: each trial
        # generator (_generate_one_eval_trial etc.) now loops internally
        # until the real candidate pool is genuinely empty, rather than
        # giving up after a fixed try-count -- see their docstrings. So
        # unlike before, a None result here isn't "unlucky, try a fresh
        # batch" -- it means this session's whole eligible pool (topic-
        # scoped, then corpus-wide fallback) has actually been exhausted.
        trial = await _generate_next_trial(corpus, state)
        if trial is None:
            raise HTTPException(
                400,
                "No more posts available to compare right now. Please try again shortly.",
            )

    # Keep the served trial around so /api/eval/choice can look up which side
    # was the rewrite and which axis it targeted -- the response only echoes
    # back post_id + chosen_side, it doesn't know the ground truth itself.
    state.eval_served_trials[trial["post_id"]] = trial

    # Top the queue back up in the background so the *next* call is instant.
    background_tasks.add_task(_fill_eval_queue, req.session_id)

    return {
        **trial,
        "done": False,
        "trials_completed": len(state.eval_responses),
        "trials_total": state.eval_trial_count,
    }


@app.post("/api/eval/choice")
async def eval_choice(req: EvalChoiceRequest):
    state = _sessions_v2.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    trial = state.eval_served_trials.pop(req.post_id, None)
    if not trial:
        raise HTTPException(404, "Trial not found — it may have already been recorded")

    chose_rewrite = req.chosen_side == trial["rewrite_side"]
    target_axes = list(trial["deltas"].keys())
    # Original/rewritten text derived from left/right + which side is the
    # rewrite, rather than stored separately in _build_trial -- avoids
    # duplicating the same two strings under a third pair of keys.
    original_text = trial["right_text"] if trial["rewrite_side"] == "left" else trial["left_text"]
    rewritten_text = trial["left_text"] if trial["rewrite_side"] == "left" else trial["right_text"]
    state.eval_responses.append({
        "post_id": req.post_id,
        "topic_name": trial["topic_name"],
        "trial_kind": trial.get("trial_kind", "toward_preference"),
        "chose_rewrite": chose_rewrite,
        "target_axes": target_axes,
        # Full trial detail, captured here because state.eval_served_trials
        # gets discarded (popped above) right after this -- without copying
        # it into eval_responses now, it's gone the moment the choice is
        # recorded and never reaches the auto-logged session record.
        "chosen_side": req.chosen_side,
        "deltas": trial["deltas"],
        "original_text": original_text,
        "rewritten_text": rewritten_text,
        "rewrite_succeeded": trial.get("rewrite_succeeded"),
        "style_verification": trial.get("style_verification"),
        "attempts": trial.get("attempts"),
        "attempts_log": trial.get("attempts_log"),
    })

    # Log automatically the moment this session crosses into completion --
    # fires exactly once per session, since eval_responses only grows by one
    # per call. This is what makes results collection automatic for anyone
    # who completes the eval via the shared link, no manual export needed.
    if len(state.eval_responses) == state.eval_trial_count:
        _log_completed_eval(state)

    return {
        "ok": True,
        "trials_completed": len(state.eval_responses),
        "trials_total": state.eval_trial_count,
    }


def _compute_eval_stats(responses: list[dict]) -> dict | None:
    """
    Shared by the per-session results endpoint and the automatic
    all-completions log, so both always agree on the same numbers.
    """
    if not responses:
        return None

    from collections import defaultdict

    def _summarize(rs: list[dict]) -> dict | None:
        if not rs:
            return None
        by_axis = defaultdict(lambda: {"wins": 0, "total": 0})
        for r in rs:
            for ax in r["target_axes"]:
                by_axis[ax]["total"] += 1
                if r["chose_rewrite"]:
                    by_axis[ax]["wins"] += 1
        return {
            "n": len(rs),
            "rewrite_win_rate": sum(1 for r in rs if r["chose_rewrite"]) / len(rs),
            "by_axis": {
                ax: {"win_rate": v["wins"] / v["total"] if v["total"] else 0, **v}
                for ax, v in by_axis.items()
            },
        }

    toward = [r for r in responses if r.get("trial_kind", "toward_preference") == "toward_preference"]
    anti = [r for r in responses if r.get("trial_kind") == "anti_preference"]
    rescue = [r for r in responses if r.get("trial_kind") == "rescue"]

    toward_summary = _summarize(toward)
    anti_summary = _summarize(anti)
    rescue_summary = _summarize(rescue)
    if anti_summary is not None:
        # For anti-preference trials, "success" is the OPPOSITE of the usual
        # win rate: the rewrite was deliberately degraded away from
        # preference, so a working system should make people prefer the
        # untouched original more often here, not the rewrite.
        anti_summary["original_preferred_rate"] = 1 - anti_summary["rewrite_win_rate"]
    # rescue_summary needs no inversion -- "chose_rewrite" is still the
    # hoped-for outcome here (did the improved version win), same semantics
    # as toward_preference, just drawn from posts the user originally skipped.

    return {
        "total_trials": len(responses),
        # Kept for any old consumers -- blends all trial kinds, which isn't
        # meaningful on its own once anti-preference trials are mixed in;
        # prefer toward_preference/anti_preference/rescue below.
        "overall_win_rate": sum(1 for r in responses if r["chose_rewrite"]) / len(responses),
        "toward_preference": toward_summary,
        "anti_preference": anti_summary,
        "rescue": rescue_summary,
    }


# Where every completed eval session gets automatically logged, so results
# from anyone who follows the link are collected without needing them to
# manually export or share anything. Append-only JSONL, one line per
# completed session.
EVAL_RESULTS_LOG_FILE = Path(__file__).parent.parent / "data" / "eval_results_log.jsonl"


def _log_completed_eval(state: SessionStateV2) -> None:
    import datetime
    stats = _compute_eval_stats(state.eval_responses)
    if stats is None:
        return
    record = {
        "session_id": state.session_id,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "style_prefs": state.style_prefs,
        "stats": stats,
        # Full per-trial detail (post_id, choice, original/rewritten text,
        # verification metadata) -- not just the aggregate stats above.
        "trials": state.eval_responses,
    }
    EVAL_RESULTS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_LOG_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/api/eval/results/{session_id}")
async def eval_results(session_id: str):
    state = _sessions_v2.get(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    stats = _compute_eval_stats(state.eval_responses)
    if stats is None:
        return {"message": "No responses yet", "responses": []}
    return {"stats": stats, "responses": state.eval_responses}


@app.get("/api/eval/all-results")
async def eval_all_results():
    """
    Every eval session that's ever reached completion, across anyone who's
    used the link -- read straight from the append-only log written by
    _log_completed_eval(). Newest first.
    """
    if not EVAL_RESULTS_LOG_FILE.exists():
        return {"results": []}
    records = []
    with open(EVAL_RESULTS_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.reverse()
    return {"results": records}


@app.delete("/api/eval/all-results")
async def reset_eval_results():
    """Clear the auto-collected eval-results log entirely (dashboard reset)."""
    if EVAL_RESULTS_LOG_FILE.exists():
        EVAL_RESULTS_LOG_FILE.unlink()
    return {"ok": True}


class TranslateRequest(BaseModel):
    texts: list[str]
    target_language: str = "Simplified Chinese"


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    """
    Pure display-layer translation for the Chinese frontend (index_zh.html /
    eval_v2_zh.html) -- everything upstream of this (post selection, rewrite
    generation, the classifier + BERTScore verification gate) stays entirely
    in English, unchanged. The classifier is distilbert-base-uncased and
    BERTScore's backbone is roberta-large -- both English-only models that
    would produce meaningless scores on Chinese text, so verification always
    happens on the original English text; translation is only ever applied
    to what gets displayed, never fed back into the pipeline.
    """
    if not req.texts:
        return {"translations": []}

    system = (
        f"You translate social media posts into {req.target_language}. "
        "Preserve tone, register, and meaning as closely as possible -- "
        "casual stays casual, formal stays formal, jokes stay jokes. "
        "Do not add or remove content. Return JSON: "
        '{"translations": ["<translation 1>", "<translation 2>", ...]} '
        "in the exact same order and count as the input list."
    )
    user = json.dumps({"texts": req.texts}, ensure_ascii=False)

    result = await llm_json(system, user, model=LLM_MODEL_PROFILE)
    translations = result.get("translations", [])
    if len(translations) != len(req.texts):
        raise HTTPException(502, "Translation count mismatch — please retry.")
    return {"translations": translations}


@app.get("/health")
async def health():
    return {"status": "ok", "architecture": "v2 — no LLM profile, no SAE, classifier+BERTScore gate with retry"}


# ── Serve frontend (same static files as the main app) ────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
