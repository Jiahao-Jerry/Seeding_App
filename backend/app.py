"""
FastAPI backend for the seeding web app.
Serves posts, handles user interactions, manages sessions.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.axes import AXES
from config.settings import (
    LLM_API_KEY_ENV, LLM_MODEL_PROFILE,
    ANNOTATED_CORPUS_FILE, PAIRS_FILE, CROSS_TOPIC_PAIRS_FILE,
    SHELF_SIZE, PROFILES_DIR
)
from backend.session import (
    SessionState, decide_next_action, update_profile,
    check_stop_condition, compute_engagement_centroid
)

# Load env
load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="Adaptive Delivery Seeding")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def _warmup_qwen():
    """Load SAE weights and Qwen into memory at startup so the first
    SAE Verification click responds instantly instead of timing out."""
    import asyncio, threading
    def _load():
        try:
            from backend.sae_verify import load_sae, load_qwen
            load_sae()
            print("[startup] SAE weights loaded.")
            load_qwen()
            print("[startup] Qwen2.5-7B ready.")
        except Exception as e:
            print(f"[startup] Qwen warm-up failed: {e}")
    threading.Thread(target=_load, daemon=True).start()


# ── Data loading ─────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

_corpus: pd.DataFrame | None = None
_pairs: pd.DataFrame | None = None
_cross_pairs: pd.DataFrame | None = None
_embeddings: np.ndarray | None = None
_sae_acts: np.ndarray | None = None        # (9500, 128) SAE feature activations
_sae_pid_to_row: dict | None = None        # post_id str → row index in _sae_acts
_sae_ridge: dict | None = None             # {axis: (coef, intercept)} for profile projection
_sessions: dict[str, SessionState] = {}
_shown_posts_cache: dict[str, list[dict]] = {}


def get_data():
    global _corpus, _pairs, _cross_pairs, _embeddings, _sae_acts, _sae_pid_to_row, _sae_ridge
    if _corpus is None:
        corpus_path = Path(__file__).parent.parent / ANNOTATED_CORPUS_FILE
        pairs_path = Path(__file__).parent.parent / PAIRS_FILE
        cross_path = Path(__file__).parent.parent / CROSS_TOPIC_PAIRS_FILE
        emb_path = DATA_DIR / "corpus_embeddings.npy"

        _corpus = pd.read_parquet(corpus_path)
        _corpus["post_id"] = _corpus["post_id"].astype(str)
        _pairs = pd.read_parquet(pairs_path) if pairs_path.exists() else pd.DataFrame()
        _cross_pairs = pd.read_parquet(cross_path) if cross_path.exists() else pd.DataFrame()
        _embeddings = np.load(emb_path) if emb_path.exists() else None

        # SAE: feature activations (qwen24_knn_k25_l0004, 128 features)
        # Row order matches annotated_posts.parquet (same 9500-post corpus)
        sae_acts_path = DATA_DIR / "sae_activations.npy"
        ridge_path = DATA_DIR / "sae_ridge_models.npz"
        if sae_acts_path.exists() and _corpus is not None:
            _sae_acts = np.load(sae_acts_path).astype(np.float32)
            _sae_pid_to_row = {str(p): i for i, p in enumerate(_corpus["post_id"].astype(str))}
        if ridge_path.exists():
            npz = np.load(ridge_path)
            axes = ["reading_level", "concreteness", "narrativity", "hedging",
                    "tone", "warmth", "self_disclosure", "casualness", "humor"]
            _sae_ridge = {ax: (npz[f"{ax}_coef"], float(npz[f"{ax}_intercept"][0])) for ax in axes}

    return _corpus, _pairs, _cross_pairs, _embeddings


def compute_sae_prefs(liked_post_ids: list[str]) -> dict:
    """
    Compute style preferences from SAE fingerprint of liked posts.

    Mean the SAE activations of liked posts (128-dim), then project each axis
    through the pre-fitted Ridge weights to get a preference score in [0, 1].
    Returns {} if SAE data is not loaded or no liked posts have SAE rows.
    """
    if _sae_acts is None or _sae_pid_to_row is None or _sae_ridge is None:
        return {}
    rows = [_sae_pid_to_row[pid] for pid in liked_post_ids if pid in _sae_pid_to_row]
    if not rows:
        return {}
    mean_act = _sae_acts[rows].mean(axis=0)  # (128,)
    prefs = {}
    for ax, (coef, intercept) in _sae_ridge.items():
        score = float(np.dot(mean_act, coef) + intercept)
        prefs[ax] = round(float(np.clip(score, 0.0, 1.0)), 4)
    return prefs


# ── API Models ───────────────────────────────────────────────────
class StartSessionResponse(BaseModel):
    session_id: str
    action: dict
    posts: list[dict]


class InteractionRequest(BaseModel):
    session_id: str
    chosen_post_ids: list[str]
    shown_post_ids: list[str]  # ALL posts that were displayed


class InteractionResponse(BaseModel):
    session_id: str
    is_complete: bool
    profile: dict
    confidence: dict
    step: int
    action: dict | None = None
    posts: list[dict] | None = None


class ProfileResponse(BaseModel):
    session_id: str
    profile: dict
    confidence: dict
    liked_post_ids: list[str]
    engagement_centroid: list[float]
    n_interactions: int
    sae_prefs: dict = {}


# ── Helpers ──────────────────────────────────────────────────────
def post_to_api_dict(row: pd.Series) -> dict:
    """Convert a corpus row to the API representation."""
    result = {
        "post_id": str(row["post_id"]),
        "topic_name": row["topic_name"],
        "text": row["text"],
    }
    if pd.notna(row.get("axes_json")):
        result["axes"] = json.loads(row["axes_json"])
    return result


def get_shelf_posts(corpus: pd.DataFrame, action: dict, state: SessionState) -> list[dict]:
    """Select posts for a shelf (Mode A)."""
    topics = action.get("topics", corpus["topic_name"].unique().tolist())

    # Exclude already-seen posts
    seen = set(state.liked_post_ids + [
        pid for h in state.history for pid in h.get("shown", [])
    ])

    if action.get("scope") == "broad":
        # Try representatives first, then random
        candidates = corpus[
            (corpus["topic_name"].isin(topics)) &
            (~corpus["post_id"].isin(seen))
        ]
        # Prefer one per topic for variety
        sampled_indices = []
        for _, g in candidates.groupby("topic_name"):
            sampled_indices.append(g.sample(min(1, len(g))).index[0])
        per_topic = candidates.loc[sampled_indices]

        if len(per_topic) >= SHELF_SIZE:
            candidates = per_topic.sample(SHELF_SIZE)
        elif len(candidates) > SHELF_SIZE:
            candidates = candidates.sample(SHELF_SIZE)
    else:
        # Drill into engaged topics
        candidates = corpus[
            (corpus["topic_name"].isin(topics)) &
            (~corpus["post_id"].isin(seen))
        ]
        if len(candidates) > SHELF_SIZE:
            candidates = candidates.sample(SHELF_SIZE)

    return [post_to_api_dict(row) for _, row in candidates.iterrows()]


def get_pair_posts(corpus: pd.DataFrame, pairs: pd.DataFrame,
                   action: dict, state: SessionState,
                   cross_pairs: pd.DataFrame = None) -> list[dict]:
    """
    Select a contrastive pair (Mode B).
    Strategy: use cross-topic pairs first (obvious style differences),
    fall back to within-topic pairs for refinement.
    """
    target_axis = action["target_axis"]

    # Exclude posts already seen in any mode
    seen_posts = set()
    used_pairs = set()
    for h in state.history:
        shown = h.get("shown", [])
        for pid in shown:
            seen_posts.add(str(pid))
        if len(shown) == 2:
            used_pairs.add(tuple(sorted(shown)))

    def pair_is_fresh(high_id, low_id):
        h, l = str(high_id), str(low_id)
        key = tuple(sorted([h, l]))
        if key in used_pairs:
            return False
        # Prefer both posts unseen; accept one seen only as last resort
        return h not in seen_posts and l not in seen_posts

    def pair_is_acceptable(high_id, low_id):
        """Fallback: at least the pair combo hasn't been shown."""
        key = tuple(sorted([str(high_id), str(low_id)]))
        return key not in used_pairs

    best_pair = None

    # Priority 1: cross-topic pairs, both posts fresh
    if cross_pairs is not None and not cross_pairs.empty:
        ct_available = cross_pairs[cross_pairs["target_axis"] == target_axis].sort_values("score", ascending=False)
        for _, pair_row in ct_available.iterrows():
            if pair_is_fresh(pair_row["high_post_id"], pair_row["low_post_id"]):
                best_pair = pair_row
                break

    # Priority 2: within-topic pairs, both posts fresh
    if best_pair is None and not pairs.empty:
        wt_available = pairs[pairs["target_axis"] == target_axis].sort_values("score", ascending=False)
        for _, pair_row in wt_available.iterrows():
            if pair_is_fresh(pair_row["high_post_id"], pair_row["low_post_id"]):
                best_pair = pair_row
                break

    # Priority 3: cross-topic pairs, at least the combo is new (one post may repeat)
    if best_pair is None and cross_pairs is not None and not cross_pairs.empty:
        ct_available = cross_pairs[cross_pairs["target_axis"] == target_axis].sort_values("score", ascending=False)
        for _, pair_row in ct_available.iterrows():
            if pair_is_acceptable(pair_row["high_post_id"], pair_row["low_post_id"]):
                best_pair = pair_row
                break

    # Priority 4: within-topic pairs, at least the combo is new
    if best_pair is None and not pairs.empty:
        wt_available = pairs[pairs["target_axis"] == target_axis].sort_values("score", ascending=False)
        for _, pair_row in wt_available.iterrows():
            if pair_is_acceptable(pair_row["high_post_id"], pair_row["low_post_id"]):
                best_pair = pair_row
                break

    # Final fallback: random unseen posts from corpus
    if best_pair is None:
        unseen = corpus[~corpus["post_id"].isin(seen_posts)]
        pool = unseen if len(unseen) >= 2 else corpus
        sampled = pool.sample(min(2, len(pool)))
        posts = [post_to_api_dict(row) for _, row in sampled.iterrows()]
        random.shuffle(posts)
        return posts

    # Look up actual posts
    post_a = corpus[corpus["post_id"] == str(best_pair["high_post_id"])]
    post_b = corpus[corpus["post_id"] == str(best_pair["low_post_id"])]
    if post_a.empty or post_b.empty:
        sampled = corpus.sample(2)
        posts = [post_to_api_dict(row) for _, row in sampled.iterrows()]
        random.shuffle(posts)
        return posts

    posts = [post_to_api_dict(post_a.iloc[0]), post_to_api_dict(post_b.iloc[0])]
    random.shuffle(posts)
    return posts


# ── Endpoints ────────────────────────────────────────────────────
@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session():
    """Start a new seeding session."""
    corpus, pairs, cross_pairs, _ = get_data()

    state = SessionState()
    _sessions[state.session_id] = state

    topics = corpus["topic_name"].unique().tolist()
    action = decide_next_action(state, topics)
    state.last_action = action
    posts = get_shelf_posts(corpus, action, state)

    return StartSessionResponse(
        session_id=state.session_id,
        action=action,
        posts=posts,
    )


@app.post("/api/session/interact", response_model=InteractionResponse)
async def interact(req: InteractionRequest):
    """Process user interaction and return next action."""
    corpus, pairs, cross_pairs, embeddings = get_data()

    state = _sessions.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    # Get ALL shown posts (chosen + skipped) for the LLM
    shown_mask = corpus["post_id"].isin(req.shown_post_ids)
    shown_posts = [post_to_api_dict(row) for _, row in corpus[shown_mask].iterrows()]

    # Update profile via LLM (pass the action the user just responded to)
    valid_topics = set(corpus["topic_name"].unique())
    state = await update_profile(state, shown_posts, req.chosen_post_ids,
                                 action=state.last_action, valid_topics=valid_topics)

    # Update SAE-based style preferences from liked posts so far
    state.sae_prefs = compute_sae_prefs(state.liked_post_ids)

    if state.is_complete:
        # Save profile
        await _save_profile(state, corpus, embeddings)
        return InteractionResponse(
            session_id=state.session_id,
            is_complete=True,
            profile=state.profile,
            confidence=state.confidence,
            step=state.step_count,
        )

    # Decide next action
    topics = corpus["topic_name"].unique().tolist()
    action = decide_next_action(state, topics)
    state.last_action = action

    if action["mode"] == "shelf":
        posts = get_shelf_posts(corpus, action, state)
    else:
        # Validate topic exists in corpus; if LLM returned a variant, fall back
        valid_topics = set(corpus["topic_name"].unique())
        if action.get("topic") not in valid_topics:
            action["topic"] = next(
                (t for t in state.confidence.get("topics", {}) if t in valid_topics),
                list(valid_topics)[0]
            )
        posts = get_pair_posts(corpus, pairs, action, state, cross_pairs=cross_pairs)

    return InteractionResponse(
        session_id=state.session_id,
        is_complete=False,
        profile=state.profile,
        confidence=state.confidence,
        step=state.step_count,
        action=action,
        posts=posts,
    )


@app.get("/api/session/{session_id}/profile", response_model=ProfileResponse)
async def get_profile(session_id: str):
    """Get the final user profile."""
    corpus, _, _, embeddings = get_data()
    state = _sessions.get(session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    # Compute engagement centroid
    pid_to_emb = {}
    if embeddings is not None:
        for i, row in corpus.iterrows():
            pid_to_emb[str(row["post_id"])] = embeddings[i]

    centroid = compute_engagement_centroid(state.liked_post_ids, pid_to_emb)

    return ProfileResponse(
        session_id=state.session_id,
        profile=state.profile,
        confidence=state.confidence,
        liked_post_ids=state.liked_post_ids,
        engagement_centroid=centroid,
        n_interactions=state.step_count,
        sae_prefs=state.sae_prefs,
    )


async def _save_profile(state: SessionState, corpus: pd.DataFrame, embeddings):
    """Persist the user profile to disk."""
    profiles_dir = Path(__file__).parent.parent / PROFILES_DIR
    profiles_dir.mkdir(parents=True, exist_ok=True)

    pid_to_emb = {}
    if embeddings is not None:
        for i, row in corpus.iterrows():
            pid_to_emb[str(row["post_id"])] = embeddings[i]
    centroid = compute_engagement_centroid(state.liked_post_ids, pid_to_emb)

    profile_data = {
        "session_id": state.session_id,
        "topics_prose": state.profile["topics_prose"],
        "style_prose": state.profile["style_prose"],
        "confidence": state.confidence,
        "sae_prefs": state.sae_prefs,
        "liked_post_ids": state.liked_post_ids,
        "engagement_centroid": centroid,
        "n_interactions": state.step_count,
        "history": state.history,
    }
    with open(profiles_dir / f"{state.session_id}.json", "w") as f:
        json.dump(profile_data, f, indent=2)


# ── Transform endpoint ────────────────────────────────────────────

class TransformRequest(BaseModel):
    post_id: str | None = None
    text: str | None = None  # Either post_id (from corpus) or raw text
    session_id: str | None = None  # Use this session's profile
    user_prefs: dict | None = None  # Or provide prefs directly
    verify: bool = True


class TransformResponse(BaseModel):
    original_text: str
    rewritten_text: str
    changes_made: str
    deltas: dict
    verification: dict | None = None
    used_original: bool
    additive_material_added: bool = False
    transform_confidence: float = 0.0


@app.post("/api/transform", response_model=TransformResponse)
async def transform_endpoint(req: TransformRequest):
    """Transform a post to match user delivery preferences."""
    from backend.transform import transform_post

    corpus, _, _, _ = get_data()

    # Get original post
    if req.post_id:
        post_row = corpus[corpus["post_id"] == req.post_id]
        if post_row.empty:
            raise HTTPException(404, "Post not found")
        post_row = post_row.iloc[0]
        original_text = post_row["text"]
        post_axes = json.loads(post_row["axes_json"])
    elif req.text:
        original_text = req.text
        # For raw text, we'd need to annotate it first — use empty axes
        post_axes = {}
    else:
        raise HTTPException(400, "Provide either post_id or text")

    # Get user preferences — SAE fingerprint takes priority over LLM-inferred confidence
    if req.session_id:
        state = _sessions.get(req.session_id)
        if not state:
            raise HTTPException(404, "Session not found")
        user_prefs = state.sae_prefs if state.sae_prefs else state.confidence.get("axes", {})
    elif req.user_prefs:
        user_prefs = req.user_prefs
    else:
        raise HTTPException(400, "Provide either session_id or user_prefs")

    result = await transform_post(
        original_text, post_axes, user_prefs,
        verify=req.verify,
        orig_post_id=req.post_id,
    )

    return TransformResponse(
        original_text=original_text,
        rewritten_text=result["rewritten_text"],
        changes_made=result["changes_made"],
        deltas=result["deltas"],
        verification=result.get("verification"),
        used_original=result["used_original"],
        additive_material_added=result.get("additive_material_added", False),
        transform_confidence=result.get("transform_confidence", 0.0),
    )


# ── Feed endpoint (post-seeding: show transformed posts) ─────────

class FeedRequest(BaseModel):
    session_id: str
    count: int = 5  # how many posts to transform


class FeedItem(BaseModel):
    post_id: str
    topic_name: str
    original_text: str
    transformed_text: str
    changes_made: str
    deltas: dict
    used_original: bool


@app.post("/api/feed")
async def get_feed(req: FeedRequest):
    """Get a feed of transformed posts based on the user's profile."""
    from backend.transform import transform_post

    corpus, _, _, _ = get_data()
    state = _sessions.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    # SAE fingerprint takes priority over LLM-inferred confidence
    user_prefs = state.sae_prefs if state.sae_prefs else state.confidence.get("axes", {})
    if not user_prefs:
        raise HTTPException(400, "Profile not ready — complete seeding first")

    # Pick posts from engaged topics that the user hasn't seen
    topic_confs = state.confidence.get("topics", {})
    engaged = [t for t, c in topic_confs.items() if c > 0.3]
    if not engaged:
        engaged = corpus["topic_name"].unique().tolist()

    seen = set(state.liked_post_ids + [
        pid for h in state.history for pid in h.get("shown", [])
    ])

    candidates = corpus[
        (corpus["topic_name"].isin(engaged)) &
        (~corpus["post_id"].isin(seen))
    ]
    if len(candidates) < req.count:
        candidates = corpus[~corpus["post_id"].isin(seen)]

    sample = candidates.sample(min(req.count, len(candidates)))

    # Transform each post
    feed_items = []
    for _, row in sample.iterrows():
        post_axes = json.loads(row["axes_json"]) if pd.notna(row.get("axes_json")) else {}
        result = await transform_post(
            row["text"], post_axes, user_prefs,
            verify=False,  # skip LLM substance check for feed speed
            orig_post_id=str(row["post_id"]),
        )
        feed_items.append({
            "post_id": str(row["post_id"]),
            "topic_name": row["topic_name"],
            "original_text": row["text"],
            "transformed_text": result["rewritten_text"],
            "changes_made": result["changes_made"],
            "deltas": result["deltas"],
            "used_original": result["used_original"],
        })

    return {"session_id": req.session_id, "items": feed_items}


# ── Evaluation endpoints ──────────────────────────────────────────

class EvalStartRequest(BaseModel):
    session_id: str


class EvalChoiceRequest(BaseModel):
    session_id: str
    trial_id: str
    chosen_side: str  # "left" or "right"


@app.post("/api/eval/start")
async def eval_start(req: EvalStartRequest):
    """Start an evaluation session: compute centroid, build trials, transform posts."""
    import asyncio
    from backend.evaluation import (
        compute_user_centroid, compute_preferred_axes, construct_trials
    )
    from backend.transform import transform_post
    from config.admin import get_settings

    settings = get_settings()

    state = _sessions.get(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found — complete seeding first")

    # Compute user preference centroid from liked posts
    liked_ids = state.liked_post_ids
    if not liked_ids:
        raise HTTPException(400, "No liked posts — complete seeding first")

    centroid = compute_user_centroid(liked_ids)
    user_prefs = compute_preferred_axes(liked_ids)

    seen_ids = set(str(x) for x in liked_ids + [
        pid for h in state.history for pid in h.get("shown", [])
    ])

    # Build trials using admin settings
    trials = construct_trials(
        centroid, user_prefs, seen_ids,
        n_type_a=settings.n_type_a,
        n_type_b=settings.n_type_b,
        n_type_c=settings.n_type_c,
        prefer_high_gap=settings.prefer_high_gap,
        diversify_topics=settings.diversify_topics,
    )

    # Load corpus for text lookup
    corpus, _, _, _ = get_data()

    # Prepare transform tasks (async parallel)
    async def process_trial(trial):
        left_row = corpus[corpus["post_id"].astype(str) == trial.left_post_id]
        right_row = corpus[corpus["post_id"].astype(str) == trial.right_post_id]
        if left_row.empty or right_row.empty:
            return None

        left_row = left_row.iloc[0]
        right_row = right_row.iloc[0]
        left_text = left_row["text"]
        right_text = right_row["text"]

        if trial.left_is_transformed:
            post_axes = json.loads(left_row["axes_json"]) if pd.notna(left_row.get("axes_json")) else {}
            result = await transform_post(
                left_text, post_axes, user_prefs,
                verify=settings.verify_transform,
                max_axes=settings.max_axes,
                orig_post_id=str(left_row["post_id"]),
            )
            left_text = result["rewritten_text"]

        if trial.right_is_transformed:
            post_axes = json.loads(right_row["axes_json"]) if pd.notna(right_row.get("axes_json")) else {}
            result = await transform_post(
                right_text, post_axes, user_prefs,
                verify=settings.verify_transform,
                max_axes=settings.max_axes,
                orig_post_id=str(right_row["post_id"]),
            )
            right_text = result["rewritten_text"]

        return {
            "trial_id": trial.trial_id,
            "trial_type": trial.trial_type,
            "left_text": left_text,
            "right_text": right_text,
            "_meta": trial.to_dict(),
        }

    # Run all transforms in parallel
    results = await asyncio.gather(*[process_trial(t) for t in trials])
    trial_data = [r for r in results if r is not None]

    # Store trials in session for later analysis
    state.eval_trials = trial_data
    state.eval_responses = []
    state.eval_user_prefs = user_prefs

    # Return only what the user sees (no metadata about which is transformed)
    visible = [{"trial_id": t["trial_id"], "left_text": t["left_text"],
                "right_text": t["right_text"]} for t in trial_data]

    return {
        "session_id": req.session_id,
        "total_trials": len(visible),
        "trials": visible,
    }


@app.post("/api/eval/respond")
async def eval_respond(req: EvalChoiceRequest):
    """Record user's choice for one trial."""
    from config.admin import get_settings

    state = _sessions.get(req.session_id)
    if not state or not hasattr(state, "eval_trials"):
        raise HTTPException(404, "No evaluation session found")

    # Find the trial (both meta and text)
    trial_entry = next(
        (t for t in state.eval_trials if t["trial_id"] == req.trial_id), None
    )
    if not trial_entry:
        raise HTTPException(404, f"Trial {req.trial_id} not found")

    trial_meta = trial_entry["_meta"]

    # Determine if user chose the transformed version
    if req.chosen_side == "left":
        chose_transformed = trial_meta["left_is_transformed"]
    else:
        chose_transformed = trial_meta["right_is_transformed"]

    response = {
        "trial_id": req.trial_id,
        "trial_type": trial_meta["trial_type"],
        "chosen_side": req.chosen_side,
        "chose_transformed": chose_transformed,
        "transform_axes": trial_meta["transform_axes"],
        "left_band": trial_meta["left_band"],
        "right_band": trial_meta["right_band"],
    }
    state.eval_responses.append(response)

    # Optionally update profile from this choice
    settings = get_settings()
    if settings.update_profile_from_eval:
        corpus, _, _, _ = get_data()
        valid_topics = set(corpus["topic_name"].unique())

        # Build shown_posts format for the profile updater
        left_id = trial_meta["left_post_id"]
        right_id = trial_meta["right_post_id"]
        shown_mask = corpus["post_id"].astype(str).isin([left_id, right_id])
        shown_posts = [post_to_api_dict(row) for _, row in corpus[shown_mask].iterrows()]

        chosen_id = left_id if req.chosen_side == "left" else right_id
        await update_profile(state, shown_posts, [chosen_id],
                             action={"mode": "pair", "target_axis": trial_meta["transform_axes"][0] if trial_meta["transform_axes"] else "tone"},
                             valid_topics=valid_topics)

    return {"recorded": True, "trials_completed": len(state.eval_responses)}


@app.get("/api/eval/results/{session_id}")
async def eval_results(session_id: str):
    """Get evaluation results + basic stats."""
    state = _sessions.get(session_id)
    if not state or not hasattr(state, "eval_responses"):
        raise HTTPException(404, "No evaluation data found")

    responses = state.eval_responses
    if not responses:
        return {"message": "No responses yet", "responses": []}

    # Compute win rates by trial type
    from collections import defaultdict
    by_type = defaultdict(lambda: {"wins": 0, "total": 0})
    by_axis = defaultdict(lambda: {"wins": 0, "total": 0})

    for r in responses:
        tt = r["trial_type"]
        by_type[tt]["total"] += 1
        if r["chose_transformed"]:
            by_type[tt]["wins"] += 1
        for ax in r["transform_axes"]:
            by_axis[ax]["total"] += 1
            if r["chose_transformed"]:
                by_axis[ax]["wins"] += 1

    stats = {
        "total_trials": len(responses),
        "overall_win_rate": sum(1 for r in responses if r["chose_transformed"]) / len(responses),
        "by_type": {k: {"win_rate": v["wins"] / v["total"] if v["total"] else 0, **v}
                    for k, v in by_type.items()},
        "by_axis": {k: {"win_rate": v["wins"] / v["total"] if v["total"] else 0, **v}
                    for k, v in by_axis.items()},
    }

    return {"stats": stats, "responses": responses}


# ── SAE rewrite verification ──────────────────────────────────────

class SAEVerifyRequest(BaseModel):
    original_text: str
    rewritten_text: str
    target_axes: list[str]
    orig_post_id: str | None = None


@app.post("/api/sae-verify")
async def sae_verify(req: SAEVerifyRequest):
    """
    SAE-based rewrite verification.
    Loads Qwen2.5-7B on first call (~14GB download if not cached).
    Runs Qwen in a thread pool to avoid blocking the event loop.
    """
    import asyncio
    from backend.sae_verify import verify_rewrite, load_sae

    load_sae()   # fast — just weights, no Qwen yet

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: verify_rewrite(
            req.original_text,
            req.rewritten_text,
            req.target_axes,
            req.orig_post_id,
        ),
    )
    return result


@app.get("/api/sae-verify/status")
async def sae_verify_status():
    """Check whether Qwen is loaded and SAE weights are ready."""
    from backend.sae_verify import _qwen_model, _sae
    return {
        "sae_loaded": _sae is not None,
        "qwen_loaded": _qwen_model is not None,
        "device": str(_get_device()) if _sae is not None else None,
    }


def _get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Admin settings API ────────────────────────────────────────────

@app.get("/api/admin/settings")
async def get_admin_settings():
    from config.admin import get_settings
    return get_settings().to_dict()


@app.post("/api/admin/settings")
async def update_admin_settings(updates: dict):
    from config.admin import update_settings
    settings = update_settings(**updates)
    return settings.to_dict()


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Serve frontend ───────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
