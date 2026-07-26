"""
No-LLM-profile session state: this is the "new architecture" variant --
topic tracking via simple frequency counting, style preferences EXCLUSIVELY
via the Bradley-Terry model (backend/bt_profile.py). No LLM call anywhere in
the profile-update path (no topics_prose/style_prose, no LLM-guessed
confidence numbers).

Reuses decide_next_action() and check_stop_condition() from backend/session.py
unmodified -- both only read state.confidence/state.step_count/state.history,
so they work identically on this duck-typed state object.
"""
import uuid
from dataclasses import dataclass, field

from config.axes import AXES
from config.settings import MAX_STEPS, EXPLORE_ROUNDS
from backend.bt_profile import (
    BTProfile, update_shelf, compute_bt_prefs, compute_all_preferred_values, AXIS_NAMES,
    MIN_CONFIDENCE_FOR_PREFS,
)

# How many chosen posts from a topic before its confidence saturates near 1.0.
TOPIC_CONFIDENCE_SATURATION = 5


@dataclass
class SessionStateV2:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: dict = field(default_factory=lambda: {
        "topics": {},
        "axes": {ax["name"]: 0.0 for ax in AXES},
    })
    topic_counts: dict = field(default_factory=dict)  # topic_name -> # chosen posts
    history: list = field(default_factory=list)
    liked_post_ids: list = field(default_factory=list)
    step_count: int = 0
    is_complete: bool = False
    last_action: dict = field(default_factory=dict)
    bt_profile: BTProfile = field(default_factory=BTProfile)
    style_prefs: dict = field(default_factory=dict)  # BT-derived, confidence-gated, top-4 + neutral-band filtered -- what rewriting actually uses
    all_style_prefs: dict = field(default_factory=dict)  # BT-derived, UNFILTERED preferred_value for every axis -- debug/dev display only, not used for rewriting
    eval_seen_post_ids: list = field(default_factory=list)  # posts whose rewrite actually SUCCEEDED and was shown in an eval trial -- permanently excluded from future sampling
    eval_in_flight_post_ids: set = field(default_factory=set)  # posts currently being attempted (reserved to prevent a concurrent background-fill task from double-sampling); released again -- whether the attempt succeeded or failed -- once it resolves, so a post that fails for one axis can still be tried later for a different one
    eval_queue: list = field(default_factory=list)  # pre-generated trials ready to serve instantly
    eval_queue_filling: bool = False  # guards against overlapping background fill tasks
    eval_served_trials: dict = field(default_factory=dict)  # post_id -> trial dict, so /api/eval/choice can look up rewrite_side/target axes
    eval_responses: list = field(default_factory=list)  # completed eval choices: {post_id, topic_name, chose_rewrite, target_axes, trial_kind}
    axis_use_counts: dict = field(default_factory=dict)  # running count of how often each axis has been picked as a rewrite target, for rotation (see compute_transform_deltas)
    eval_trials_generated: int = 0  # counts trials as they're generated (queued or live), used to schedule anti-preference/rescue trials
    anti_preference_generated: int = 0  # how many bidirectional anti-preference trials have actually been generated so far this session
    disliked_post_ids: list = field(default_factory=list)  # posts shown but NOT chosen during seeding (shelf or pair rounds)
    rescue_generated: int = 0  # how many "rescue" trials (disliked post pushed toward preference) have been generated so far this session
    anti_preference_attempt_failed: bool = False  # once True, retry anti-preference on every subsequent trial generation, not just at preferred points
    rescue_attempt_failed: bool = False  # once True, retry rescue on every subsequent trial generation, not just at preferred points

    # Per-session overrides, defaulting to match the standard flow -- lets a
    # specific variant (e.g. the shortened Chinese demo flow) run fewer
    # seeding rounds and/or a shorter, all-standard eval without changing
    # behavior for anyone else. Set at /api/session/start based on the
    # ?variant= query param; read by decide_next_action()/
    # check_stop_condition() (backend/session.py) and by the eval endpoints/
    # _generate_next_trial() (backend/app_v2.py) instead of their module
    # constants directly.
    max_steps: int = MAX_STEPS
    explore_rounds: int = EXPLORE_ROUNDS
    eval_trial_count: int = 10  # mirrors app_v2.EVAL_TRIAL_COUNT's default
    eval_all_standard: bool = False  # if True, every eval trial is toward_preference -- no anti_preference/rescue trials attempted at all
    # Confidence floor for an axis to enter style_prefs (see compute_bt_prefs).
    # A shortened seeding session (fewer pairwise comparisons) naturally
    # produces lower per-axis confidence across the board, so the standard
    # 0.3 floor can leave style_prefs with just 1 qualifying axis -- which
    # then forces every eval trial to target that one axis, making the
    # fixed per-call retry budget (see _generate_one_eval_trial) run out
    # much more easily than with the usual 3-9 qualifying axes. Lowered for
    # the short variant specifically (see SHORT_VARIANT_MIN_CONFIDENCE).
    style_prefs_min_confidence: float = MIN_CONFIDENCE_FOR_PREFS


def update_profile_v2(state: SessionStateV2, shown_posts: list[dict],
                       user_choices: list[str], action: dict = None) -> SessionStateV2:
    """No-LLM profile update. Topics: frequency counting. Style axes: Bradley-Terry only."""
    for p in shown_posts:
        if p["post_id"] in user_choices:
            t = p.get("topic_name")
            if t:
                state.topic_counts[t] = state.topic_counts.get(t, 0) + 1

    for t, count in state.topic_counts.items():
        state.confidence["topics"][t] = min(1.0, count / TOPIC_CONFIDENCE_SATURATION)

    update_shelf(state.bt_profile, shown_posts, user_choices)
    for ax in AXIS_NAMES:
        state.confidence["axes"][ax] = state.bt_profile.confidence(ax)
    state.style_prefs = compute_bt_prefs(state.bt_profile, min_confidence=state.style_prefs_min_confidence)
    state.all_style_prefs = compute_all_preferred_values(state.bt_profile)

    history_entry = {
        "step": state.step_count,
        "shown": [p["post_id"] for p in shown_posts],
        "chosen": user_choices,
        "confidence_after": {
            "topics": dict(state.confidence["topics"]),
            "axes": dict(state.confidence["axes"]),
        },
    }
    if action:
        history_entry["action_mode"] = action.get("mode")
        history_entry["target_topic"] = action.get("topic")
        history_entry["target_axis"] = action.get("target_axis")
    state.history.append(history_entry)

    state.liked_post_ids.extend(user_choices)
    state.disliked_post_ids.extend(
        p["post_id"] for p in shown_posts if p["post_id"] not in user_choices
    )
    state.step_count += 1

    from backend.session import check_stop_condition
    state.is_complete = check_stop_condition(state)

    return state
