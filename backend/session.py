"""
Seeding session: the core online loop.
Manages user state, calls LLM for profile updates, decides next action.
"""

import json
import uuid
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.axes import AXES
from config.settings import (
    MIN_TOPICS_CONFIDENT, CONFIDENCE_THRESHOLD, MAX_STEPS,
    SHELF_SIZE, LLM_MODEL_PROFILE
)
from backend.llm_helper import llm_json


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile: dict = field(default_factory=lambda: {
        "topics_prose": "",
        "style_prose": "",
    })
    confidence: dict = field(default_factory=lambda: {
        "topics": {},
        "axes": {ax["name"]: 0.0 for ax in AXES},
    })
    history: list = field(default_factory=list)
    liked_post_ids: list = field(default_factory=list)
    step_count: int = 0
    is_complete: bool = False
    last_action: dict = field(default_factory=dict)
    sae_prefs: dict = field(default_factory=dict)  # SAE-derived axis preferences (updated after each interaction)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "profile": self.profile,
            "confidence": self.confidence,
            "history": self.history,
            "liked_post_ids": self.liked_post_ids,
            "step_count": self.step_count,
            "is_complete": self.is_complete,
        }


def check_stop_condition(state: SessionState) -> bool:
    """Deterministic stop check."""
    if state.step_count >= MAX_STEPS:
        return True

    # Check if enough topics are confident
    topic_confs = state.confidence.get("topics", {})
    confident_topics = sum(
        1 for v in topic_confs.values() if v >= CONFIDENCE_THRESHOLD
    )
    if confident_topics < MIN_TOPICS_CONFIDENT:
        return False

    # Check if all axes are confident for engaged topics
    axis_confs = state.confidence.get("axes", {})
    all_axes_confident = all(
        v >= CONFIDENCE_THRESHOLD for v in axis_confs.values()
    )
    return all_axes_confident


def _count_pairs_used_for(state: SessionState, topic: str, axis: str) -> int:
    """Count how many pair interactions have targeted this topic×axis."""
    count = 0
    for h in state.history:
        if h.get("action_mode") == "pair" and h.get("target_topic") == topic and h.get("target_axis") == axis:
            count += 1
    return count


def decide_next_action(state: SessionState, available_topics: list[str]) -> dict:
    """
    Deterministic controller: decide what to show next.
    
    Strategy:
    - Step 0: broad shelf (discover topics)
    - Only show another shelf if topic coverage is genuinely thin (< 2 engaged)
    - Otherwise: pairs that rotate across engaged topics × weak axes
    """
    topic_confs = state.confidence.get("topics", {})
    axis_confs = state.confidence.get("axes", {})

    # First interaction → broad shelf
    if state.step_count == 0 or not topic_confs:
        return {
            "mode": "shelf",
            "scope": "broad",
            "topics": available_topics,
        }

    # Find engaged topics (any signal above noise)
    engaged_topics = sorted(
        [(t, c) for t, c in topic_confs.items() if c > 0.25],
        key=lambda x: -x[1]
    )

    if not engaged_topics:
        return {
            "mode": "shelf",
            "scope": "broad",
            "topics": available_topics,
        }

    engaged_names = [t for t, _ in engaged_topics]

    # Only show another shelf if we truly lack topic diversity
    if len(engaged_names) < MIN_TOPICS_CONFIDENT and state.step_count <= MAX_STEPS - 3:
        return {
            "mode": "shelf",
            "scope": "broad",
            "topics": available_topics,
        }

    # Mode B: find the best (topic, axis) pair to probe next
    weak_axes = [
        (ax_name, conf)
        for ax_name, conf in axis_confs.items()
        if conf < CONFIDENCE_THRESHOLD
    ]

    if not weak_axes:
        # All axes confident — we're essentially done
        return {
            "mode": "pair",
            "target_axis": min(axis_confs, key=axis_confs.get),
            "topic": engaged_names[0],
        }

    # Build scored candidates: rotate across (topic, axis) combos
    candidates = []
    for ax_name, ax_conf in weak_axes:
        for topic_name in engaged_names:
            times_probed = _count_pairs_used_for(state, topic_name, ax_name)
            priority = ax_conf + (times_probed * 0.2)
            candidates.append((topic_name, ax_name, priority))

    candidates.sort(key=lambda x: x[2])
    target_topic, target_axis, _ = candidates[0]

    return {
        "mode": "pair",
        "target_axis": target_axis,
        "topic": target_topic,
    }


def build_profile_update_prompt(state: SessionState, shown_posts: list[dict],
                                 user_choices: list[str]) -> tuple[str, str]:
    """Build the LLM profile update prompt using the prompts module."""
    from prompts import profile_update_system, profile_update_user
    system = profile_update_system()
    user = profile_update_user(state.profile, shown_posts, user_choices)
    return system, user


async def update_profile(state: SessionState, shown_posts: list[dict],
                         user_choices: list[str], action: dict = None,
                         valid_topics: set = None) -> SessionState:
    """Call LLM to update the user profile based on this interaction."""
    system, user = build_profile_update_prompt(state, shown_posts, user_choices)

    result = await llm_json(system, user, model=LLM_MODEL_PROFILE)

    # Update state
    state.profile["topics_prose"] = result.get("topics_prose", state.profile["topics_prose"])
    state.profile["style_prose"] = result.get("style_prose", state.profile["style_prose"])

    new_conf = result.get("confidence", {})
    if "topics" in new_conf:
        # Only accept topic names that exist in our corpus
        for k, v in new_conf["topics"].items():
            if valid_topics is None or k in valid_topics:
                state.confidence["topics"][k] = v
    if "axes" in new_conf:
        # Only accept known axis names
        from config.axes import ALL_AXIS_NAMES
        for k, v in new_conf["axes"].items():
            if k in ALL_AXIS_NAMES:
                state.confidence["axes"][k] = v

    # Record in history (include action metadata for rotation tracking)
    history_entry = {
        "step": state.step_count,
        "shown": [p["post_id"] for p in shown_posts],
        "chosen": user_choices,
        "profile_after": state.profile.copy(),
        "confidence_after": state.confidence.copy(),
    }
    if action:
        history_entry["action_mode"] = action.get("mode")
        history_entry["target_topic"] = action.get("topic")
        history_entry["target_axis"] = action.get("target_axis")
    state.history.append(history_entry)

    state.liked_post_ids.extend(user_choices)
    state.step_count += 1
    state.is_complete = check_stop_condition(state)

    return state


def compute_engagement_centroid(liked_ids: list[str], post_id_to_embedding: dict) -> list[float]:
    """Mean embedding of liked posts."""
    vecs = [post_id_to_embedding[pid] for pid in liked_ids if pid in post_id_to_embedding]
    if not vecs:
        return []
    return np.mean(vecs, axis=0).tolist()
