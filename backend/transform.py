"""
Transformation pipeline: rewrite post delivery to match user preferences.

Given:
  - An original post (with its current axis scores)
  - A user profile (preferred axis values from seeding)

Produces:
  - A rewritten post (same substance, adapted delivery)
  - A verification result (substance preserved? style shifted?)

Design constraints:
  - WHAT is said must not change (stance, facts, arguments)
  - HOW it's delivered can change (reading level, casualness, humor, etc.)
  - Subtractive transforms are safer than additive
  - Values/stance are NEVER transformed
"""

from config.axes import ALL_AXIS_NAMES, ADDITIVE_AXES
from config.settings import LLM_MODEL_PROFILE
from backend.llm_helper import llm_json
from prompts import transform_system, transform_user, verify_system, verify_user


# ── Transform spec ────────────────────────────────────────────────

def compute_transform_deltas(post_axes: dict, user_prefs: dict,
                             threshold: float = 0.3, max_axes: int = 2,
                             axis_use_counts: dict | None = None,
                             rotation_penalty: float = 0.1) -> dict:
    """
    Compute which axes need transformation and in which direction.
    Only picks the top max_axes (1-2) with largest gap, not all of them.

    axis_use_counts: optional {axis_name: times already picked this session}.
    Without it, ranking is pure |gap| -- but whichever of the user's
    preferred axes tends to have the largest average gap against the corpus
    will then dominate almost every pick (observed empirically: 2 of a
    user's 4 confident axes took ~99% of picks over 200 sampled posts, the
    other 2 almost never won). Passing a running per-session count makes
    ranking prefer axes with a large gap that *haven't* won recently --
    subtracting rotation_penalty per prior use is enough to spread picks
    close to evenly across all qualifying axes (validated empirically: same
    profile went from a 107/91/1/1 split to ~52/51/49/48 across 200 posts
    with penalty=0.1, and the result is not sensitive to the exact value).

    Returns:
        dict of {axis_name: {"current": float, "target": float, "direction": str}}
    """
    candidates = []
    for axis_name in ALL_AXIS_NAMES:
        post_val = post_axes.get(axis_name, {})
        if isinstance(post_val, dict):
            post_val = post_val.get("score")
        user_val = user_prefs.get(axis_name)

        if post_val is None or user_val is None:
            continue

        gap = user_val - post_val
        if abs(gap) >= threshold:
            candidates.append((axis_name, gap, post_val, user_val))

    if axis_use_counts:
        candidates.sort(
            key=lambda x: abs(x[1]) - rotation_penalty * axis_use_counts.get(x[0], 0),
            reverse=True,
        )
    else:
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    candidates = candidates[:max_axes]

    deltas = {}
    for axis_name, gap, post_val, user_val in candidates:
        deltas[axis_name] = {
            "current": post_val,
            "target": user_val,
            "direction": "increase" if gap > 0 else "decrease",
            "gap": round(gap, 3),
            "is_additive": axis_name in ADDITIVE_AXES,
        }
    return deltas


def compute_anti_preference_deltas(post_axes: dict, user_prefs: dict,
                                    threshold: float = 0.15, max_axes: int = 1,
                                    axis_use_counts: dict | None = None) -> dict:
    """
    Bidirectional manipulation check (see bt_preference_model memory /
    eval feature): among axes the user actually has a meaningful preference
    gap on for this post, push AWAY from that preference instead of toward
    it -- used to take a post the user already liked and deliberately
    degrade it on an axis they care about, to see if they now prefer the
    untouched original.

    NOT a mirror of compute_transform_deltas's target across the post's
    current value -- that degenerates to a zero-size shift whenever the
    post's current value is already near the 0/1 boundary in the away
    direction (e.g. humor already at 0.0 and the user wants MORE humor: away
    means pushing humor even lower, but there's no room below 0.0). Instead,
    each candidate axis's available "room" in the away direction is checked
    directly, and axes without enough room to produce a real shift are
    skipped in favor of one that has room -- rather than silently returning
    a no-op delta.
    """
    candidates = []
    for axis_name in ALL_AXIS_NAMES:
        post_val = post_axes.get(axis_name, {})
        if isinstance(post_val, dict):
            post_val = post_val.get("score")
        user_val = user_prefs.get(axis_name)
        if post_val is None or user_val is None:
            continue

        toward_gap = user_val - post_val
        if abs(toward_gap) < threshold:
            continue  # user has no real preference gap on this axis for this post

        if toward_gap > 0:
            # user wants MORE of this axis -> away means pushing it even LOWER
            away_direction = "decrease"
            room = post_val  # distance available down to 0.0
        else:
            away_direction = "increase"
            room = 1.0 - post_val  # distance available up to 1.0

        if room < threshold:
            continue  # not enough room to produce a real away-shift on this axis

        # Cap the shift below the full available room -- pushing all the way
        # to the 0/1 boundary (empirically) tends to make the LLM produce a
        # large, imprecise rewrite that disturbs several other axes instead
        # of a controlled, targeted one, since it isn't a natural point (see
        # bt_preference_model memory). A moderate degradation of the same
        # rough size as the normal toward-preference gap is both more
        # achievable and more comparable to the "toward" direction anyway.
        shift = min(max(abs(toward_gap), threshold), room, 0.35)
        away_target = post_val - shift if away_direction == "decrease" else post_val + shift
        away_target = max(0.0, min(1.0, away_target))
        candidates.append((axis_name, away_target - post_val, post_val, away_target, away_direction))

    if axis_use_counts:
        candidates.sort(
            key=lambda x: abs(x[1]) - 0.1 * axis_use_counts.get(x[0], 0),
            reverse=True,
        )
    else:
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    candidates = candidates[:max_axes]

    deltas = {}
    for axis_name, gap, post_val, away_target, direction in candidates:
        deltas[axis_name] = {
            "current": post_val,
            "target": away_target,
            "direction": direction,
            "gap": round(gap, 3),
            "is_additive": axis_name in ADDITIVE_AXES,
        }
    return deltas


# ── Pipeline ──────────────────────────────────────────────────────

async def transform_post(original_text: str, post_axes: dict,
                         user_prefs: dict, model: str = LLM_MODEL_PROFILE,
                         verify: bool = True, max_axes: int = 2,
                         orig_post_id: str | None = None,
                         style_gate: bool = True) -> dict:
    """
    Full transformation pipeline.

    Args:
        original_text: The original post text
        post_axes: The post's current axis scores (from annotation)
        user_prefs: User's preferred axis values (from seeding profile)
        model: LLM model to use
        max_axes: Maximum number of axes to transform (1-2 for natural output)
        verify: Whether to run verification step

    Returns:
        dict with keys: rewritten_text, changes_made, deltas, verification, used_original
    """
    deltas = compute_transform_deltas(post_axes, user_prefs, max_axes=max_axes)

    if not deltas:
        return {
            "rewritten_text": original_text,
            "changes_made": "No transformation needed — post already matches preferences.",
            "deltas": {},
            "verification": None,
            "used_original": True,
        }

    result = await llm_json(transform_system(), transform_user(original_text, deltas), model=model)

    rewritten = result.get("rewritten_text", original_text)
    transform_confidence = result.get("confidence", 0.0)

    # ── Gate 1: LLM substance check ───────────────────────────────
    verification = None
    if verify:
        verification = await llm_json(verify_system(), verify_user(original_text, rewritten), model=model)

        if not verification.get("substance_preserved", True):
            return {
                "rewritten_text": original_text,
                "changes_made": "Transformation rejected — substance not preserved.",
                "deltas": deltas,
                "verification": verification,
                "style_verification": None,
                "used_original": True,
            }

    # ── Gate 2: style-axis + content-preservation check ────────────
    # Fine-tuned classifier (axis shifts) + BGE-M3 cosine similarity
    # (content), replacing the old Qwen -> SAE -> Ridge pipeline -- see
    # sae_verification_findings memory: the classifier beat Ridge decisively
    # in head-to-head testing, and doesn't need a 7B model loaded at all.
    # Content check switched from BERTScore to BGE-M3 cosine after a
    # 100-pair human-rating test showed BERTScore is blind to targeted
    # meaning-inverting edits (e.g. a flipped claim) that BGE-M3 catches.
    style_verification = None
    if style_gate:
        import asyncio
        from backend.verify_classifier import verify_rewrite as _style_verify
        target_axes = list(deltas.keys())
        loop = asyncio.get_event_loop()
        style_verification = await loop.run_in_executor(
            None,
            lambda: _style_verify(original_text, rewritten, target_axes, orig_post_id),
        )
        if style_verification.get("verdict") == "leaked":
            return {
                "rewritten_text": original_text,
                "changes_made": (
                    f"Transformation rejected — verification detected axis leakage or "
                    f"content drift (target shift {style_verification['min_target_shift']:.3f}, "
                    f"unintended shift {style_verification['max_other_shift']:.3f}, "
                    f"content similarity {style_verification['bge_cosine']:.3f})."
                ),
                "deltas": deltas,
                "verification": verification,
                "style_verification": style_verification,
                "used_original": True,
            }

    return {
        "rewritten_text": rewritten,
        "changes_made": result.get("changes_made", ""),
        "additive_material_added": result.get("additive_material_added", False),
        "transform_confidence": transform_confidence,
        "deltas": deltas,
        "verification": verification,
        "style_verification": style_verification,
        "used_original": False,
    }
