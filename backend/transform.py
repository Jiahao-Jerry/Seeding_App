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
                             threshold: float = 0.15, max_axes: int = 2) -> dict:
    """
    Compute which axes need transformation and in which direction.
    Only picks the top max_axes (1-2) with largest gap, not all of them.

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


# ── Pipeline ──────────────────────────────────────────────────────

async def transform_post(original_text: str, post_axes: dict,
                         user_prefs: dict, model: str = LLM_MODEL_PROFILE,
                         verify: bool = True, max_axes: int = 2,
                         orig_post_id: str | None = None,
                         sae_gate: bool = True) -> dict:
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
                "sae_verification": None,
                "used_original": True,
            }

    # ── Gate 2: SAE axis-leak check ───────────────────────────────
    sae_verification = None
    if sae_gate:
        import asyncio
        from backend.sae_verify import verify_rewrite as _sae_verify, _qwen_model
        if _qwen_model is not None:  # only gate if Qwen is already loaded
            target_axes = list(deltas.keys())
            loop = asyncio.get_event_loop()
            sae_verification = await loop.run_in_executor(
                None,
                lambda: _sae_verify(original_text, rewritten, target_axes, orig_post_id),
            )
            if sae_verification.get("verdict") == "leaked":
                return {
                    "rewritten_text": original_text,
                    "changes_made": (
                        f"Transformation rejected — SAE detected axis leakage "
                        f"(target shift {sae_verification['max_target_shift']:.3f}, "
                        f"unintended shift {sae_verification['max_other_shift']:.3f})."
                    ),
                    "deltas": deltas,
                    "verification": verification,
                    "sae_verification": sae_verification,
                    "used_original": True,
                }

    return {
        "rewritten_text": rewritten,
        "changes_made": result.get("changes_made", ""),
        "additive_material_added": result.get("additive_material_added", False),
        "transform_confidence": transform_confidence,
        "deltas": deltas,
        "verification": verification,
        "sae_verification": sae_verification,
        "used_original": False,
    }
