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

import json
from config.axes import AXES, ALL_AXIS_NAMES, SUBTRACTIVE_AXES, ADDITIVE_AXES
from backend.llm_helper import llm_json


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

    # Sort by gap magnitude, pick top max_axes
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


# ── Prompts ───────────────────────────────────────────────────────

def _axis_instruction(axis_name: str, delta: dict) -> str:
    """Generate a single axis transformation instruction."""
    ax_def = next((a["definition"] for a in AXES if a["name"] == axis_name), "")
    direction = delta["direction"]
    current = delta["current"]
    target = delta["target"]

    return (
        f"- {axis_name.replace('_', ' ').title()} ({ax_def}): "
        f"Currently {current:.2f}/1.0, target {target:.2f}/1.0. "
        f"{direction.upper()} this quality."
    )


def transform_system_prompt() -> str:
    return """You rewrite social media posts to adapt their DELIVERY STYLE while preserving their SUBSTANCE.

HARD CONSTRAINTS (never violate):
1. Preserve the core claim, factual content, and the author's stance.
2. Do not invent new arguments or remove key information.
3. STAY WITHIN ±20% of the original word count. This is strict. If the original is 40 words, your rewrite must be 32–48 words. Restructure rather than expand.
4. Preserve the textual register of the original. If it uses fragments, slang, internet shorthand, lowercase — keep that. Do not polish into formal prose. A messy post rewritten in a different style should still feel messy.

STYLE SHIFT GUIDANCE:
- You are changing HOW something is delivered along specific axes. Not cleaning it up.
- For narrativity: restructure into story/anecdote form WITHIN the same length. Cut filler to make room for framing.
- For tone: shift the emotional temperature without adding hedges or qualifiers that inflate word count.
- Do NOT default to first-person "I" framing. Narrative can be observational, second-person, or scene-setting without "I started..." or "I was thinking..."
- The result should read like someone with a DIFFERENT style wrote the SAME post on the SAME platform. Not like an editor rewrote it for a magazine.

Return JSON:
{
  "rewritten_text": "<the transformed post>",
  "changes_made": "<1-2 sentences: what you changed and why it fits>",
  "additive_material_added": <true/false — did you introduce framing/material beyond the original?>,
  "confidence": <0.0-1.0 — how confident the original claim/facts are intact>
}"""


def transform_user_prompt(original_text: str, deltas: dict) -> str:
    instructions = "\n".join(
        _axis_instruction(ax, d) for ax, d in deltas.items()
    )
    word_count = len(original_text.split())

    return f"""ORIGINAL POST ({word_count} words):
\"{original_text}\"

STYLE SHIFT (focus on these {len(deltas)} dimension{'s' if len(deltas) > 1 else ''} only):
{instructions}

Rewrite this post in {word_count}±{max(5, word_count // 5)} words. Same substance, different delivery. Keep the original's register and platform feel."""


# ── Verification ──────────────────────────────────────────────────

def verify_system_prompt() -> str:
    return """You verify whether a rewritten post preserves the substance of the original.

Compare the original and rewrite on these dimensions:
1. Core claim/opinion — is it the same?
2. Factual content — is anything added or removed?
3. Author's stance — is the position unchanged?
4. Key evidence/examples — are they preserved (even if rephrased)?

Return JSON:
{
  "substance_preserved": <true/false>,
  "style_shifted": <true/false — does the rewrite actually sound different?>,
  "issues": "<describe any substance violations, or 'none'>",
  "fidelity_score": <0.0-1.0 — 1.0 means perfect preservation>
}"""


def verify_user_prompt(original_text: str, rewritten_text: str) -> str:
    return f"""ORIGINAL:
\"{original_text}\"

REWRITE:
\"{rewritten_text}\"

Verify substance preservation and style shift."""


# ── Pipeline ──────────────────────────────────────────────────────

async def transform_post(original_text: str, post_axes: dict,
                         user_prefs: dict, model: str = "gpt-5.4-mini",
                         verify: bool = True, max_axes: int = 2) -> dict:
    """
    Full transformation pipeline.

    Args:
        original_text: The original post text
        post_axes: The post's current axis scores (from annotation)
        user_prefs: User's preferred axis values (from seeding profile)
        model: LLM model to use
        max_axes: Maximum number of axes to transform (1-2 for natural output)
        verify: Whether to run verification step
        max_axes: Maximum number of axes to transform (1-2 for natural output)

    Returns:
        dict with keys: rewritten_text, changes_made, deltas, verification, used_original
    """
    # Step 1: Compute what needs to change
    deltas = compute_transform_deltas(post_axes, user_prefs, max_axes=max_axes)

    if not deltas:
        # Post already matches user preferences — no transformation needed
        return {
            "rewritten_text": original_text,
            "changes_made": "No transformation needed — post already matches preferences.",
            "deltas": {},
            "verification": None,
            "used_original": True,
        }

    # Step 2: Transform
    system = transform_system_prompt()
    user = transform_user_prompt(original_text, deltas)
    result = await llm_json(system, user, model=model)

    rewritten = result.get("rewritten_text", original_text)
    transform_confidence = result.get("confidence", 0.0)

    # Step 3: Verify (optional but recommended)
    verification = None
    if verify:
        v_system = verify_system_prompt()
        v_user = verify_user_prompt(original_text, rewritten)
        verification = await llm_json(v_system, v_user, model=model)

        # If substance not preserved, fall back to original
        if not verification.get("substance_preserved", True):
            return {
                "rewritten_text": original_text,
                "changes_made": "Transformation rejected — substance not preserved.",
                "deltas": deltas,
                "verification": verification,
                "used_original": True,
            }

    return {
        "rewritten_text": rewritten,
        "changes_made": result.get("changes_made", ""),
        "additive_material_added": result.get("additive_material_added", False),
        "transform_confidence": transform_confidence,
        "deltas": deltas,
        "verification": verification,
        "used_original": False,
    }
