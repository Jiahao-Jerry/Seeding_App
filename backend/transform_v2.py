"""
No-LLM-verification transform pipeline: the "new architecture" variant.

LLM is used ONLY to generate the rewrite (transform_system/transform_user) --
never to grade it. Verification is classifier (style-axis shift) + BERTScore
(content preservation) exclusively, no LLM substance check. If the gate
rejects a rewrite (leaked axis OR poor content preservation), retry
generation up to max_retries times before falling back to the original text.

Parallel to backend/transform.py -- that module (and its LLM Gate 1 substance
check) is untouched.
"""
import asyncio

from config.settings import LLM_MODEL_PROFILE
from backend.llm_helper import llm_json
from prompts import transform_system, transform_user
from backend.transform import compute_transform_deltas  # pure function, no LLM involved
from backend.verify_classifier import verify_rewrite as _style_verify


async def transform_post_v2(original_text: str, post_axes: dict, user_prefs: dict,
                             model: str = LLM_MODEL_PROFILE, max_axes: int = 1,
                             orig_post_id: str | None = None, max_retries: int = 3,
                             axis_use_counts: dict | None = None,
                             deltas_override: dict | None = None,
                             axis_ratio: float = 1.5) -> dict:
    """
    Full no-LLM-grading transformation pipeline:
      1. Compute deltas from the BT-model-derived user_prefs (no LLM).
      2. Generate a rewrite (LLM -- this is the one legitimate generation call).
      3. Verify with the classifier + BERTScore (no LLM).
      4. If rejected, retry generation up to max_retries times.
      5. If still rejected, fall back to the original text.

    axis_use_counts: optional running per-session {axis: times picked}, passed
    straight to compute_transform_deltas() to keep target-axis selection from
    being dominated by whichever axis usually has the largest gap -- see that
    function's docstring. Mutated in place (once per call, right after
    picking) so the caller's dict reflects usage across posts as they're
    generated, including this pick, before any concurrent sibling call
    (asyncio.gather batches) resumes past its own first await.

    deltas_override: if given, skip computing deltas from post_axes/user_prefs
    and use these directly instead -- used by the anti-preference eval trials
    (compute_anti_preference_deltas), which need the same generate/verify/
    retry pipeline but a deliberately reversed target direction.
    """
    if deltas_override is not None:
        deltas = deltas_override
    else:
        deltas = compute_transform_deltas(post_axes, user_prefs, max_axes=max_axes,
                                           axis_use_counts=axis_use_counts)
        if axis_use_counts is not None:
            for axis_name in deltas:
                axis_use_counts[axis_name] = axis_use_counts.get(axis_name, 0) + 1
    if not deltas:
        return {
            "rewritten_text": original_text,
            "changes_made": "No transformation needed — post already matches preferences.",
            "deltas": {},
            "style_verification": None,
            "used_original": True,
            "attempts": 0,
        }

    target_axes = list(deltas.keys())
    target_directions = {ax: deltas[ax]["direction"] for ax in target_axes}
    loop = asyncio.get_event_loop()

    # Escalating temperature per retry: attempt 1 stays low (consistent,
    # predictable first try), later attempts explore more so a rejected
    # rewrite doesn't just get re-asked at ~temperature 0 and come back
    # nearly identical (observed empirically -- see bt_preference_model /
    # sae_verification_findings memory).
    RETRY_TEMPERATURES = [0.2, 0.6, 0.9, 1.0]

    # Attempts are fired concurrently rather than one-at-a-time: the
    # escalating temperature is just a diversity knob, not real feedback from
    # a previous attempt's failure, so there's nothing to learn by waiting
    # for attempt N to fail before starting attempt N+1. This turns a
    # worst-case ~15-20s * max_retries sequential chain into roughly one
    # attempt's worth of wall-clock time, at the cost of always spending
    # max_retries LLM calls instead of stopping early on a first success.
    async def _run_attempt(attempt: int) -> dict:
        temp = RETRY_TEMPERATURES[min(attempt - 1, len(RETRY_TEMPERATURES) - 1)]
        result = await llm_json(transform_system(), transform_user(original_text, deltas),
                                 model=model, temperature=temp)
        rewritten = result.get("rewritten_text", original_text)
        style_verification = await loop.run_in_executor(
            None,
            lambda: _style_verify(original_text, rewritten, target_axes, orig_post_id,
                                   target_directions=target_directions, axis_ratio=axis_ratio),
        )
        return {
            "attempt": attempt,
            "temperature": temp,
            "rewritten": rewritten,
            "result": result,
            "style_verification": style_verification,
        }

    attempt_results = await asyncio.gather(*[_run_attempt(a) for a in range(1, max_retries + 1)])
    attempt_results.sort(key=lambda r: r["attempt"])  # deterministic: prefer earliest clean attempt

    attempts_log = [{
        "attempt": r["attempt"],
        "temperature": r["temperature"],
        "verdict": r["style_verification"]["verdict"],
        "min_target_shift": r["style_verification"]["min_target_shift"],
        "max_other_shift": r["style_verification"]["max_other_shift"],
        "bge_cosine": r["style_verification"]["bge_cosine"],
    } for r in attempt_results]

    clean = next((r for r in attempt_results if r["style_verification"]["verdict"] == "clean"), None)
    if clean is not None:
        result, style_verification = clean["result"], clean["style_verification"]
        return {
            "rewritten_text": clean["rewritten"],
            "changes_made": result.get("changes_made", ""),
            "additive_material_added": result.get("additive_material_added", False),
            "transform_confidence": result.get("confidence", 0.0),
            "deltas": deltas,
            "style_verification": style_verification,
            "used_original": False,
            "attempts": clean["attempt"],
            "attempts_log": attempts_log,
        }

    # Every concurrent attempt leaked or failed content preservation.
    last = attempt_results[-1]
    return {
        "rewritten_text": original_text,
        "changes_made": (
            f"Transformation rejected after {max_retries} attempts — classifier/"
            f"content-preservation gate never passed."
        ),
        "deltas": deltas,
        "style_verification": last["style_verification"],  # kept for debugging
        "used_original": True,
        "attempts": max_retries,
        "attempts_log": attempts_log,
    }
