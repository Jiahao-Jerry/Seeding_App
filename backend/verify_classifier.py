"""
Alternative rewrite verification: fine-tuned DistilBERT classifier (axis
shifts) + BGE-M3 whole-post cosine similarity (content preservation),
instead of Qwen2.5-7B -> SAE -> Ridge (backend/sae_verify.py).

This is a PARALLEL implementation -- it does not modify or replace
sae_verify.py, and nothing in the live app calls it yet. It mirrors
verify_rewrite()'s interface/return shape so it can be swapped in later
once tried out. See scripts/try_verify_classifier.py to run it.

Why: head-to-head testing (see memory / sae_verification_findings) showed
the classifier beats Ridge decisively on both a 50-pair and a balanced
90-pair test set. Content preservation switched from BERTScore to BGE-M3
cosine similarity after a 100-pair human-rating experiment
(data/bertscore_human_test/) showed BERTScore's token-level greedy matching
is structurally blind to targeted meaning-inverting edits (a flipped claim
scored *higher* than legitimate rewrites, both raw and baseline-rescaled,
across every layer tested) -- because a single antonym swap is swamped by
the surrounding unchanged tokens. BGE-M3's whole-post embedding correlated
far better with real human judgment on the same pairs (r=0.706 vs
BERTScore's 0.071 on a severely-corrupted test set) and correctly separated
content-preserved from content-altered rewrites where BERTScore did not.
"""

import numpy as np
import torch
from pathlib import Path

from backend.classifier.model import AXIS_NAMES, AxisRegressor
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).parent.parent / "data"
CLASSIFIER_DIR = DATA_DIR / "classifier"
MAX_LEN = 128

# Floor picked from the real "good" (content-preserved) rewrite distribution
# in the 100-pair human-rating test (data/bertscore_human_test/): mean 0.964,
# p5 0.882, p1 0.828. 0.85 sits just above the p1 tail -- close enough to the
# floor that legitimate rewrites rarely get falsely rejected, while severely
# content-altered rewrites (mean 0.739 on the same test) sit clearly below it.
BGE_COSINE_FLOOR = 0.85

_model: AxisRegressor | None = None
_tokenizer = None
_bge_model = None


def load_classifier():
    global _model, _tokenizer
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR / "tokenizer")
    _model = AxisRegressor()
    _model.load_state_dict(torch.load(CLASSIFIER_DIR / "axis_regressor.pt", map_location="cpu"))
    _model.eval()


def _score_text(text: str) -> np.ndarray:
    """Text -> 9-dim axis score vector (0-1 per axis)."""
    load_classifier()
    enc = _tokenizer(text, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = _model(enc["input_ids"], enc["attention_mask"])
    return out.squeeze(0).numpy()


def _load_bge_model():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer("BAAI/bge-m3")
    return _bge_model


def _bge_cosine(original_text: str, rewritten_text: str) -> float:
    """Content-preservation similarity: BGE-M3 whole-post embedding cosine, CPU."""
    model = _load_bge_model()
    embs = model.encode([original_text, rewritten_text], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def verify_rewrite(
    original_text: str,
    rewritten_text: str,
    target_axes: list[str],
    orig_post_id: str | None = None,  # accepted for interface parity with sae_verify; unused here
    target_directions: dict[str, str] | None = None,
    axis_ratio: float = 1.5,
) -> dict:
    """
    Classifier + BGE-M3 cosine similarity verification of a rewrite.

    Mirrors sae_verify.verify_rewrite()'s return shape (axis_shifts,
    target_axes, passed, verdict, max_other_shift) and adds a content-
    preservation section (bge_cosine, content_passed). NOTE: unlike
    sae_verify, target shift is min_target_shift here (see axis_passed) --
    the worst-performing target axis must individually clear the bar, not
    just the best one.

    target_directions: optional {axis: "increase"|"decrease"}. Without it,
    only the MAGNITUDE of the shift is checked -- a rewrite that moved a
    target axis by enough, but in the WRONG direction, would still pass.
    That's a real gap for anything that needs to push a specific direction
    on purpose (e.g. the anti-preference bidirectional eval trials, which
    deliberately target "decrease" where compute_transform_deltas would
    normally say "increase" -- without a direction check, a same-direction
    rewrite that happened to clear the magnitude bar would be wrongly
    marked "clean" even though it moved the wrong way).

    axis_ratio: how much bigger the target shift must be than the largest
    collateral shift on any other axis. Default 1.5 is the bar for the main
    product (feed/normal eval trials). The anti-preference bidirectional
    trials use a lower value (documented at their call site) -- empirically,
    asking the LLM to push an already-liked post AWAY from preference tends
    to produce much larger collateral changes than a normal toward-
    preference rewrite (median best-case ratio ~0.3-0.4 vs routinely >1.2
    for toward-preference on the same corpus), so holding it to the exact
    same bar made it fail almost every liked post. This is a deliberate,
    separate, and honestly-labeled threshold for a control condition, not a
    silent weakening of the main gate.
    """
    scores_orig = _score_text(original_text)
    scores_new = _score_text(rewritten_text)
    delta = scores_new - scores_orig
    shifts = {ax: float(delta[i]) for i, ax in enumerate(AXIS_NAMES)}

    # min(), not max(): with 2 target axes, require the WORST-performing
    # target axis to still individually clear the bar -- otherwise a rewrite
    # could pass on the strength of just one of the two intended axes while
    # the other barely moved at all.
    target_mag = min((abs(shifts[ax]) for ax in target_axes if ax in shifts), default=0.0)
    other_mag = max((abs(v) for ax, v in shifts.items() if ax not in target_axes), default=0.0)
    magnitude_passed = target_mag > 0.05 and (other_mag == 0.0 or target_mag / other_mag > axis_ratio)

    direction_passed = True
    if target_directions:
        for ax in target_axes:
            if ax not in shifts or ax not in target_directions:
                continue
            expected_sign = 1 if target_directions[ax] == "increase" else -1
            actual_sign = 1 if shifts[ax] >= 0 else -1
            if expected_sign != actual_sign:
                direction_passed = False
                break

    axis_passed = magnitude_passed and direction_passed

    bge_cosine = _bge_cosine(original_text, rewritten_text)
    content_passed = bge_cosine >= BGE_COSINE_FLOOR

    passed = axis_passed and content_passed

    return {
        "axis_shifts": {ax: round(v, 4) for ax, v in shifts.items()},
        "target_axes": target_axes,
        "target_directions": target_directions,
        "axis_ratio_used": axis_ratio,
        "scores_orig": {ax: round(float(scores_orig[i]), 4) for i, ax in enumerate(AXIS_NAMES)},
        "scores_new": {ax: round(float(scores_new[i]), 4) for i, ax in enumerate(AXIS_NAMES)},
        "bge_cosine": round(bge_cosine, 4),
        "content_passed": content_passed,
        "axis_passed": axis_passed,
        "direction_passed": direction_passed,
        "passed": passed,
        "verdict": "clean" if passed else "leaked",
        "min_target_shift": round(target_mag, 4),  # min() over target axes -- see axis_passed above
        "max_other_shift": round(other_mag, 4),
    }
