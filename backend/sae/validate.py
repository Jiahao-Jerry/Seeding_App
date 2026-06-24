"""
SAE2 validation harness — how we know the fingerprint works.

Three jobs (see §6 of docs/style_sae_handover.md):

  coverage(...)         For each of the 9 axes (from config/axes.py), does at least
                        one SAE feature track it? Judged against the LLM axis-score
                        reference (Pearson r + axis lift; reuse backend.sae.correlate).
                        Use each axis's `measurability` field to set expectations —
                        high-measurability axes should come out cleanly; `humor`
                        (low) is the bellwether we explicitly report on.

  pair_separation(...)  The feature that best tracks an axis should SEPARATE that
                        axis's synthetic single-axis pairs: report the activation
                        gap (shifted vs base) and ROC-AUC.

  assert_object_type    The hard guard for the train/infer invariant (§2): an SAE
                        trained on single posts must never be fed differences, and
                        vice-versa. Implemented below; everything that fingerprints
                        should call it.

Acceptance: a variant "passes" when every high-measurability axis is covered at
>= SAE2_CONFIRM by some feature, with a documented verdict on humor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_object_type(meta: dict, input_kind: str) -> None:
    """
    Raise if a model is asked to encode the wrong kind of input.
    meta is the variant's meta.json dict; input_kind is "single_post" or
    "pair_difference".
    """
    trained = meta.get("object_type")
    if trained is None:
        raise ValueError("model meta is missing 'object_type'.")
    if trained != input_kind:
        raise ValueError(
            f"object-type mismatch: model was trained on '{trained}' but received "
            f"'{input_kind}'. f(A-B) != f(A)-f(B); feed the trained object type."
        )


def coverage(activations: np.ndarray, labels: pd.DataFrame, axis_names: list[str],
             confirm: float, partial: float) -> dict:
    """
    For each axis, find the best feature by max(|Pearson r|, |axis lift|) against the
    labelled subset, and bucket it as confirmed / partial / uncovered.

    activations : (N, F) feature activations for the labelled posts (subset-aligned)
    labels      : DataFrame with one column per axis name (LLM scores in [0,1])
    Returns a per-axis record plus an overall count of covered axes.

    TODO: reuse backend.sae.correlate for r + lift; fold in axes.py `measurability`
    so the report flags expected-vs-actual per axis.
    """
    raise NotImplementedError


def pair_separation(model, meta: dict, pairs: pd.DataFrame, axis_names: list[str],
                    representation_fn) -> dict:
    """
    Per axis, measure whether its best-tracking feature separates that axis's
    synthetic pairs. representation_fn maps text(s) -> the model's input object
    (respecting meta['object_type'] via assert_object_type).

    Returns per-axis activation gap (shifted - base) and ROC-AUC.

    TODO: implement once pairs.py produces the synthetic set.
    """
    raise NotImplementedError
