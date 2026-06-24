"""
Per-feature axis alignment using BOTH Pearson r AND axis lift.

For each (feature, axis) pair we compute:
    r    = Pearson correlation between activation magnitudes and axis scores
    lift = mean(axis_score | feature active) − mean(axis_score | feature inactive)

Why report both:
- Pearson r uses activation magnitudes (continuous signal) and is informative
  when features are densely active (e.g. L1-regularized SAE, ~30-40% density).
  But it suffers "zero dilution" when features are very sparse (most rows are 0,
  forming a vertical line that fights the linear fit).
- Lift uses binary active/inactive and is robust to very sparse features
  (e.g. TopK SAE, ~10% density). But its discrimination collapses at high
  density because both pools span the full axis range.

We classify each feature using whichever has the larger absolute effect:
    score = max(|r|, |lift|)
    - score >= confirm_lift  → "confirms_axis"
    - score >= partial_lift  → "partial_overlap"
    - else                   → "novel_candidate"
    - density < dead_density → "dead"
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _post_axis_matrix(df: pd.DataFrame, axis_names: list[str]) -> np.ndarray:
    """Return an (N, A) matrix of axis scores. NaN where missing."""
    n = len(df)
    arr = np.full((n, len(axis_names)), np.nan, dtype=np.float64)
    for row_i, axes_json in enumerate(df["axes_json"].values):
        if not axes_json or pd.isna(axes_json):
            continue
        try:
            data = json.loads(axes_json)
        except Exception:
            continue
        for col_i, ax in enumerate(axis_names):
            v = data.get(ax)
            if isinstance(v, dict) and "score" in v:
                arr[row_i, col_i] = float(v["score"])
            elif isinstance(v, (int, float)):
                arr[row_i, col_i] = float(v)
    return arr


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r ignoring NaNs. Returns 0.0 if undefined."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return 0.0
    aa, bb = a[mask], b[mask]
    if aa.std() < 1e-9 or bb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _axis_lift(activations_col: np.ndarray, axis_col: np.ndarray) -> float:
    """mean(axis | feature active) − mean(axis | feature inactive)."""
    active = activations_col > 0
    valid = ~np.isnan(axis_col)
    active_valid = active & valid
    inactive_valid = (~active) & valid
    if not active_valid.any() or not inactive_valid.any():
        return 0.0
    return float(axis_col[active_valid].mean() - axis_col[inactive_valid].mean())


def correlate_features_with_axes(
    activations: np.ndarray,
    df: pd.DataFrame,
    axis_names: list[str],
    confirm_lift: float = 0.20,
    partial_lift: float = 0.10,
    dead_density: float = 0.01,
) -> list[dict[str, Any]]:
    """Return one record per feature with both r and lift per axis."""
    if len(df) != activations.shape[0]:
        raise ValueError("df and activations row count mismatch")

    axis_mat = _post_axis_matrix(df, axis_names)
    n_features = activations.shape[1]
    records: list[dict[str, Any]] = []

    for f_idx in range(n_features):
        col = activations[:, f_idx].astype(np.float64)
        density = float((col > 0).mean())

        rs: dict[str, float] = {}
        lifts: dict[str, float] = {}
        for a_idx, ax in enumerate(axis_names):
            rs[ax] = round(_pearson(col, axis_mat[:, a_idx]), 3)
            lifts[ax] = round(_axis_lift(col, axis_mat[:, a_idx]), 3)

        best_axis_r = max(rs, key=lambda k: abs(rs[k])) if rs else None
        best_axis_lift = max(lifts, key=lambda k: abs(lifts[k])) if lifts else None
        best_r = rs[best_axis_r] if best_axis_r else 0.0
        best_lift = lifts[best_axis_lift] if best_axis_lift else 0.0
        score = max(abs(best_r), abs(best_lift))
        best_axis = best_axis_r if abs(best_r) >= abs(best_lift) else best_axis_lift

        if density < dead_density:
            category = "dead"
        elif score >= confirm_lift:
            category = "confirms_axis"
        elif score >= partial_lift:
            category = "partial_overlap"
        else:
            category = "novel_candidate"

        records.append({
            "feature": f_idx,
            "density": round(density, 4),
            "correlations": rs,
            "lifts": lifts,
            "best_axis": best_axis,
            "best_r": round(best_r, 3),
            "best_lift": round(best_lift, 3),
            "category": category,
        })

    return records


def summarize_categories(corr_records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in corr_records:
        c = r["category"]
        out[c] = out.get(c, 0) + 1
    return out

