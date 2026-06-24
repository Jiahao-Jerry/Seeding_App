"""
SAE feature interpretation.

For each feature, surface:
- Density: fraction of posts that activate it (non-zero)
- Mean / max activation
- Top-K activating posts (text, axes, topic) so a human can read them
- Bottom-K (lowest-but-active) posts for contrast
- Dead-feature flag

Output is a list of dicts (one per feature), ready for JSON serialization or
report rendering.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _axes_summary(axes_json: str | None) -> dict:
    """Reduce the LLM annotation to {axis: score} for compact display."""
    if not axes_json or pd.isna(axes_json):
        return {}
    try:
        data = json.loads(axes_json)
    except Exception:
        return {}
    out = {}
    for ax, val in data.items():
        if isinstance(val, dict) and "score" in val:
            out[ax] = round(float(val["score"]), 2)
    return out


def interpret_features(
    activations: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 20,
    dead_density: float = 0.01,
) -> list[dict[str, Any]]:
    """
    Build per-feature interpretation records.

    activations : (N, F) array of SAE feature activations
    df          : aligned corpus DataFrame with post_id, topic_name, text, axes_json
    top_k       : number of top-activating (and bottom-active) posts to record
    dead_density: density threshold below which a feature is flagged dead
    """
    if len(df) != activations.shape[0]:
        raise ValueError("df and activations row count mismatch")

    n_features = activations.shape[1]
    posts = df.reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for f_idx in range(n_features):
        col = activations[:, f_idx]
        active_mask = col > 0
        density = float(active_mask.mean())

        record: dict[str, Any] = {
            "feature": f_idx,
            "density": round(density, 4),
            "mean_activation": round(float(col.mean()), 4),
            "mean_active_activation": round(
                float(col[active_mask].mean()) if active_mask.any() else 0.0, 4
            ),
            "max_activation": round(float(col.max()), 4),
            "is_dead": density < dead_density,
            "top_posts": [],
            "bottom_active_posts": [],
        }

        if not active_mask.any():
            records.append(record)
            continue

        top_idx = np.argsort(-col)[:top_k]
        for i in top_idx:
            if col[i] <= 0:
                break
            row = posts.iloc[int(i)]
            record["top_posts"].append({
                "post_id": str(row["post_id"]),
                "topic": row.get("topic_name", ""),
                "activation": round(float(col[int(i)]), 4),
                "text": str(row.get("text", ""))[:500],
                "axes": _axes_summary(row.get("axes_json")),
            })

        active_idx = np.where(active_mask)[0]
        if len(active_idx) > 0:
            order = active_idx[np.argsort(col[active_idx])]
            for i in order[:top_k]:
                row = posts.iloc[int(i)]
                record["bottom_active_posts"].append({
                    "post_id": str(row["post_id"]),
                    "topic": row.get("topic_name", ""),
                    "activation": round(float(col[int(i)]), 4),
                    "text": str(row.get("text", ""))[:500],
                    "axes": _axes_summary(row.get("axes_json")),
                })

        records.append(record)

    return records
