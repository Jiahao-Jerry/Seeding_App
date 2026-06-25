"""
SAE2 layer sweep — pick the Qwen layer empirically instead of guessing.

Style is encoded across a broad band of the network, strongest in the
upper-middle (Konen et al., Style Vectors). We extract a few candidate layers
(config SAE2_QWEN_LAYERS) in one pass, then score each by how well its features
track the 9 axes against the labelled subset, and keep the winner.

Selection metric: mean over axes of the best single-feature max(|r|, |lift|).
Higher = the layer's representation exposes more of the delivery dimensions.

Run:  python backend/sae/layer_sweep.py   (after extract_qwen.py + label subset)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))

from config.axes import ALL_AXIS_NAMES
from config.settings import (
    SAE2_VARIANTS_DIR, SAE2_QWEN_DIR, SAE2_QWEN_LAYERS,
    SAE2_DATASET_FILE, SAE2_LABELS_FILE, SAE2_REPORT_FILE,
    SAE2_CONFIRM, SAE2_PARTIAL,
)
from backend.sae.correlate import _pearson, _axis_lift
from backend.sae.run_variant import train_variant, load_variant


def _load_aligned_labels() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load axis_labels.parquet and return (labels_df, row_indices_in_dataset).
    row_indices_in_dataset[i] is the row in dataset.parquet that matches labels_df.iloc[i].
    """
    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)

    post_id_to_idx = {str(pid): i for i, pid in enumerate(dataset["post_id"].astype(str))}
    row_indices = np.array([
        post_id_to_idx[str(pid)]
        for pid in labels["post_id"].astype(str)
    ], dtype=np.int64)

    return labels, row_indices


def _axis_score_matrix(labels: pd.DataFrame) -> np.ndarray:
    """Return (N, 9) float64 matrix of axis scores from direct parquet columns."""
    arr = np.full((len(labels), len(ALL_AXIS_NAMES)), np.nan, dtype=np.float64)
    for col_i, ax in enumerate(ALL_AXIS_NAMES):
        if ax in labels.columns:
            arr[:, col_i] = labels[ax].to_numpy(dtype=np.float64)
    return arr


def _alignment_score(activations: np.ndarray, axis_mat: np.ndarray) -> tuple[float, dict]:
    """
    For each axis find the best single feature by max(|r|, |lift|).
    Returns (mean_over_axes, per_axis_dict).
    """
    per_axis: dict[str, float] = {}
    for a_idx, ax in enumerate(ALL_AXIS_NAMES):
        axis_col = axis_mat[:, a_idx]
        best = 0.0
        for f_idx in range(activations.shape[1]):
            feat_col = activations[:, f_idx].astype(np.float64)
            r = abs(_pearson(feat_col, axis_col))
            lift = abs(_axis_lift(feat_col, axis_col))
            score = max(r, lift)
            if score > best:
                best = score
        per_axis[ax] = round(best, 4)
    return float(np.mean(list(per_axis.values()))), per_axis


def score_layer(layer: int, labels: pd.DataFrame, row_indices: np.ndarray,
                axis_mat: np.ndarray) -> float:
    """
    Train a SAE on qwen{layer}_knn, score its features against axis labels.
    Returns mean-over-axes best-feature max(|r|, |lift|).
    """
    variant_id = f"qwen{layer}_knn"
    variant_dir = APP_ROOT / SAE2_VARIANTS_DIR / variant_id

    if not (variant_dir / "meta.json").exists():
        rep_file = APP_ROOT / SAE2_VARIANTS_DIR / f"{variant_id}.npy"
        if not rep_file.exists():
            raise FileNotFoundError(
                f"Representation matrix not found: {rep_file}\n"
                "Run representations.py first."
            )
        x = np.load(rep_file)
        train_variant(variant_id, x, "single_post",
                      meta_extra={"space": "qwen", "layer": layer, "removal": "knn"})

    _, _, activations = load_variant(variant_id)
    aligned_acts = activations[row_indices]

    mean_score, per_axis = _alignment_score(aligned_acts, axis_mat)
    return mean_score, per_axis


def sweep() -> dict:
    """
    Score every SAE2_QWEN_LAYERS candidate. Writes results to SAE2_REPORT_FILE.
    Returns {"scores": {layer: score}, "per_axis": {layer: {...}}, "best": layer}.
    """
    print("Loading label alignment…")
    labels, row_indices = _load_aligned_labels()
    axis_mat = _axis_score_matrix(labels)
    print(f"  {len(labels)} labeled posts aligned.")

    scores: dict[int, float] = {}
    per_axis_all: dict[int, dict] = {}

    for layer in SAE2_QWEN_LAYERS:
        print(f"\n── Layer {layer} ──────────────────────────────")
        mean_score, per_axis = score_layer(layer, labels, row_indices, axis_mat)
        scores[layer] = round(mean_score, 4)
        per_axis_all[layer] = per_axis
        print(f"  alignment score: {mean_score:.4f}")
        for ax, v in per_axis.items():
            bar = "█" * int(v * 40)
            flag = " ✓" if v >= SAE2_CONFIRM else (" ~" if v >= SAE2_PARTIAL else "")
            print(f"    {ax:20s}: {v:.3f}  {bar}{flag}")

    best_layer = max(scores, key=lambda k: scores[k])

    # ── Write report ────────────────────────────────────────────────
    report_path = APP_ROOT / SAE2_REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# SAE2 Layer Sweep Report\n"]
    lines.append(f"Candidate layers: {SAE2_QWEN_LAYERS}\n")
    lines.append(f"Labeled posts used: {len(labels)}\n")
    lines.append(f"Metric: mean over 9 axes of best-feature max(|r|, |lift|)\n\n")

    lines.append("## Summary\n\n")
    lines.append("| Layer | Alignment Score | Winner |\n")
    lines.append("|-------|----------------|--------|\n")
    for layer in SAE2_QWEN_LAYERS:
        marker = " **best**" if layer == best_layer else ""
        lines.append(f"| L{layer}  | {scores[layer]:.4f}         |{marker} |\n")

    lines.append(f"\n**Selected layer: {best_layer}**\n\n")

    lines.append("## Per-Axis Best-Feature Scores\n\n")
    header = "| Axis | " + " | ".join(f"L{l}" for l in SAE2_QWEN_LAYERS) + " |\n"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(SAE2_QWEN_LAYERS)) + "\n")
    for ax in ALL_AXIS_NAMES:
        row = f"| {ax} | "
        row += " | ".join(f"{per_axis_all[l][ax]:.3f}" for l in SAE2_QWEN_LAYERS)
        row += " |\n"
        lines.append(row)

    lines.append(f"\n*Confirm threshold: ≥{SAE2_CONFIRM}, Partial: ≥{SAE2_PARTIAL}*\n")

    report_path.write_text("".join(lines))
    print(f"\nReport written → {report_path}")

    result = {
        "scores": scores,
        "per_axis": per_axis_all,
        "best": best_layer,
    }

    print(f"\n{'='*50}")
    print(f"Best layer: {best_layer}  (score={scores[best_layer]:.4f})")
    print(f"{'='*50}")
    return result


if __name__ == "__main__":
    result = sweep()
    sys.exit(0)
