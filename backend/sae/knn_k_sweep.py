"""
Sweep k in kNN residual across layers 16, 18, 22, 24 at L1=0.05.

For each (layer, k) pair:
  1. Build kNN residual with that k from raw Qwen activations
  2. Train SAE at L1=0.05
  3. Compute correlations
  4. Print comparison table including k=20 baselines

Variant naming: qwen{layer}_knn_k{k}  (k=20 baselines: qwen{layer}_knn)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))

from config.settings import (
    SAE2_VARIANTS_DIR, SAE2_QWEN_DIR, SAE2_LABELS_FILE, SAE2_DATASET_FILE,
    SAE2_CONFIRM, SAE2_PARTIAL, SAE2_DEAD_DENSITY,
)
from config.axes import ALL_AXIS_NAMES
from backend.sae.representations import knn_residual
from backend.sae.run_variant import train_variant, load_variant
from backend.sae.correlate import correlate_features_with_axes, summarize_categories
from backend.sae.layer_sweep import _alignment_score, _axis_score_matrix

VARIANTS_DIR = APP_ROOT / SAE2_VARIANTS_DIR
QWEN_DIR     = APP_ROOT / SAE2_QWEN_DIR

LAYERS   = [16, 18, 22, 24]
K_VALUES = [13, 17, 25]         # new; k=20 baseline already exists as qwen{l}_knn


def variant_id(layer: int, k: int) -> str:
    if k == 20:
        return f"qwen{layer}_knn"
    return f"qwen{layer}_knn_k{k}"


def repr_path(layer: int, k: int) -> Path:
    return VARIANTS_DIR / f"{variant_id(layer, k)}.npy"


def ensure_repr(layer: int, k: int) -> np.ndarray:
    """Build and cache the kNN residual .npy if not already present."""
    dst = repr_path(layer, k)
    if dst.exists():
        print(f"  repr exists: {dst.name}")
        return np.load(dst).astype(np.float32)

    src = QWEN_DIR / f"qwen_L{layer}.npy"
    print(f"  building kNN residual k={k} for L{layer} from {src.name} …")
    x = np.load(src).astype(np.float32)
    resid = knn_residual(x, k=k)
    np.save(dst, resid)
    print(f"  saved → {dst.name}  shape={resid.shape}")
    return resid


def load_aligned_labels():
    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels  = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)
    pid_to_idx = {str(p): i for i, p in enumerate(dataset["post_id"].astype(str))}
    row_indices = np.array([
        pid_to_idx[str(p)] for p in labels["post_id"].astype(str)
    ], dtype=np.int64)
    return labels, row_indices


def train_and_score(layer: int, k: int,
                    labels: pd.DataFrame, row_indices: np.ndarray) -> dict:
    vid  = variant_id(layer, k)
    vdir = VARIANTS_DIR / vid

    # ── Train ──────────────────────────────────────────────────────
    if not (vdir / "meta.json").exists():
        x = ensure_repr(layer, k)
        train_variant(vid, x, "single_post",
                      meta_extra={"space": "qwen", "layer": layer,
                                  "removal": "knn", "knn_k": k})
    else:
        print(f"  {vid}: already trained")

    # ── Correlate ──────────────────────────────────────────────────
    _, _, activations = load_variant(vid)
    corr_file = vdir / "correlations.json"
    if not corr_file.exists():
        print(f"  {vid}: computing correlations …")
        aligned = activations[row_indices]
        records = correlate_features_with_axes(
            aligned, labels, ALL_AXIS_NAMES,
            confirm_lift=SAE2_CONFIRM,
            partial_lift=SAE2_PARTIAL,
            dead_density=SAE2_DEAD_DENSITY,
        )
        corr_file.write_text(json.dumps(records, indent=2))
    else:
        records = json.loads(corr_file.read_text())

    # ── Score ──────────────────────────────────────────────────────
    aligned  = activations[row_indices]
    axis_mat = _axis_score_matrix(labels)
    mean_score, per_axis = _alignment_score(aligned, axis_mat)

    meta = json.loads((vdir / "meta.json").read_text())
    fl   = meta["final_loss"]
    cats = summarize_categories(records)

    return {
        "score":    round(mean_score, 4),
        "density":  round(fl.get("mean_density_sample", 0.0), 3),
        "dead":     fl.get("dead_features", 0),
        "recon":    round(fl.get("recon", 0.0), 4),
        "sparsity": round(fl.get("sparsity", 0.0), 4),
        "total":    round(fl.get("total", 0.0), 4),
        "conf":     cats.get("confirms_axis", 0),
        "part":     cats.get("partial_overlap", 0),
        "nov":      cats.get("novel_candidate", 0),
        "dead_cnt": cats.get("dead", 0),
        "per_axis": per_axis,
    }


def print_table(results: dict):
    all_k = [13, 17, 20, 25]
    print("\n" + "=" * 105)
    print(f"{'Variant':<22} {'Score':>7} {'Dead':>8} {'Dead%':>6} {'Density':>8} "
          f"{'Recon':>7} {'Spar':>7} {'TotLoss':>8} {'Conf':>5} {'Part':>5} {'Nov':>5}")
    print("-" * 105)
    for layer in LAYERS:
        for k in all_k:
            vid = variant_id(layer, k)
            if vid not in results:
                continue
            r   = results[vid]
            pct = 100 * r["dead"] / 128
            tag = " ◀ baseline" if k == 20 else ""
            print(f"{vid:<22} {r['score']:>7.4f} {r['dead']:>4}/128 {pct:>5.0f}% "
                  f"{r['density']:>8.3f} {r['recon']:>7.4f} {r['sparsity']:>7.4f} "
                  f"{r['total']:>8.4f} {r['conf']:>5} {r['part']:>5} {r['nov']:>5}{tag}")
        print()
    print("=" * 105)

    # Per-axis table grouped by layer
    for layer in LAYERS:
        k_cols = [k for k in all_k if variant_id(layer, k) in results]
        header = f"\nL{layer}  {'Axis':<22}" + "".join(f"{'k='+str(k):>10}" for k in k_cols)
        print(header)
        print("     " + "-" * (22 + 10 * len(k_cols)))
        for ax in ALL_AXIS_NAMES:
            row = f"     {ax:<22}" + "".join(
                f"{results[variant_id(layer, k)]['per_axis'].get(ax, 0.0):>10.3f}"
                for k in k_cols
            )
            print(row)
        means = f"     {'Mean':<22}" + "".join(
            f"{results[variant_id(layer, k)]['score']:>10.4f}" for k in k_cols
        )
        print("     " + "-" * (22 + 10 * len(k_cols)))
        print(means)


def main():
    labels, row_indices = load_aligned_labels()
    print(f"Loaded {len(labels)} labeled posts.\n")

    results: dict[str, dict] = {}

    # New k values
    for layer in LAYERS:
        for k in K_VALUES:
            vid = variant_id(layer, k)
            print(f"\n── {vid} ──────────────────────────────────────")
            results[vid] = train_and_score(layer, k, labels, row_indices)

    # Load k=20 baselines (already trained)
    print("\nLoading k=20 baselines …")
    for layer in LAYERS:
        vid = variant_id(layer, 20)
        print(f"  {vid}")
        results[vid] = train_and_score(layer, 20, labels, row_indices)

    print_table(results)


if __name__ == "__main__":
    main()
