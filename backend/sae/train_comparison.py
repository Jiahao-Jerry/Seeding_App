"""
Trains SAE on 4 key representation variants at L1=0.05 and produces a
side-by-side comparison table:

  qwen24_raw   — Qwen layer-24 activations, no topic removal
  qwen24_knn   — Qwen layer-24, kNN residual (already trained; reuse)
  bge_raw      — BGE-M3 embeddings, no topic removal
  bge_knn      — BGE-M3, kNN residual

Columns: alignment score, density, dead, recon, sparsity, total loss,
         confirmed / partial / novel / dead counts.
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
from backend.sae.run_variant import train_variant, load_variant
from backend.sae.correlate import correlate_features_with_axes, summarize_categories
from backend.sae.layer_sweep import _alignment_score, _axis_score_matrix

VARIANTS_DIR = APP_ROOT / SAE2_VARIANTS_DIR

VARIANTS = [
    # (variant_id,  repr_npy,                   space,  layer, removal)
    ("qwen14_raw",  "variants/qwen14_raw.npy",  "qwen",  14,   "raw"),
    ("qwen14_knn",  "variants/qwen14_knn.npy",  "qwen",  14,   "knn"),
    ("qwen18_raw",  "variants/qwen18_raw.npy",  "qwen",  18,   "raw"),
    ("qwen18_knn",  "variants/qwen18_knn.npy",  "qwen",  18,   "knn"),
    ("qwen22_raw",  "variants/qwen22_raw.npy",  "qwen",  22,   "raw"),
    ("qwen22_knn",  "variants/qwen22_knn.npy",  "qwen",  22,   "knn"),
    ("qwen24_raw",  "variants/qwen24_raw.npy",  "qwen",  24,   "raw"),
    ("qwen24_knn",  "variants/qwen24_knn.npy",  "qwen",  24,   "knn"),
    ("bge_raw",     "variants/bge_raw.npy",     "bge",   None, "raw"),
    ("bge_knn",     "variants/bge_knn.npy",     "bge",   None, "knn"),
]


def ensure_qwen24_raw():
    """Build qwen24_raw.npy from the raw Qwen layer-24 activation file."""
    dst = APP_ROOT / "data/sae2/variants/qwen24_raw.npy"
    if dst.exists():
        return
    src = APP_ROOT / SAE2_QWEN_DIR / "qwen_L24.npy"
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    print("Building qwen24_raw.npy from qwen_L24.npy …")
    x = np.load(src).astype(np.float32)
    np.save(dst, x)
    print(f"  saved → {dst}  shape={x.shape}")


def load_aligned_labels():
    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels  = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)
    pid_to_idx = {str(p): i for i, p in enumerate(dataset["post_id"].astype(str))}
    row_indices = np.array([
        pid_to_idx[str(p)] for p in labels["post_id"].astype(str)
    ], dtype=np.int64)
    return labels, row_indices


def train_and_correlate(variant_id: str, repr_npy: str,
                        space: str, layer, removal: str,
                        labels: pd.DataFrame, row_indices: np.ndarray) -> dict:
    vdir = VARIANTS_DIR / variant_id
    corr_file = vdir / "correlations.json"

    # ── Train if needed ────────────────────────────────────────────
    if not (vdir / "meta.json").exists():
        print(f"\n{variant_id}: training (l1_coef=0.05) …")
        x = np.load(APP_ROOT / "data/sae2" / repr_npy)
        meta_extra = {"space": space, "removal": removal}
        if layer is not None:
            meta_extra["layer"] = layer
        train_variant(variant_id, x, "single_post", meta_extra=meta_extra)
    else:
        print(f"\n{variant_id}: already trained, loading …")

    # ── Correlate if needed ────────────────────────────────────────
    _, _, activations = load_variant(variant_id)
    if not corr_file.exists():
        print(f"{variant_id}: computing correlations …")
        aligned_acts = activations[row_indices]
        records = correlate_features_with_axes(
            aligned_acts, labels, ALL_AXIS_NAMES,
            confirm_lift=SAE2_CONFIRM,
            partial_lift=SAE2_PARTIAL,
            dead_density=SAE2_DEAD_DENSITY,
        )
        corr_file.write_text(json.dumps(records, indent=2))
    else:
        print(f"{variant_id}: correlations.json exists, loading …")
        records = json.loads(corr_file.read_text())

    # ── Alignment score ────────────────────────────────────────────
    aligned_acts = activations[row_indices]
    axis_mat = _axis_score_matrix(labels)
    mean_score, per_axis = _alignment_score(aligned_acts, axis_mat)

    # ── Meta stats ─────────────────────────────────────────────────
    meta = json.loads((vdir / "meta.json").read_text())
    fl = meta["final_loss"]
    cats = summarize_categories(records)

    return {
        "score": round(mean_score, 4),
        "density": round(fl.get("mean_density_sample", 0.0), 3),
        "dead": fl.get("dead_features", 0),
        "recon": round(fl.get("recon", 0.0), 4),
        "sparsity": round(fl.get("sparsity", 0.0), 4),
        "total": round(fl.get("total", 0.0), 4),
        "conf": cats.get("confirms_axis", 0),
        "part": cats.get("partial_overlap", 0),
        "nov": cats.get("novel_candidate", 0),
        "dead_cnt": cats.get("dead", 0),
        "per_axis": per_axis,
    }


def print_table(results: dict[str, dict]):
    labels_w = 12
    print("\n" + "=" * 100)
    print(f"{'Variant':<20} {'Score':>7} {'Dead':>8} {'Dead%':>6} {'Density':>8} "
          f"{'Recon':>7} {'Spar':>7} {'TotLoss':>8} {'Conf':>5} {'Part':>5} {'Nov':>5} {'Dead':>5}")
    print("-" * 100)
    for vid, r in results.items():
        pct = 100 * r["dead"] / 128
        print(f"{vid:<20} {r['score']:>7.4f} {r['dead']:>4}/128 {pct:>5.0f}% "
              f"{r['density']:>8.3f} {r['recon']:>7.4f} {r['sparsity']:>7.4f} "
              f"{r['total']:>8.4f} {r['conf']:>5} {r['part']:>5} {r['nov']:>5} {r['dead_cnt']:>5}")
    print("=" * 100)

    # Per-axis breakdown
    print("\nPer-axis alignment scores:")
    header = f"{'Axis':<22}" + "".join(f"{vid:>14}" for vid in results)
    print(header)
    print("-" * (22 + 14 * len(results)))
    for ax in ALL_AXIS_NAMES:
        row = f"{ax:<22}" + "".join(f"{results[vid]['per_axis'].get(ax, 0.0):>14.3f}" for vid in results)
        print(row)
    means = f"{'Mean':22}" + "".join(f"{results[vid]['score']:>14.4f}" for vid in results)
    print("-" * (22 + 14 * len(results)))
    print(means)


def main():
    labels, row_indices = load_aligned_labels()
    print(f"Loaded {len(labels)} labeled posts.")

    results: dict[str, dict] = {}
    for variant_id, repr_npy, space, layer, removal in VARIANTS:
        results[variant_id] = train_and_correlate(
            variant_id, repr_npy, space, layer, removal, labels, row_indices
        )

    print_table(results)


if __name__ == "__main__":
    main()
