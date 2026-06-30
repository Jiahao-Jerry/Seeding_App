"""
Train SAE with 256 features at L24, L1=0.04, k=25.
Compare against 128-feature baseline (qwen24_knn_k25_l0004).
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
    SAE2_VARIANTS_DIR, SAE2_LABELS_FILE, SAE2_DATASET_FILE,
    SAE2_LR, SAE2_EPOCHS, SAE2_BATCH_SIZE, SAE2_SEED,
    SAE2_CONFIRM, SAE2_PARTIAL, SAE2_DEAD_DENSITY,
)
from config.axes import ALL_AXIS_NAMES
from backend.sae.train import train_sae, save_model
from backend.sae.run_variant import load_variant
from backend.sae.correlate import correlate_features_with_axes, summarize_categories
from backend.sae.layer_sweep import _alignment_score, _axis_score_matrix

LAYER    = 24
L1_COEF  = 0.04
K        = 25
N_FEATS  = 256
VID      = f"qwen{LAYER}_knn_k{K}_l0004_f{N_FEATS}"
BASELINE = f"qwen{LAYER}_knn_k{K}_l0004"

VARIANTS_DIR = APP_ROOT / SAE2_VARIANTS_DIR


def train_256(labels, row_indices):
    vdir = VARIANTS_DIR / VID
    if not (vdir / "meta.json").exists():
        print(f"\n{VID}: training (n_features={N_FEATS}, l1={L1_COEF}, k={K}) …")
        x = np.load(VARIANTS_DIR / f"qwen{LAYER}_knn_k{K}.npy").astype(np.float32)

        norms = np.linalg.norm(x, axis=1)
        scale = float(np.median(norms))
        if scale < 1e-8:
            scale = 1.0
        x_scaled = (x / scale).astype(np.float32)
        print(f"  shape={x.shape}, scale={scale:.3f}")

        model, history, activations = train_sae(
            x_scaled,
            n_features=N_FEATS,
            l1_coef=L1_COEF,
            lr=SAE2_LR,
            epochs=SAE2_EPOCHS,
            batch_size=SAE2_BATCH_SIZE,
            seed=SAE2_SEED,
        )
        vdir.mkdir(parents=True, exist_ok=True)
        save_model(model, vdir / "sae_model.pt")
        np.save(vdir / "feature_activations.npy", activations)

        final = history[-1]
        meta = {
            "object_type": "single_post",
            "variant_id": VID,
            "scale": scale,
            "input_dim": int(x.shape[1]),
            "n_features": N_FEATS,
            "hparams": {"l1_coef": L1_COEF, "lr": SAE2_LR, "epochs": SAE2_EPOCHS,
                        "batch_size": SAE2_BATCH_SIZE, "seed": SAE2_SEED},
            "final_loss": {
                "recon":             round(final["recon"], 5),
                "sparsity":          round(final["sparsity"], 5),
                "total":             round(final["total"], 5),
                "dead_features":     int(final["dead_features"]),
                "mean_density_sample": round(final["mean_density_sample"], 5),
            },
            "space": "qwen", "layer": LAYER, "removal": "knn", "knn_k": K,
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Saved → {vdir}")
    else:
        print(f"\n{VID}: already trained")

    _, _, activations = load_variant(VID)
    corr_file = vdir / "correlations.json"
    if not corr_file.exists():
        print(f"  {VID}: computing correlations …")
        records = correlate_features_with_axes(
            activations[row_indices], labels, ALL_AXIS_NAMES,
            confirm_lift=SAE2_CONFIRM, partial_lift=SAE2_PARTIAL,
            dead_density=SAE2_DEAD_DENSITY,
        )
        corr_file.write_text(json.dumps(records, indent=2))
    else:
        records = json.loads(corr_file.read_text())

    axis_mat = _axis_score_matrix(labels)
    score, per_axis = _alignment_score(activations[row_indices], axis_mat)
    meta = json.loads((vdir / "meta.json").read_text())
    fl   = meta["final_loss"]
    cats = summarize_categories(records)
    return {
        "score":    round(score, 4),
        "density":  round(fl["mean_density_sample"], 3),
        "dead":     fl["dead_features"],
        "recon":    round(fl["recon"], 4),
        "sparsity": round(fl["sparsity"], 4),
        "total":    round(fl["total"], 4),
        "conf":     cats.get("confirms_axis", 0),
        "part":     cats.get("partial_overlap", 0),
        "nov":      cats.get("novel_candidate", 0),
        "dead_cnt": cats.get("dead", 0),
        "per_axis": per_axis,
        "records":  records,
    }


def load_baseline(labels, row_indices):
    _, _, activations = load_variant(BASELINE)
    records = json.loads((VARIANTS_DIR / BASELINE / "correlations.json").read_text())
    axis_mat = _axis_score_matrix(labels)
    score, per_axis = _alignment_score(activations[row_indices], axis_mat)
    meta = json.loads((VARIANTS_DIR / BASELINE / "meta.json").read_text())
    fl   = meta["final_loss"]
    cats = summarize_categories(records)
    return {
        "score":    round(score, 4),
        "density":  round(fl["mean_density_sample"], 3),
        "dead":     fl["dead_features"],
        "recon":    round(fl["recon"], 4),
        "sparsity": round(fl["sparsity"], 4),
        "total":    round(fl["total"], 4),
        "conf":     cats.get("confirms_axis", 0),
        "part":     cats.get("partial_overlap", 0),
        "nov":      cats.get("novel_candidate", 0),
        "dead_cnt": cats.get("dead", 0),
        "per_axis": per_axis,
        "records":  records,
    }


def print_results(r128, r256):
    n128, n256 = 128, 256
    print("\n" + "=" * 100)
    print("%-30s %7s %8s %6s %8s %7s %7s %8s %5s %5s %5s" % (
        "Variant", "Score", "Dead", "Dead%", "Density", "Recon", "Spar", "TotLoss", "Conf", "Part", "Nov"))
    print("-" * 100)
    for label, r, n in [("128f  qwen24_knn_k25_l0004", r128, n128),
                        ("256f  qwen24_knn_k25_l0004_f256", r256, n256)]:
        pct = 100 * r["dead"] / n
        print("%-30s %7.4f %4d/%-3d %5.0f%% %8.3f %7.4f %7.4f %8.4f %5d %5d %5d" % (
            label, r["score"], r["dead"], n, pct, r["density"],
            r["recon"], r["sparsity"], r["total"],
            r["conf"], r["part"], r["nov"]))
    print("=" * 100)

    # Per-axis breakdown
    print("\n%-22s %14s %14s %8s" % ("Axis", "128 features", "256 features", "Δ"))
    print("-" * 60)
    for ax in ALL_AXIS_NAMES:
        a = r128["per_axis"].get(ax, 0.0)
        b = r256["per_axis"].get(ax, 0.0)
        flag = " ▲" if b - a > 0.01 else (" ▼" if b - a < -0.01 else "")
        print("%-22s %14.3f %14.3f %+8.3f%s" % (ax, a, b, b - a, flag))
    print("-" * 60)
    print("%-22s %14.4f %14.4f %+8.4f" % ("Mean", r128["score"], r256["score"], r256["score"] - r128["score"]))

    # Per-axis confirmed features breakdown
    print("\nPer-axis confirmed features (axis_labels count, score >= 0.2):")
    print("%-22s %14s %14s" % ("Axis", "128 features", "256 features"))
    print("-" * 52)
    def axis_coverage(records, threshold=0.2):
        counts = {ax: 0 for ax in ALL_AXIS_NAMES}
        for feat in records:
            if feat["category"] == "dead":
                continue
            for ax in ALL_AXIS_NAMES:
                r_val = abs(feat["correlations"].get(ax, 0))
                l_val = abs(feat["lifts"].get(ax, 0))
                if max(r_val, l_val) >= threshold:
                    counts[ax] += 1
        return counts
    cov128 = axis_coverage(r128["records"])
    cov256 = axis_coverage(r256["records"])
    for ax in ALL_AXIS_NAMES:
        print("%-22s %14d %14d" % (ax, cov128[ax], cov256[ax]))


def main():
    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels  = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)
    pid_to_idx  = {str(p): i for i, p in enumerate(dataset["post_id"].astype(str))}
    row_indices = np.array([pid_to_idx[str(p)] for p in labels["post_id"].astype(str)], dtype=np.int64)
    print(f"Loaded {len(labels)} labeled posts.")

    r256 = train_256(labels, row_indices)
    print("\nLoading 128-feature baseline …")
    r128 = load_baseline(labels, row_indices)
    print_results(r128, r256)


if __name__ == "__main__":
    main()
