"""
Compare the two best configurations across L16, L18, L22, L24:
  A) L1=0.05, k=20  — already trained as qwen{l}_knn (baseline)
  B) L1=0.04, k=25  — train now as qwen{l}_knn_k25_l0004

k=25 residual .npy files already exist from the k sweep.
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
    SAE2_N_FEATURES, SAE2_LR, SAE2_EPOCHS, SAE2_BATCH_SIZE, SAE2_SEED,
    SAE2_CONFIRM, SAE2_PARTIAL, SAE2_DEAD_DENSITY,
)
from config.axes import ALL_AXIS_NAMES
from backend.sae.train import train_sae, save_model
from backend.sae.run_variant import load_variant
from backend.sae.correlate import correlate_features_with_axes, summarize_categories
from backend.sae.layer_sweep import _alignment_score, _axis_score_matrix

LAYERS   = [16, 18, 22, 24]
L1_NEW   = 0.04
K_NEW    = 25
VARIANTS_DIR = APP_ROOT / SAE2_VARIANTS_DIR


def train_l0004_k25(layer: int, labels: pd.DataFrame, row_indices: np.ndarray) -> dict:
    vid  = f"qwen{layer}_knn_k25_l0004"
    vdir = VARIANTS_DIR / vid

    if not (vdir / "meta.json").exists():
        print(f"\n{vid}: training (l1={L1_NEW}, k={K_NEW}) …")
        x = np.load(VARIANTS_DIR / f"qwen{layer}_knn_k25.npy").astype(np.float32)

        norms = np.linalg.norm(x, axis=1)
        scale = float(np.median(norms))
        if scale < 1e-8:
            scale = 1.0
        x_scaled = (x / scale).astype(np.float32)

        print(f"  shape={x.shape}, scale={scale:.3f}")
        model, history, activations = train_sae(
            x_scaled,
            n_features=SAE2_N_FEATURES,
            l1_coef=L1_NEW,
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
            "variant_id": vid,
            "scale": scale,
            "input_dim": int(x.shape[1]),
            "n_features": SAE2_N_FEATURES,
            "hparams": {"l1_coef": L1_NEW, "lr": SAE2_LR, "epochs": SAE2_EPOCHS,
                        "batch_size": SAE2_BATCH_SIZE, "seed": SAE2_SEED},
            "final_loss": {
                "recon": round(final["recon"], 5),
                "sparsity": round(final["sparsity"], 5),
                "total": round(final["total"], 5),
                "dead_features": int(final["dead_features"]),
                "mean_density_sample": round(final["mean_density_sample"], 5),
            },
            "space": "qwen", "layer": layer, "removal": "knn",
            "knn_k": K_NEW,
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Saved → {vdir}")
    else:
        print(f"\n{vid}: already trained")

    _, _, activations = load_variant(vid)
    corr_file = vdir / "correlations.json"
    if not corr_file.exists():
        print(f"  {vid}: computing correlations …")
        records = correlate_features_with_axes(
            activations[row_indices], labels, ALL_AXIS_NAMES,
            confirm_lift=SAE2_CONFIRM, partial_lift=SAE2_PARTIAL,
            dead_density=SAE2_DEAD_DENSITY,
        )
        corr_file.write_text(json.dumps(records, indent=2))
    else:
        records = json.loads(corr_file.read_text())

    return _summarize(vid, activations, records, row_indices, labels)


def load_baseline(layer: int, labels: pd.DataFrame, row_indices: np.ndarray) -> dict:
    vid = f"qwen{layer}_knn"
    print(f"  Loading baseline {vid} …")
    _, _, activations = load_variant(vid)
    records = json.loads((VARIANTS_DIR / vid / "correlations.json").read_text())
    return _summarize(vid, activations, records, row_indices, labels)


def _summarize(vid, activations, records, row_indices, labels) -> dict:
    axis_mat = _axis_score_matrix(labels)
    score, per_axis = _alignment_score(activations[row_indices], axis_mat)
    meta = json.loads((VARIANTS_DIR / vid / "meta.json").read_text())
    fl   = meta["final_loss"]
    cats = summarize_categories(records)
    return {
        "score":    round(score, 4),
        "density":  round(fl.get("mean_density_sample", 0.0), 3),
        "dead":     fl.get("dead_features", 0),
        "recon":    round(fl.get("recon", 0.0), 4),
        "sparsity": round(fl.get("sparsity", 0.0), 4),
        "total":    round(fl.get("total", 0.0), 4),
        "conf":     cats.get("confirms_axis", 0),
        "part":     cats.get("partial_overlap", 0),
        "nov":      cats.get("novel_candidate", 0),
        "per_axis": per_axis,
    }


def print_table(results: dict):
    print("\n" + "=" * 105)
    print("%-26s %7s %8s %6s %8s %7s %7s %8s %5s %5s %5s" % (
        "Variant", "Score", "Dead", "Dead%", "Density", "Recon", "Spar", "TotLoss", "Conf", "Part", "Nov"))
    print("-" * 105)
    for layer in LAYERS:
        for label, vid in [
            ("L1=0.05 k=20", f"qwen{layer}_knn"),
            ("L1=0.04 k=25", f"qwen{layer}_knn_k25_l0004"),
        ]:
            r = results.get(vid)
            if not r:
                continue
            pct = 100 * r["dead"] / 128
            tag = f"  [{label}]"
            print("%-26s %7.4f %4d/128 %5.0f%% %8.3f %7.4f %7.4f %8.4f %5d %5d %5d" % (
                vid + tag if len(vid) < 20 else vid,
                r["score"], r["dead"], pct, r["density"],
                r["recon"], r["sparsity"], r["total"],
                r["conf"], r["part"], r["nov"]))
        print()
    print("=" * 105)

    # Per-axis comparison
    for layer in LAYERS:
        vid_a = f"qwen{layer}_knn"
        vid_b = f"qwen{layer}_knn_k25_l0004"
        if vid_a not in results or vid_b not in results:
            continue
        print(f"\nL{layer}  {'Axis':<22} {'L1=0.05 k=20':>14} {'L1=0.04 k=25':>14}  {'Δ':>8}")
        print("     " + "-" * 60)
        for ax in ALL_AXIS_NAMES:
            a = results[vid_a]["per_axis"].get(ax, 0.0)
            b = results[vid_b]["per_axis"].get(ax, 0.0)
            delta = b - a
            flag = " ▲" if delta > 0.01 else (" ▼" if delta < -0.01 else "")
            print("     %-22s %14.3f %14.3f  %+8.3f%s" % (ax, a, b, delta, flag))
        print("     " + "-" * 60)
        sa = results[vid_a]["score"]
        sb = results[vid_b]["score"]
        print("     %-22s %14.4f %14.4f  %+8.4f" % ("Mean", sa, sb, sb - sa))


def main():
    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels  = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)
    pid_to_idx = {str(p): i for i, p in enumerate(dataset["post_id"].astype(str))}
    row_indices = np.array([pid_to_idx[str(p)] for p in labels["post_id"].astype(str)], dtype=np.int64)
    print(f"Loaded {len(labels)} labeled posts.\n")

    results = {}

    # Train L1=0.04, k=25
    for layer in LAYERS:
        vid = f"qwen{layer}_knn_k25_l0004"
        results[vid] = train_l0004_k25(layer, labels, row_indices)

    # Load L1=0.05, k=20 baselines
    print("\nLoading L1=0.05, k=20 baselines …")
    for layer in LAYERS:
        vid = f"qwen{layer}_knn"
        results[vid] = load_baseline(layer, labels, row_indices)

    print_table(results)


if __name__ == "__main__":
    main()
