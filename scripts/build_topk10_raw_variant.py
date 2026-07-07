"""
Build the qwen24_raw_topk4 variant: Top-K=4 SAE trained on raw (non-kNN-
residualized) Qwen L24 embeddings — approach #1 from SAE_verification2.docx.

1. Train on data/sae2/variants/qwen24_raw.npy (already built, no topic removal)
2. Save weights + meta.json + correlations.json (against the 1,997 labeled posts)
"""

import sys, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from config.settings import (
    SAE2_VARIANTS_DIR,
    SAE2_N_FEATURES, SAE2_LR, SAE2_EPOCHS, SAE2_BATCH_SIZE, SAE2_SEED,
    SAE2_CONFIRM, SAE2_PARTIAL, SAE2_DEAD_DENSITY,
)
from config.axes import ALL_AXIS_NAMES
from backend.sae.model_topk import TopKSparseAutoencoder, topk_loss
from backend.sae.correlate import correlate_features_with_axes
from backend.sae.layer_sweep import _load_aligned_labels

VARIANT_ID = "qwen24_raw_topk10"
K = 10
RAW_NPY = APP / "data/sae2/variants/qwen24_raw.npy"
VARIANT_DIR = APP / SAE2_VARIANTS_DIR / VARIANT_ID


def main():
    x = np.load(RAW_NPY).astype(np.float32)
    print(f"Raw embeddings: shape={x.shape}")

    norms = np.linalg.norm(x, axis=1)
    scale = float(np.median(norms))
    print(f"Median norm (scale) = {scale:.4f}  (min={norms.min():.2f}, max={norms.max():.2f})")
    x_scaled = (x / scale).astype(np.float32)

    torch.manual_seed(SAE2_SEED)
    np.random.seed(SAE2_SEED)
    x_t = torch.from_numpy(x_scaled)
    ds = TensorDataset(x_t)
    loader = DataLoader(ds, batch_size=SAE2_BATCH_SIZE, shuffle=True,
                         generator=torch.Generator().manual_seed(SAE2_SEED))

    model = TopKSparseAutoencoder(input_dim=x.shape[1], n_features=SAE2_N_FEATURES, k=K, seed=SAE2_SEED)
    opt = torch.optim.Adam(model.parameters(), lr=SAE2_LR)

    print(f"\nTraining Top-K={K} SAE ({SAE2_N_FEATURES} features, {SAE2_EPOCHS} epochs)...")
    t0 = time.time()
    history = []
    for epoch in range(SAE2_EPOCHS):
        epoch_recon = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            opt.zero_grad()
            features, recon = model(batch_x)
            loss, parts = topk_loss(batch_x, recon)
            loss.backward()
            opt.step()
            model.normalize_decoder_columns()
            epoch_recon += parts["recon"]
            n_batches += 1
        epoch_recon /= max(n_batches, 1)

        with torch.no_grad():
            all_features, _ = model(x_t)
            density_per_feature = (all_features > 0).float().mean(dim=0)
            dead = int((density_per_feature == 0).sum())
        history.append({"epoch": epoch, "recon": epoch_recon, "dead_features": dead})

        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == SAE2_EPOCHS - 1:
            print(f"  epoch {epoch+1:4d}/{SAE2_EPOCHS} | recon={epoch_recon:.4f} | dead={dead}/{SAE2_N_FEATURES}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    with torch.no_grad():
        all_features, _ = model(x_t)
    activations = all_features.numpy().astype(np.float32)

    density_per_sample = (activations > 0).mean(axis=1)
    density_per_feature = (activations > 0).mean(axis=0)
    print(f"\nDensity per sample: mean={density_per_sample.mean():.4f} (expect ~{K}/{SAE2_N_FEATURES}={K/SAE2_N_FEATURES:.4f})")
    print(f"Dead features (never fire): {int((density_per_feature == 0).sum())}/{SAE2_N_FEATURES}")

    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": model.input_dim,
                "n_features": model.n_features, "k": K}, VARIANT_DIR / "sae_model.pt")
    np.save(VARIANT_DIR / "feature_activations.npy", activations)

    final = history[-1]
    meta = {
        "object_type": "single_post",
        "variant_id": VARIANT_ID,
        "scale": scale,
        "input_dim": int(x.shape[1]),
        "n_features": SAE2_N_FEATURES,
        "sparsity_mode": "topk",
        "k": K,
        "hparams": {"lr": SAE2_LR, "epochs": SAE2_EPOCHS, "batch_size": SAE2_BATCH_SIZE, "seed": SAE2_SEED},
        "final_loss": {"recon": round(final["recon"], 5), "dead_features": final["dead_features"],
                       "mean_density_sample": round(float(density_per_sample.mean()), 5)},
        "space": "qwen", "layer": 24, "removal": "none",
    }
    (VARIANT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model + meta -> {VARIANT_DIR}")

    # ── Correlate against labeled subset ──────────────────────────
    print("\nComputing feature-axis correlations...")
    labels, row_indices = _load_aligned_labels()
    records = correlate_features_with_axes(
        activations[row_indices], labels, ALL_AXIS_NAMES,
        confirm_lift=SAE2_CONFIRM, partial_lift=SAE2_PARTIAL,
        dead_density=SAE2_DEAD_DENSITY,
    )
    (VARIANT_DIR / "correlations.json").write_text(json.dumps(records, indent=2))

    cats = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"Feature categories: {cats}")
    print(f"Saved correlations -> {VARIANT_DIR / 'correlations.json'}")


if __name__ == "__main__":
    main()
