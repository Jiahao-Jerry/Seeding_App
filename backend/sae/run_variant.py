"""
SAE2 variant trainer — train one SAE on one representation and persist it with
the metadata everything else relies on.

Reuses backend.sae.train / backend.sae.model (the existing L1 SAE). Inputs are
rescaled to unit median norm before training so the L1 penalty means the same
thing across spaces (BGE norms and Qwen norms differ by orders of magnitude).

The single most important thing this writes is meta.json's `object_type`
("single_post" or "pair_difference"): the audit refuses to run if it's fed the
wrong kind of input (validate.assert_object_type).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2]


def train_variant(variant_id: str, x: np.ndarray, object_type: str,
                  meta_extra: dict | None = None) -> Path:
    """
    variant_id   : directory name under SAE2_VARIANTS_DIR (e.g. 'qwen18_knn')
    x            : (N, D) representation matrix
    object_type  : "single_post" or "pair_difference"
    Returns the variant directory path. Saves sae_model.pt, feature_activations.npy,
    and meta.json.
    """
    import sys
    sys.path.insert(0, str(APP_ROOT))
    from config.settings import (
        SAE2_VARIANTS_DIR, SAE2_N_FEATURES, SAE2_L1_COEF,
        SAE2_LR, SAE2_EPOCHS, SAE2_BATCH_SIZE, SAE2_SEED,
    )
    from backend.sae.train import train_sae, save_model

    # Rescale to unit median norm so L1 penalty is comparable across spaces
    norms = np.linalg.norm(x, axis=1)
    scale = float(np.median(norms))
    if scale < 1e-8:
        scale = 1.0
    x_scaled = (x / scale).astype(np.float32)

    print(f"  Training SAE on {variant_id}: shape={x.shape}, scale={scale:.3f}")
    model, history, activations = train_sae(
        x_scaled,
        n_features=SAE2_N_FEATURES,
        l1_coef=SAE2_L1_COEF,
        lr=SAE2_LR,
        epochs=SAE2_EPOCHS,
        batch_size=SAE2_BATCH_SIZE,
        seed=SAE2_SEED,
    )

    out_dir = APP_ROOT / SAE2_VARIANTS_DIR / variant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    save_model(model, out_dir / "sae_model.pt")
    np.save(out_dir / "feature_activations.npy", activations)

    final = history[-1]
    meta = {
        "object_type": object_type,
        "variant_id": variant_id,
        "scale": scale,
        "input_dim": int(x.shape[1]),
        "n_features": SAE2_N_FEATURES,
        "hparams": {
            "l1_coef": SAE2_L1_COEF,
            "lr": SAE2_LR,
            "epochs": SAE2_EPOCHS,
            "batch_size": SAE2_BATCH_SIZE,
            "seed": SAE2_SEED,
        },
        "final_loss": {
            "recon": round(final["recon"], 5),
            "sparsity": round(final["sparsity"], 5),
            "total": round(final["total"], 5),
            "dead_features": final["dead_features"],
            "mean_density_sample": round(final["mean_density_sample"], 4),
        },
    }
    if meta_extra:
        meta.update(meta_extra)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  Saved → {out_dir}")
    return out_dir


def load_variant(variant_id: str) -> tuple:
    """Return (model, meta_dict, activations) for a trained variant."""
    import sys
    sys.path.insert(0, str(APP_ROOT))
    from config.settings import SAE2_VARIANTS_DIR
    from backend.sae.train import load_model

    out_dir = APP_ROOT / SAE2_VARIANTS_DIR / variant_id
    if not out_dir.exists():
        raise FileNotFoundError(f"Variant not trained yet: {out_dir}")

    meta = json.loads((out_dir / "meta.json").read_text())
    model = load_model(out_dir / "sae_model.pt")
    activations = np.load(out_dir / "feature_activations.npy")
    return model, meta, activations
