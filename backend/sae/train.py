"""
SAE training loop and activation extraction (L1 variant).

train_sae(residuals, **kwargs) → (model, history, activations)

- Standard mini-batch Adam on (MSE + L1) loss
- Decoder columns renormalized to unit norm after every step
- Returns full activation matrix (N, F) after training for downstream interp
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import SparseAutoencoder, sae_loss


def train_sae(
    residuals: np.ndarray,
    n_features: int,
    l1_coef: float,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int = 42,
    log_every: int = 25,
    device: str = "cpu",
) -> tuple[SparseAutoencoder, list[dict], np.ndarray]:
    """Train L1-regularized SAE on residuals. Returns (model, training_history, activations)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.from_numpy(residuals.astype(np.float32)).to(device)
    ds = TensorDataset(x)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = SparseAutoencoder(input_dim=residuals.shape[1],
                              n_features=n_features, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    t0 = time.time()
    for epoch in range(epochs):
        epoch_losses = {"recon": 0.0, "sparsity": 0.0, "total": 0.0}
        n_batches = 0

        for (batch_x,) in loader:
            opt.zero_grad()
            features, recon = model(batch_x)
            loss, parts = sae_loss(batch_x, recon, features, l1_coef)
            loss.backward()
            opt.step()
            model.normalize_decoder_columns()

            for k_name, v in parts.items():
                epoch_losses[k_name] += v
            n_batches += 1

        for k_name in epoch_losses:
            epoch_losses[k_name] /= max(n_batches, 1)

        # End-of-epoch diagnostics on full data
        with torch.no_grad():
            all_features, _ = model(x)
            density_per_feature = (all_features > 0).float().mean(dim=0)
            density_per_sample = (all_features > 0).float().mean(dim=1)
        epoch_losses["mean_density_feat"] = float(density_per_feature.mean())
        epoch_losses["mean_density_sample"] = float(density_per_sample.mean())
        epoch_losses["dead_features"] = int((density_per_feature == 0).sum())
        epoch_losses["epoch"] = epoch
        history.append(epoch_losses)

        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch+1:4d}/{epochs} | "
                f"recon={epoch_losses['recon']:.4f} "
                f"L1={epoch_losses['sparsity']:.3f} "
                f"density(sample)={epoch_losses['mean_density_sample']:.3f} "
                f"dead={epoch_losses['dead_features']}/{n_features}"
            )

    elapsed = time.time() - t0
    print(f"  training done in {elapsed:.1f}s")

    with torch.no_grad():
        all_features, _ = model(x)
    activations = all_features.cpu().numpy().astype(np.float32)

    return model, history, activations


def save_model(model: SparseAutoencoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "n_features": model.n_features,
    }, path)


def load_model(path: Path, device: str = "cpu") -> SparseAutoencoder:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SparseAutoencoder(input_dim=ckpt["input_dim"],
                              n_features=ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model

