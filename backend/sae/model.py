"""
Sparse Autoencoder model — L1-regularized variant.

Architecture
- encoder:  features = ReLU(W_enc @ x + b_enc),  shape (F,)
- decoder:  x_hat    = W_dec @ features + b_dec,  shape (D,)

Untied weights. Decoder columns are renormalized to unit norm after each
optimizer step (prevents trivial collapse: feature magnitude shrinks while
the corresponding decoder column grows to compensate).

Loss
- L = MSE(x, x_hat) + l1_coef * ||features||_1     (per-sample, mean over batch)

Why L1 (not TopK) for this setup
- BGE-M3 + small data (2550 posts, 1024 dim): TopK fragments composite registers
  by forcing exactly k active features per post regardless of register clarity.
- L1 allows variable per-sample density — easy posts use few features, register-y
  posts use many. Empirically yields more nameable composite features for this data.
- (TopK is mathematically cleaner and is the default in OpenAI/Anthropic SAE papers
  for LLM hidden states; revisit if we switch to Option 2 LLM activations.)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, n_features: int, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        self.input_dim = input_dim
        self.n_features = n_features

        w_enc = torch.empty(n_features, input_dim)
        nn.init.kaiming_uniform_(w_enc, a=5 ** 0.5, generator=gen)
        self.W_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        w_dec = torch.randn(input_dim, n_features, generator=gen)
        w_dec = w_dec / w_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(w_dec)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) → features: (B, F), non-negative."""
        return torch.relu(x @ self.W_enc.T + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        recon = self.decode(features)
        return features, recon

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        """Renormalize each decoder column (feature direction) to unit norm."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.data.div_(norms)


def sae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    features: torch.Tensor,
    l1_coef: float,
) -> tuple[torch.Tensor, dict]:
    """MSE reconstruction loss + L1 sparsity penalty on features."""
    recon_loss = ((x - recon) ** 2).sum(dim=1).mean()
    sparsity = features.abs().sum(dim=1).mean()
    loss = recon_loss + l1_coef * sparsity
    return loss, {
        "recon": float(recon_loss.detach()),
        "sparsity": float(sparsity.detach()),
        "total": float(loss.detach()),
    }

