"""
Top-K Sparse Autoencoder — hard sparsity variant, for approach #1 experiment.

Unlike the L1 variant (model.py), sparsity here is a structural constraint,
not a soft penalty: only the K largest pre-activations survive per sample,
everything else is forced to exactly zero. Loss is reconstruction MSE only —
no L1 term needed since top-k already fixes the active-feature count.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, n_features: int, k: int, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        self.input_dim = input_dim
        self.n_features = n_features
        self.k = k

        w_enc = torch.empty(n_features, input_dim)
        nn.init.kaiming_uniform_(w_enc, a=5 ** 0.5, generator=gen)
        self.W_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        w_dec = torch.randn(input_dim, n_features, generator=gen)
        w_dec = w_dec / w_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(w_dec)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) -> features: (B, F), at most k nonzero per row."""
        pre_acts = x @ self.W_enc.T + self.b_enc          # (B, F)
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=1)
        mask = torch.zeros_like(pre_acts)
        mask.scatter_(1, topk_idx, 1.0)
        return torch.relu(pre_acts) * mask

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        recon = self.decode(features)
        return features, recon

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec.data.div_(norms)


def topk_loss(x: torch.Tensor, recon: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """MSE reconstruction loss only — sparsity is structural, not penalized."""
    recon_loss = ((x - recon) ** 2).sum(dim=1).mean()
    return recon_loss, {"recon": float(recon_loss.detach()), "total": float(recon_loss.detach())}
