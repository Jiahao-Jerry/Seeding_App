"""
SAE2 representations: turn the corpus into the matrix an SAE actually trains on.

Two orthogonal levers (see §3 of docs/style_sae_handover.md):

  space         "bge"           BGE-M3 (1024-d), the retrieval-embedding baseline
                "qwen@<layer>"  Qwen2.5-7B mean-pooled activation at a layer (3584-d)

  topic removal "raw"           whole vectors (SAE separates style from topic itself)
                "knn"           subtract the mean of each post's k nearest neighbors
                                (fluid topics — no labels, fully decoupled)

A variant id is "<space>_<removal>", e.g. "bge_knn", "qwen18_raw".

The kNN residual is the workhorse: because a residual is itself a difference
(post minus its neighborhood), the same object serves both the user model and
the audit. The cheap correctness signal is that kNN residual norms are clearly
smaller than raw norms — the subtraction removed shared local signal.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[2]


def _p(rel: str) -> Path:
    return (APP_ROOT / rel).resolve()


def load_space(space: str) -> np.ndarray:
    """
    space: "bge" or "qwen<layer>" (e.g. "qwen18"). Returns an (N, D) float32 matrix
    aligned with dataset.load_dataset()'s row order.
    """
    from config.settings import SAE2_BGE_FILE, SAE2_QWEN_DIR

    if space == "bge":
        return np.load(_p(SAE2_BGE_FILE)).astype(np.float32)

    m = re.fullmatch(r"qwen(\d+)", space)
    if m:
        layer = int(m.group(1))
        path = _p(SAE2_QWEN_DIR) / f"qwen_L{layer}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run the Qwen multilayer extraction first."
            )
        return np.load(path).astype(np.float32)

    raise ValueError(f"unknown space '{space}' (expected 'bge' or 'qwen<layer>').")


def knn_residual(x: np.ndarray, k: int, chunk: int = 1000) -> np.ndarray:
    """
    For each row, subtract the mean of its k nearest neighbors by cosine similarity
    (self excluded). Computed in row-chunks to bound memory on the full N×N similarity.
    """
    n = x.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than N={n}.")

    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    resid = np.empty_like(x, dtype=np.float32)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = xn[start:end] @ xn.T            # (chunk, N) cosine similarities
        for local, global_i in enumerate(range(start, end)):
            sims[local, global_i] = -np.inf    # exclude self
        nn = np.argpartition(-sims, kth=k, axis=1)[:, :k]   # top-k neighbor indices
        neigh_mean = x[nn].mean(axis=1)        # (chunk, D)
        resid[start:end] = x[start:end] - neigh_mean

    return resid.astype(np.float32)


def build_representation(variant_id: str) -> np.ndarray:
    """variant_id = '<space>_<removal>', e.g. 'bge_knn', 'qwen18_raw'."""
    from config.settings import SAE2_KNN_K

    try:
        space, removal = variant_id.rsplit("_", 1)
    except ValueError:
        raise ValueError(f"bad variant id '{variant_id}' (want '<space>_<removal>').")

    x = load_space(space)
    if removal == "raw":
        return x
    if removal == "knn":
        return knn_residual(x, k=SAE2_KNN_K)
    raise ValueError(f"unknown removal '{removal}' (expected 'raw' or 'knn').")


def norms_summary(x: np.ndarray) -> dict:
    nrm = np.linalg.norm(x, axis=1)
    return {
        "shape": tuple(int(v) for v in x.shape),
        "median_norm": float(np.median(nrm)),
        "mean_norm": float(nrm.mean()),
    }
