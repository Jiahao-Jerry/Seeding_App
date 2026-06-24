"""
SAE2 data foundation: assemble the aligned working table for the 9,500-post
curated corpus and slice its BGE-M3 vectors.

The curated corpus (phase3_curation/out/curated_corpus.parquet) carries a
`row_idx` into the repo-root embeddings.npy (the global ~2.2M BGE matrix). The
post's BGE vector is therefore a slice — no re-embedding. We verify alignment
against post_ids.npy before trusting it.

Everything downstream (representations, SAE, validation) takes its row order
from the dataset this module writes, so the corpus and the BGE matrix never
drift out of sync.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2]   # seeding_app/


def _p(rel: str) -> Path:
    """Resolve a settings path that is relative to the app root."""
    return (APP_ROOT / rel).resolve()


def build_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the curated corpus, slice its BGE vectors, verify alignment, and persist
    both. Returns (df, bge) in a single shared row order.
    """
    from config.settings import (
        SAE2_CURATED_CORPUS, SAE2_GLOBAL_EMBEDDINGS, SAE2_GLOBAL_POST_IDS,
        SAE2_DATASET_FILE, SAE2_BGE_FILE, SAE2_BGE_DIM,
    )

    df = pd.read_parquet(_p(SAE2_CURATED_CORPUS))
    df["post_id"] = df["post_id"].astype(str)
    if "row_idx" not in df.columns:
        raise ValueError("curated corpus has no 'row_idx' column — cannot slice BGE vectors.")

    emb = np.load(_p(SAE2_GLOBAL_EMBEDDINGS), mmap_mode="r")
    pids = np.load(_p(SAE2_GLOBAL_POST_IDS), allow_pickle=True)

    row_idx = df["row_idx"].to_numpy()
    if row_idx.min() < 0 or row_idx.max() >= len(emb):
        raise ValueError("row_idx out of bounds for the global embeddings array.")

    # Alignment guard: the post the embedding belongs to must be the curated post.
    aligned = np.array([str(x) for x in pids[row_idx]]) == df["post_id"].to_numpy()
    if not aligned.all():
        n_bad = int((~aligned).sum())
        raise ValueError(f"BGE alignment failed for {n_bad}/{len(df)} rows "
                         f"(post_ids[row_idx] != post_id).")

    bge = np.ascontiguousarray(emb[row_idx], dtype=np.float32)
    if bge.shape[1] != SAE2_BGE_DIM:
        raise ValueError(f"expected BGE dim {SAE2_BGE_DIM}, got {bge.shape[1]}.")

    out_df = _p(SAE2_DATASET_FILE)
    out_bge = _p(SAE2_BGE_FILE)
    out_df.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(out_df)
    np.save(out_bge, bge)
    return df.reset_index(drop=True), bge


def load_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the persisted dataset + BGE matrix, building them on first call."""
    from config.settings import SAE2_DATASET_FILE, SAE2_BGE_FILE
    df_path, bge_path = _p(SAE2_DATASET_FILE), _p(SAE2_BGE_FILE)
    if df_path.exists() and bge_path.exists():
        df = pd.read_parquet(df_path)
        bge = np.load(bge_path)
        if len(df) == len(bge):
            return df, bge
    return build_dataset()


def summary(df: pd.DataFrame, bge: np.ndarray) -> dict:
    return {
        "n_posts": int(len(df)),
        "n_topics": int(df["topic_name"].nunique()) if "topic_name" in df else None,
        "bge_shape": tuple(int(x) for x in bge.shape),
        "substance_counts": {int(k): int(v) for k, v in df["substance"].value_counts().items()}
        if "substance" in df else None,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(APP_ROOT))
    d, b = build_dataset()
    for k, v in summary(d, b).items():
        print(f"{k}: {v}")
