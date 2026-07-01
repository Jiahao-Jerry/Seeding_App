"""
Prepare local data files for the seeding app.

Builds from local SAE2 data:
  data/sae2/dataset.parquet                             → 9500 posts
  data/sae2/axis_labels.parquet                         → 1997 posts with ground-truth axis scores
  data/sae2/bge.npy                                     → (9500, 1024) BGE-M3 embeddings
  data/sae2/variants/qwen24_knn_k25_l0004/
    feature_activations.npy                             → (9500, 128) SAE activations
    correlations.json                                   → per-feature axis correlations

Produces:
  data/annotated_posts.parquet    → ALL 9500 posts with axes_json
                                     (ground-truth for 1997 labeled; SAE-estimated for the rest)
  data/corpus_embeddings.npy      → (9500, 1024) BGE embeddings
  data/contrastive_pairs.parquet  → same-topic pairs from labeled posts
  data/cross_topic_pairs.parquet  → cross-topic pairs from labeled posts
  data/sae_ridge_models.npz       → Ridge weights for SAE → axis projection (used at runtime)
"""

import json
import shutil
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SAE2 = DATA / "sae2"
SAE_VARIANT = SAE2 / "variants" / "qwen24_knn_k25_l0004"

AXIS_NAMES = [
    "reading_level", "concreteness", "narrativity", "hedging",
    "tone", "warmth", "self_disclosure", "casualness", "humor",
]


def fit_sae_ridge(acts: np.ndarray, labels: pd.DataFrame, pid_to_row: dict) -> dict:
    """
    Fit one Ridge regression per axis: SAE activations (128) → axis score (0-1).
    Trained on the 1997 labeled posts. Returns dict of {axis: (weights, bias)}.
    """
    from sklearn.linear_model import Ridge

    labeled_rows = np.array([pid_to_row[str(p)] for p in labels["post_id"].astype(str)
                             if str(p) in pid_to_row])
    label_mask = labels["post_id"].astype(str).isin(pid_to_row)
    labels_aligned = labels[label_mask].reset_index(drop=True)
    X = acts[labeled_rows]  # (n_labeled, 128)

    models = {}
    print("  Fitting Ridge regression per axis (SAE → axis score):")
    for ax in AXIS_NAMES:
        y = labels_aligned[ax].values.astype(float)
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        r2 = model.score(X, y)
        models[ax] = (model.coef_.astype(np.float32), float(model.intercept_))
        print(f"    {ax:<20} R²={r2:.3f}  (trained on {len(y)} labeled posts)")

    return models


def build_annotated_posts(acts: np.ndarray, pid_to_row: dict) -> pd.DataFrame:
    """
    Build annotated_posts.parquet with axes_json for ALL 9500 posts:
    - Labeled posts (1997): ground-truth scores from axis_labels.parquet
    - Unlabeled posts (7503): SAE Ridge-estimated scores
    """
    print("Building annotated_posts.parquet...")
    dataset = pd.read_parquet(SAE2 / "dataset.parquet")
    labels = pd.read_parquet(SAE2 / "axis_labels.parquet")
    dataset["post_id"] = dataset["post_id"].astype(str)

    # Fit Ridge models
    models = fit_sae_ridge(acts, labels, pid_to_row)

    # SAE-estimated scores for all 9500 posts
    sae_estimated = {}  # ax → (9500,) array
    for ax, (coef, intercept) in models.items():
        scores = acts @ coef + intercept
        scores = np.clip(scores, 0.0, 1.0).astype(float)
        sae_estimated[ax] = scores

    # Ground-truth scores for labeled posts
    pid_to_gt = {}
    for _, row in labels.iterrows():
        pid = str(row["post_id"])
        axes = {}
        for ax in AXIS_NAMES:
            score = row.get(ax)
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                axes[ax] = {"score": float(score), "source": "labeled"}
        if axes:
            pid_to_gt[pid] = axes

    # Build axes_json for every post
    axes_json_list = []
    for _, row in dataset.iterrows():
        pid = str(row["post_id"])
        row_idx = pid_to_row.get(pid)
        if pid in pid_to_gt:
            # Ground truth — keep as-is
            axes_json_list.append(json.dumps(pid_to_gt[pid]))
        elif row_idx is not None:
            # SAE estimate
            axes = {ax: {"score": round(float(sae_estimated[ax][row_idx]), 4), "source": "sae"}
                    for ax in AXIS_NAMES}
            axes_json_list.append(json.dumps(axes))
        else:
            axes_json_list.append(None)

    dataset["axes_json"] = axes_json_list
    out = DATA / "annotated_posts.parquet"
    dataset.to_parquet(out, index=False)

    n_gt = sum(1 for v in axes_json_list if v and '"source": "labeled"' in v)
    n_sae = sum(1 for v in axes_json_list if v and '"source": "sae"' in v)
    print(f"  {len(dataset)} posts: {n_gt} ground-truth + {n_sae} SAE-estimated = {n_gt+n_sae} with axes_json")

    # Save Ridge weights for runtime SAE profile projection
    np.savez(DATA / "sae_ridge_models.npz",
             **{f"{ax}_coef": c for ax, (c, _) in models.items()},
             **{f"{ax}_intercept": np.array([b]) for ax, (_, b) in models.items()})
    print(f"  Ridge weights → {DATA / 'sae_ridge_models.npz'}")

    return dataset


def copy_embeddings():
    print("Copying corpus_embeddings.npy...")
    src = SAE2 / "bge.npy"
    dst = DATA / "corpus_embeddings.npy"
    if not src.exists():
        print(f"  WARNING: {src} not found — skipping")
        return
    shutil.copy2(src, dst)
    arr = np.load(dst, mmap_mode="r")
    print(f"  {arr.shape} → {dst}")


def build_pairs(dataset: pd.DataFrame):
    print("Building contrastive pairs...")
    labels = pd.read_parquet(SAE2 / "axis_labels.parquet")
    labels["post_id"] = labels["post_id"].astype(str)
    valid_ids = set(dataset["post_id"].astype(str))
    labels = labels[labels["post_id"].isin(valid_ids)].copy()

    same_topic_pairs = []
    cross_topic_pairs = []

    for ax in AXIS_NAMES:
        if ax not in labels.columns:
            continue
        ax_data = labels[["post_id", "topic_name", ax]].dropna(subset=[ax])
        topics = ax_data["topic_name"].unique()

        for topic in topics:
            group = ax_data[ax_data["topic_name"] == topic].sort_values(ax)
            if len(group) < 2:
                continue
            high = group.tail(max(1, len(group) // 3))
            low = group.head(max(1, len(group) // 3))
            for _, h_row in high.iterrows():
                for _, l_row in low.iterrows():
                    gap = float(h_row[ax]) - float(l_row[ax])
                    if gap >= 0.3:
                        same_topic_pairs.append({
                            "target_axis": ax, "score": round(gap, 3),
                            "high_post_id": str(h_row["post_id"]),
                            "low_post_id": str(l_row["post_id"]),
                        })

        for t1, t2 in combinations(topics, 2):
            g1 = ax_data[ax_data["topic_name"] == t1].sort_values(ax)
            g2 = ax_data[ax_data["topic_name"] == t2].sort_values(ax)
            if len(g1) < 2 or len(g2) < 2:
                continue
            for (h, l) in [(g1.iloc[-1], g2.iloc[0]), (g2.iloc[-1], g1.iloc[0])]:
                gap = float(h[ax]) - float(l[ax])
                if gap >= 0.35:
                    cross_topic_pairs.append({
                        "target_axis": ax, "score": round(gap, 3),
                        "high_post_id": str(h["post_id"]),
                        "low_post_id": str(l["post_id"]),
                    })

    pairs_df = pd.DataFrame(same_topic_pairs).drop_duplicates(
        subset=["target_axis", "high_post_id", "low_post_id"]
    ).sort_values("score", ascending=False)
    cross_df = pd.DataFrame(cross_topic_pairs).drop_duplicates(
        subset=["target_axis", "high_post_id", "low_post_id"]
    ).sort_values("score", ascending=False)

    pairs_df.to_parquet(DATA / "contrastive_pairs.parquet", index=False)
    cross_df.to_parquet(DATA / "cross_topic_pairs.parquet", index=False)
    print(f"  {len(pairs_df)} same-topic pairs, {len(cross_df)} cross-topic pairs")
    for ax in AXIS_NAMES:
        n_s = len(pairs_df[pairs_df["target_axis"] == ax]) if len(pairs_df) else 0
        n_c = len(cross_df[cross_df["target_axis"] == ax]) if len(cross_df) else 0
        print(f"    {ax:<20} same={n_s:5d}  cross={n_c:3d}")


def main():
    DATA.mkdir(exist_ok=True)

    print(f"SAE variant: {SAE_VARIANT.name}")
    acts = np.load(SAE_VARIANT / "feature_activations.npy").astype(np.float32)
    dataset_raw = pd.read_parquet(SAE2 / "dataset.parquet")
    dataset_raw["post_id"] = dataset_raw["post_id"].astype(str)
    pid_to_row = {str(p): i for i, p in enumerate(dataset_raw["post_id"])}
    print(f"  SAE activations: {acts.shape}  dataset: {len(dataset_raw)} posts")

    dataset = build_annotated_posts(acts, pid_to_row)
    copy_embeddings()
    build_pairs(dataset)

    print("\nDone. Files in", DATA)


if __name__ == "__main__":
    main()
