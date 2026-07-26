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


# Minimum |gap| on the target axis before a pair is even considered.
# Raised from the original 0.3/0.35 (same-topic/cross-topic) -- with a
# collateral-shift penalty now factored into score, a smaller target gap
# would too easily lose to the penalty term and never win regardless.
TARGET_GAP_THRESHOLD = 0.45

# How many of the OTHER 8 axes' collateral shifts get penalized -- just the
# two worst offenders, not a sum over all 8 (which could be dominated by
# noise across many weakly-correlated axes) and not just the single worst
# (which could miss a pair with two moderately-bad axes). Both posts in
# every candidate pair have all 9 axes labeled (verified: 0 missing values
# in the 1,997-post labeled set), so this never has to handle partial data.
COLLATERAL_TOP_K = 2


def _score_group_pairs(target_a: np.ndarray, other_a: np.ndarray,
                        target_b: np.ndarray, other_b: np.ndarray) -> tuple:
    """
    Vectorized: every post in group A vs every post in group B (same group
    for same-topic, since a post is never paired with itself the diagonal is
    masked out by the caller). Returns (abs_gap, a_is_high, score) matrices,
    all shape (len(A), len(B)).

    score = |target_gap| - (sum of the COLLATERAL_TOP_K largest absolute
    gaps among the other 8 axes) -- rewards a pair that differs A LOT on the
    probed axis while staying close on everything else, not just the
    biggest raw gap on the target axis alone.
    """
    gap = target_a[:, None] - target_b[None, :]                    # (nA, nB), signed
    abs_gap = np.abs(gap)
    a_is_high = gap > 0

    other_diff = np.abs(other_a[:, None, :] - other_b[None, :, :])  # (nA, nB, 8)
    top_k = np.sort(other_diff, axis=-1)[..., -COLLATERAL_TOP_K:].sum(axis=-1)

    score = abs_gap - top_k
    return abs_gap, a_is_high, score


def build_pairs(dataset: pd.DataFrame):
    """
    Contrastive pair generation. For a candidate pair to probe axis X: filter
    to |gap on X| >= TARGET_GAP_THRESHOLD, then rank survivors by
    score = |gap on X| - (sum of the 2 largest collateral gaps on the other
    8 axes) -- i.e. prefer pairs that differ a lot on X specifically while
    staying as close as possible on everything else, not just whichever
    pair has the single biggest gap on X. Applied to both same-topic and
    cross-topic candidate generation, using the FULL cross-product of posts
    (not just top/bottom-third or the single most extreme post per topic --
    see scripts/build_pairs_legacy_v1.py for the original, simpler method).
    """
    print("Building contrastive pairs (collateral-penalized scoring)...")
    labels = pd.read_parquet(SAE2 / "axis_labels.parquet")
    labels["post_id"] = labels["post_id"].astype(str)
    valid_ids = set(dataset["post_id"].astype(str))
    labels = labels[labels["post_id"].isin(valid_ids)].copy()
    labels = labels.dropna(subset=AXIS_NAMES)  # every candidate needs all 9 axes to score collateral shift

    topics = labels["topic_name"].unique()
    same_topic_pairs = []
    cross_topic_pairs = []

    for ax in AXIS_NAMES:
        other_axes = [a for a in AXIS_NAMES if a != ax]

        # Same-topic: full cross-product within each topic.
        for topic in topics:
            group = labels[labels["topic_name"] == topic]
            n = len(group)
            if n < 2:
                continue
            ids = group["post_id"].to_numpy()
            target_v = group[ax].to_numpy()
            other_v = group[other_axes].to_numpy()

            abs_gap, a_is_high, score = _score_group_pairs(target_v, other_v, target_v, other_v)
            np.fill_diagonal(abs_gap, 0)  # a post can't be paired with itself
            i_idx, j_idx = np.where(abs_gap >= TARGET_GAP_THRESHOLD)
            for i, j in zip(i_idx, j_idx):
                if i >= j:
                    continue  # (i,j) and (j,i) are the same unordered pair -- keep one
                hi, lo = (i, j) if a_is_high[i, j] else (j, i)
                same_topic_pairs.append({
                    "target_axis": ax, "score": round(float(score[i, j]), 3),
                    "high_post_id": str(ids[hi]), "low_post_id": str(ids[lo]),
                })

        # Cross-topic: full cross-product between every pair of topics.
        for t1, t2 in combinations(topics, 2):
            g1 = labels[labels["topic_name"] == t1]
            g2 = labels[labels["topic_name"] == t2]
            if len(g1) < 1 or len(g2) < 1:
                continue
            ids1, ids2 = g1["post_id"].to_numpy(), g2["post_id"].to_numpy()
            tv1, tv2 = g1[ax].to_numpy(), g2[ax].to_numpy()
            ov1, ov2 = g1[other_axes].to_numpy(), g2[other_axes].to_numpy()

            abs_gap, g1_is_high, score = _score_group_pairs(tv1, ov1, tv2, ov2)
            i_idx, j_idx = np.where(abs_gap >= TARGET_GAP_THRESHOLD)
            for i, j in zip(i_idx, j_idx):
                if g1_is_high[i, j]:
                    high_id, low_id = ids1[i], ids2[j]
                else:
                    high_id, low_id = ids2[j], ids1[i]
                cross_topic_pairs.append({
                    "target_axis": ax, "score": round(float(score[i, j]), 3),
                    "high_post_id": str(high_id), "low_post_id": str(low_id),
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
