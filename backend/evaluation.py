"""
Evaluation engine: constructs pairwise trials from the corpus + user profile.

Given a completed seeding session, this module:
1. Computes the user's interest centroid (mean embedding of chosen posts)
2. Computes preferred axis values (mean of chosen posts' axis scores)
3. Ranks all posts by cosine similarity → assigns distance bands
4. Constructs trial pairs (Type A, B, C) with appropriate controls
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal

from config.axes import ALL_AXIS_NAMES

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Load once ─────────────────────────────────────────────────────

_corpus: pd.DataFrame | None = None
_embeddings: np.ndarray | None = None


def _load():
    global _corpus, _embeddings
    if _corpus is None:
        _corpus = pd.read_parquet(DATA_DIR / "annotated_posts.parquet")
        _embeddings = np.load(DATA_DIR / "corpus_embeddings.npy")
    return _corpus, _embeddings


# ── User profile from seeding ─────────────────────────────────────

def compute_user_centroid(liked_post_ids: list[str]) -> np.ndarray:
    """Mean embedding of posts the user chose during seeding."""
    corpus, embeddings = _load()
    mask = corpus["post_id"].astype(str).isin([str(x) for x in liked_post_ids])
    indices = np.where(mask.values)[0]
    if len(indices) == 0:
        return embeddings.mean(axis=0)
    return embeddings[indices].mean(axis=0)


def compute_preferred_axes(liked_post_ids: list[str]) -> dict[str, float]:
    """Mean axis score across posts the user chose. This IS the preference."""
    corpus, _ = _load()
    mask = corpus["post_id"].astype(str).isin([str(x) for x in liked_post_ids])
    liked = corpus[mask]

    axis_names = ALL_AXIS_NAMES
    prefs = {}
    for _, row in liked.iterrows():
        axes = json.loads(row["axes_json"]) if pd.notna(row.get("axes_json")) else {}
        for ax in axis_names:
            val = axes.get(ax, {})
            score = val.get("score") if isinstance(val, dict) else val
            if score is not None:
                prefs.setdefault(ax, []).append(score)

    return {ax: np.mean(scores) for ax, scores in prefs.items()}


# ── Distance bands ────────────────────────────────────────────────

def rank_by_similarity(centroid: np.ndarray, exclude_ids: set[str] = None):
    """Rank all posts by cosine similarity to centroid. Returns df with sim column."""
    corpus, embeddings = _load()

    # Cosine similarity
    norm_c = centroid / (np.linalg.norm(centroid) + 1e-9)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    sims = (embeddings / norms) @ norm_c

    ranked = corpus.copy()
    ranked["similarity"] = sims
    ranked = ranked.sort_values("similarity", ascending=False).reset_index(drop=True)

    if exclude_ids:
        ranked = ranked[~ranked["post_id"].astype(str).isin(exclude_ids)]

    # Assign bands based on percentile rank
    n = len(ranked)
    ranked["band"] = "far"
    ranked.iloc[:int(n * 0.15), ranked.columns.get_loc("band")] = "near"
    ranked.iloc[int(n * 0.15):int(n * 0.45), ranked.columns.get_loc("band")] = "mid"

    return ranked


# ── Trial construction ────────────────────────────────────────────

@dataclass
class Trial:
    trial_id: str
    trial_type: Literal["A", "B", "C"]  # A=same-band, B=reach, C=same-post
    left_post_id: str
    right_post_id: str
    left_is_transformed: bool
    right_is_transformed: bool
    left_band: str
    right_band: str
    transform_axes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _pick_transform_axes(post_axes: dict, user_prefs: dict, max_axes: int = 2) -> list[str]:
    """Pick the 1-2 axes with largest gap between post and user preference."""
    gaps = []
    for ax, pref_val in user_prefs.items():
        post_val = post_axes.get(ax, {})
        score = post_val.get("score") if isinstance(post_val, dict) else post_val
        if score is not None:
            gaps.append((ax, abs(pref_val - score)))

    gaps.sort(key=lambda x: x[1], reverse=True)
    # Only include axes with meaningful gap (>0.15)
    return [ax for ax, gap in gaps[:max_axes] if gap > 0.15]


def construct_trials(
    centroid: np.ndarray,
    user_prefs: dict,
    seen_ids: set[str],
    n_type_a: int = 8,
    n_type_b: int = 8,
    n_type_c: int = 4,
    seed: int = 42,
    prefer_high_gap: bool = True,
    diversify_topics: bool = True,
) -> list[Trial]:
    """Build a balanced set of evaluation trials."""
    rng = np.random.default_rng(seed)
    ranked = rank_by_similarity(centroid, exclude_ids=seen_ids)

    # Compute per-post style gap (sum of top-2 axis gaps) for smart selection
    def _style_gap(row):
        axes = json.loads(row["axes_json"]) if pd.notna(row.get("axes_json")) else {}
        gaps = []
        for ax, pref_val in user_prefs.items():
            post_val = axes.get(ax, {})
            score = post_val.get("score") if isinstance(post_val, dict) else post_val
            if score is not None:
                gaps.append(abs(pref_val - score))
        gaps.sort(reverse=True)
        return sum(gaps[:2])  # top-2 axis gap magnitude

    if prefer_high_gap:
        ranked["style_gap"] = ranked.apply(_style_gap, axis=1)
        # Sort within each band by style gap (descending) — prefer posts where
        # transformation will be most visible
        ranked = ranked.sort_values(["band", "style_gap"], ascending=[True, False])

    near = ranked[ranked["band"] == "near"]
    mid = ranked[ranked["band"] == "mid"]

    # Topic-diversified sampling
    def _diverse_sample(df, n, rng_seed):
        if not diversify_topics or len(df) <= n:
            return df.head(n) if prefer_high_gap else df.sample(min(n, len(df)), random_state=rng_seed)
        # Round-robin across topics, preferring high-gap within each topic
        topics = df["topic_name"].unique()
        per_topic = max(1, n // len(topics))
        selected = []
        for topic in topics:
            topic_df = df[df["topic_name"] == topic]
            selected.append(topic_df.head(per_topic))
        result = pd.concat(selected)
        # Fill remaining slots from whatever has highest gap
        if len(result) < n:
            remaining = df[~df.index.isin(result.index)].head(n - len(result))
            result = pd.concat([result, remaining])
        return result.head(n)

    trials = []
    trial_idx = 0
    used_in_trials: set[str] = set()  # track posts used across all trial types

    def _get_axes(row):
        return json.loads(row["axes_json"]) if pd.notna(row.get("axes_json")) else {}

    def _fresh(df: pd.DataFrame) -> pd.DataFrame:
        """Filter out posts already used in earlier trial types."""
        return df[~df["post_id"].astype(str).isin(used_in_trials)]

    # ── Type A: same band, one transformed ────────────────────────
    for band_df, band_name, count in [(near, "near", n_type_a // 2), (mid, "mid", n_type_a // 2)]:
        pool = _fresh(band_df)
        if len(pool) < count * 2:
            count = len(pool) // 2
        sample = _diverse_sample(pool, count * 2, seed)
        pairs = [(sample.iloc[i], sample.iloc[i + count]) for i in range(count)]

        for original_row, transform_row in pairs:
            axes = _get_axes(transform_row)
            tx_axes = _pick_transform_axes(axes, user_prefs)
            if not tx_axes:
                continue

            used_in_trials.add(str(original_row["post_id"]))
            used_in_trials.add(str(transform_row["post_id"]))

            if rng.random() < 0.5:
                trials.append(Trial(
                    trial_id=f"A-{trial_idx}",
                    trial_type="A",
                    left_post_id=str(original_row["post_id"]),
                    right_post_id=str(transform_row["post_id"]),
                    left_is_transformed=False,
                    right_is_transformed=True,
                    left_band=band_name,
                    right_band=band_name,
                    transform_axes=tx_axes,
                ))
            else:
                trials.append(Trial(
                    trial_id=f"A-{trial_idx}",
                    trial_type="A",
                    left_post_id=str(transform_row["post_id"]),
                    right_post_id=str(original_row["post_id"]),
                    left_is_transformed=True,
                    right_is_transformed=False,
                    left_band=band_name,
                    right_band=band_name,
                    transform_axes=tx_axes,
                ))
            trial_idx += 1

    # ── Type B: transformed-mid vs original-near ──────────────────
    n_b = min(n_type_b, len(_fresh(near)), len(_fresh(mid)))
    near_sample = _diverse_sample(_fresh(near), n_b, seed + 1)
    mid_sample = _diverse_sample(_fresh(mid), n_b, seed + 2)

    for i in range(min(n_b, len(near_sample), len(mid_sample))):
        near_row = near_sample.iloc[i]
        mid_row = mid_sample.iloc[i]
        axes = _get_axes(mid_row)
        tx_axes = _pick_transform_axes(axes, user_prefs)
        if not tx_axes:
            continue

        used_in_trials.add(str(near_row["post_id"]))
        used_in_trials.add(str(mid_row["post_id"]))

        if rng.random() < 0.5:
            trials.append(Trial(
                trial_id=f"B-{trial_idx}",
                trial_type="B",
                left_post_id=str(near_row["post_id"]),
                right_post_id=str(mid_row["post_id"]),
                left_is_transformed=False,
                right_is_transformed=True,
                left_band="near",
                right_band="mid",
                transform_axes=tx_axes,
            ))
        else:
            trials.append(Trial(
                trial_id=f"B-{trial_idx}",
                trial_type="B",
                left_post_id=str(mid_row["post_id"]),
                right_post_id=str(near_row["post_id"]),
                left_is_transformed=True,
                right_is_transformed=False,
                left_band="mid",
                right_band="near",
                transform_axes=tx_axes,
            ))
        trial_idx += 1

    # ── Type C: same post, two versions ───────────────────────────
    c_pool_df = _fresh(pd.concat([near, mid]))
    c_pool = c_pool_df.sample(min(n_type_c, len(c_pool_df)),
                              random_state=seed + 3, replace=False)
    for i in range(len(c_pool)):
        row = c_pool.iloc[i]
        axes = _get_axes(row)
        tx_axes = _pick_transform_axes(axes, user_prefs)
        if not tx_axes:
            continue

        used_in_trials.add(str(row["post_id"]))
        band = row.get("band", "mid") if "band" in c_pool.columns else "mid"
        if rng.random() < 0.5:
            trials.append(Trial(
                trial_id=f"C-{trial_idx}",
                trial_type="C",
                left_post_id=str(row["post_id"]),
                right_post_id=str(row["post_id"]),  # same post, two versions
                left_is_transformed=False,
                right_is_transformed=True,
                left_band=band,
                right_band=band,
                transform_axes=tx_axes,
            ))
        else:
            trials.append(Trial(
                trial_id=f"C-{trial_idx}",
                trial_type="C",
                left_post_id=str(row["post_id"]),
                right_post_id=str(row["post_id"]),
                left_is_transformed=True,
                right_is_transformed=False,
                left_band=band,
                right_band=band,
                transform_axes=tx_axes,
            ))
        trial_idx += 1

    rng.shuffle(trials)
    return trials
