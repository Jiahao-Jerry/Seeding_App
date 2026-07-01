"""
SAE2 validation harness — how we know the fingerprint works.

Three jobs (see §6 of docs/style_sae_handover.md):

  coverage(...)         For each of the 9 axes, does at least one SAE feature
                        track it? Judged against LLM axis-score labels using
                        max(|Pearson r|, |lift|) + ROC-AUC on top/bottom quartiles.
                        Uses axes.py `measurability` to set expectations:
                        high → must confirm; low (humor) → reported as bellwether.

  pair_separation(...)  Per axis:
                        (a) LLM-score shift — did the rewriter actually move the
                            axis? (target shift + direction consistency from
                            purity-gate cache)
                        (b) Base-activation proxy — does the best feature's
                            activation on the BASE post predict rewrite direction?
                            (ROC-AUC; higher activation → more room to go down)
                        NOTE: SAE activation gap on SHIFTED texts (the gold-standard
                        pair separation) requires encoding shifted texts through
                        Qwen2.5-7B. See TODO at bottom of this file.

  assert_object_type    Hard guard for the train/infer invariant (§2 of handover):
                        an SAE trained on single posts must never be fed differences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.sae.correlate import correlate_features_with_axes


# ── Object-type guard ──────────────────────────────────────────────────────────

def assert_object_type(meta: dict, input_kind: str) -> None:
    """Raise if a model is asked to encode the wrong kind of input."""
    trained = meta.get("object_type")
    if trained is None:
        raise ValueError("model meta is missing 'object_type'.")
    if trained != input_kind:
        raise ValueError(
            f"object-type mismatch: model was trained on '{trained}' but received "
            f"'{input_kind}'. f(A-B) != f(A)-f(B); feed the trained object type."
        )


# ── Coverage ───────────────────────────────────────────────────────────────────

def coverage(
    activations: np.ndarray,
    labels: pd.DataFrame,
    axis_names: list[str],
    confirm: float,
    partial: float,
    axes_meta: list[dict] | None = None,
) -> dict[str, Any]:
    """
    For each axis: find the best feature by max(|r|, |lift|), classify as
    confirmed/partial/uncovered, and compute ROC-AUC on top-25% vs bottom-25%
    labeled posts.

    activations : (N, F) aligned to labeled rows
    labels      : DataFrame with one float column per axis name
    axes_meta   : list of axis dicts from config/axes.py (for measurability)
    """
    records = correlate_features_with_axes(
        activations, labels, axis_names,
        confirm_lift=confirm, partial_lift=partial,
    )
    measurability = {}
    if axes_meta:
        measurability = {a["name"]: a.get("measurability", "medium") for a in axes_meta}

    per_axis: dict[str, Any] = {}
    for ax in axis_names:
        # Best feature for this axis
        best_score, best_feat, best_r, best_lift = 0.0, None, 0.0, 0.0
        for rec in records:
            if rec["category"] == "dead":
                continue
            r    = rec["correlations"].get(ax, 0.0)
            lift = rec["lifts"].get(ax, 0.0)
            s    = max(abs(r), abs(lift))
            if s > best_score:
                best_score, best_feat, best_r, best_lift = s, rec["feature"], r, lift

        # Label-based ROC-AUC (top 25% vs bottom 25%)
        auc = None
        if best_feat is not None and ax in labels.columns:
            ax_vals  = labels[ax].to_numpy(dtype=np.float64)
            feat_col = activations[:, best_feat]
            valid    = ~np.isnan(ax_vals)
            ax_v, f_v = ax_vals[valid], feat_col[valid]
            q25, q75  = np.percentile(ax_v, [25, 75])
            mask = (ax_v <= q25) | (ax_v >= q75)
            if mask.sum() >= 10:
                y_true  = (ax_v[mask] >= q75).astype(int)
                y_score = f_v[mask]
                try:
                    auc = float(roc_auc_score(y_true, y_score))
                    if auc < 0.5:
                        auc = 1.0 - auc  # flip sign if feature is negatively correlated
                except Exception:
                    pass

        if best_score >= confirm:
            status = "confirmed"
        elif best_score >= partial:
            status = "partial"
        else:
            status = "uncovered"

        per_axis[ax] = {
            "best_feature":  best_feat,
            "best_r":        round(best_r, 3),
            "best_lift":     round(best_lift, 3),
            "score":         round(best_score, 3),
            "status":        status,
            "measurability": measurability.get(ax, "medium"),
            "label_auc":     round(auc, 3) if auc is not None else None,
        }

    confirmed = sum(1 for v in per_axis.values() if v["status"] == "confirmed")
    return {"per_axis": per_axis, "confirmed": confirmed, "total": len(axis_names)}


# ── Pair separation ────────────────────────────────────────────────────────────

def pair_separation(
    pairs: pd.DataFrame,
    correlations: list[dict],
    activations_full: np.ndarray,
    dataset: pd.DataFrame,
    purity_cache: dict,
    axis_names: list[str],
) -> dict[str, Any]:
    """
    Per axis:
    (a) LLM-score shift from purity cache — mean target shift + direction consistency.
    (b) Base-activation AUC — does best feature activation on base post predict
        which direction the rewrite went?

    Gold-standard SAE activation gap (shifted - base) is deferred until shifted
    texts are encoded through Qwen2.5-7B (see TODO).
    """
    post_id_to_idx = {str(pid): i for i, pid in enumerate(dataset["post_id"].astype(str))}

    # Best feature per axis (by score, skipping dead features)
    best_feat_per_axis: dict[str, int] = {}
    for ax in axis_names:
        best_s, best_f = 0.0, 0
        for rec in correlations:
            if rec["category"] == "dead":
                continue
            s = max(abs(rec["correlations"].get(ax, 0)), abs(rec["lifts"].get(ax, 0)))
            if s > best_s:
                best_s, best_f = s, rec["feature"]
        best_feat_per_axis[ax] = best_f

    kept = pairs[pairs["kept"] == True]
    per_axis: dict[str, Any] = {}

    for ax in axis_names:
        sub = kept[kept["axis"] == ax]
        if len(sub) == 0:
            per_axis[ax] = {"n_pairs": 0, "note": "no kept pairs"}
            continue

        best_feat = best_feat_per_axis[ax]
        shifts, dir_correct, base_acts, dir_labels = [], [], [], []

        for _, row in sub.iterrows():
            pair_id   = row["pair_id"]
            direction = row["direction"]
            base_pid  = str(row["base_post_id"])

            # (a) LLM shift
            purity = purity_cache.get(f"purity:{pair_id}")
            if purity:
                b = purity["base_scores"].get(ax, 0.5)
                s = purity["shift_scores"].get(ax, 0.5)
                shifts.append(s - b)
                dir_correct.append((direction == "up" and s > b) or (direction == "down" and s < b))

            # (b) Base activation
            if base_pid in post_id_to_idx:
                idx = post_id_to_idx[base_pid]
                base_acts.append(float(activations_full[idx, best_feat]))
                dir_labels.append(1 if direction == "up" else 0)

        mean_shift  = float(np.mean(shifts))   if shifts else None
        dir_consist = float(np.mean(dir_correct)) if dir_correct else None

        # AUC: higher base activation → predicted to go DOWN (more room)
        base_auc = None
        if len(base_acts) >= 5:
            try:
                raw = float(roc_auc_score(dir_labels, base_acts))
                base_auc = max(raw, 1.0 - raw)  # direction-invariant
            except Exception:
                pass

        per_axis[ax] = {
            "n_pairs":             len(sub),
            "best_feature":        best_feat,
            "mean_target_shift":   round(mean_shift, 3)  if mean_shift  is not None else None,
            "direction_consistency": round(dir_consist, 3) if dir_consist is not None else None,
            "base_activation_auc": round(base_auc, 3)    if base_auc    is not None else None,
        }

    return per_axis


# ── Driver ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    APP_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(APP_ROOT))

    from config.axes import ALL_AXIS_NAMES, AXES
    from config.settings import (
        SAE2_DATASET_FILE, SAE2_LABELS_FILE, SAE2_VARIANTS_DIR,
        SAE2_PAIRS_DIR, SAE2_CONFIRM, SAE2_PARTIAL, SAE2_DEAD_DENSITY,
    )

    VARIANT = "qwen22_knn"
    variant_dir = APP_ROOT / SAE2_VARIANTS_DIR / VARIANT

    print("Loading data…")
    dataset     = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labels      = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)
    act_full    = np.load(variant_dir / "feature_activations.npy")   # (9500, 128)
    correlations = json.loads((variant_dir / "correlations.json").read_text())

    # Align activations to labeled rows
    pid_to_idx  = {str(p): i for i, p in enumerate(dataset["post_id"].astype(str))}
    row_indices = np.array([pid_to_idx[str(p)] for p in labels["post_id"].astype(str)])
    act_labeled = act_full[row_indices]          # (1997, 128)
    print(f"  activations aligned: {act_labeled.shape}")

    # Load purity cache
    cache_file  = APP_ROOT / SAE2_PAIRS_DIR / "pair_cache.jsonl"
    purity_cache: dict = {}
    with open(cache_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec["key"].startswith("purity:"):
                    purity_cache[rec["key"]] = rec["value"]
    print(f"  purity cache entries: {len(purity_cache)}")

    # Load pairs
    pairs_path = APP_ROOT / SAE2_PAIRS_DIR / "synthetic_pairs.parquet"
    pairs_df   = pd.read_parquet(pairs_path)
    kept_n     = pairs_df["kept"].sum()
    print(f"  synthetic pairs: {len(pairs_df)} total, {kept_n} kept\n")

    # ── 1. Coverage ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("1. COVERAGE")
    print("=" * 60)
    cov = coverage(act_labeled, labels, ALL_AXIS_NAMES,
                   confirm=SAE2_CONFIRM, partial=SAE2_PARTIAL, axes_meta=AXES)

    print(f"\n  {'Axis':20s}  {'Status':12s}  {'Score':>6}  {'r':>6}  {'Lift':>6}  {'AUC':>6}  {'Meas.':6s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for ax in ALL_AXIS_NAMES:
        r = cov["per_axis"][ax]
        auc_str = f"{r['label_auc']:.3f}" if r["label_auc"] else "  n/a"
        flag = "✓" if r["status"] == "confirmed" else ("~" if r["status"] == "partial" else "✗")
        print(f"  {flag} {ax:19s}  {r['status']:12s}  {r['score']:>6.3f}  {r['best_r']:>6.3f}  {r['best_lift']:>6.3f}  {auc_str:>6}  {r['measurability']}")

    print(f"\n  Confirmed: {cov['confirmed']}/{cov['total']}")
    humor = cov["per_axis"].get("humor", {})
    print(f"  Humor verdict (bellwether): {humor.get('status','?')} "
          f"(score={humor.get('score','?')}, measurability=low)")

    # ── 2. Pair separation ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. PAIR SEPARATION")
    print("=" * 60)
    ps = pair_separation(
        pairs_df, correlations, act_full, dataset, purity_cache, ALL_AXIS_NAMES
    )

    print(f"\n  {'Axis':20s}  {'N':>4}  {'Feat':>4}  {'Shift':>7}  {'Dir%':>6}  {'BaseAUC':>8}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*8}")
    for ax in ALL_AXIS_NAMES:
        r = ps[ax]
        if r.get("n_pairs", 0) == 0:
            print(f"  {'':2}{ax:19s}  {'0':>4}  {'—':>4}  {'—':>7}  {'—':>6}  {'—':>8}")
            continue
        shift_str = f"{r['mean_target_shift']:+.3f}" if r["mean_target_shift"] is not None else "  n/a"
        dir_str   = f"{100*r['direction_consistency']:.0f}%" if r["direction_consistency"] is not None else "  n/a"
        auc_str   = f"{r['base_activation_auc']:.3f}"       if r["base_activation_auc"]  is not None else "   n/a"
        print(f"  {'':2}{ax:19s}  {r['n_pairs']:>4}  {r['best_feature']:>4}  {shift_str:>7}  {dir_str:>6}  {auc_str:>8}")

    print("\n  Shift: mean LLM-scored axis change (+ = moved up, - = moved down)")
    print("  Dir%:  % of pairs where shift matched intended direction")
    print("  BaseAUC: ROC-AUC of best-feature activation on base post predicting rewrite direction")
    print("\n  NOTE: SAE activation gap on shifted texts (gold-standard) requires")
    print("  encoding shifted_text through Qwen2.5-7B → rerun with --activation-gap")

    # ── 3. Object-type guard smoke test ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. OBJECT-TYPE GUARD")
    print("=" * 60)
    meta = json.loads((variant_dir / "meta.json").read_text())
    try:
        assert_object_type(meta, "single_post")
        print(f"  ✓ assert_object_type('single_post') passed  (trained on '{meta['object_type']}')")
    except ValueError as e:
        print(f"  ✗ {e}")
    try:
        assert_object_type(meta, "pair_difference")
        print("  ✗ should have raised for pair_difference")
    except ValueError:
        print("  ✓ assert_object_type('pair_difference') correctly rejected")

# TODO: gold-standard pair separation
# When shifted texts are encoded through Qwen2.5-7B:
#   1. encode shifted_text column → qwen hidden states → kNN residual → SAE activation
#   2. activation_gap = act_shifted[:, best_feat] - act_base[:, best_feat]
#   3. AUC: does gap > 0 predict direction == "up"?
#   Add --activation-gap flag to this driver to enable.
