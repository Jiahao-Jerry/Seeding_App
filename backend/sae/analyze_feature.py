"""
Analyze SAE features using Claude API.

For each feature, sends the correlation metrics + top-20 activating posts to
Claude and gets back a 3-sentence analysis of what the feature represents.

Usage:
    python -m backend.sae.analyze_feature                    # all confirmed features
    python -m backend.sae.analyze_feature 6 7 30 42 120     # specific feature IDs
    python -m backend.sae.analyze_feature --variant qwen24_knn_k25_l0004
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_ROOT))
load_dotenv(APP_ROOT / ".env")

from config.settings import SAE2_VARIANTS_DIR

VARIANTS_DIR = APP_ROOT / SAE2_VARIANTS_DIR
DEFAULT_VARIANT = "qwen24_knn_k25_l0004"
MODEL = "claude-opus-4-8"
TOP_K = 20

ALL_AXIS_NAMES = [
    "narrativity", "casualness", "concreteness", "humor",
    "warmth", "self_disclosure", "hedging", "tone", "reading_level",
]

AXIS_DEFINITIONS = """
1. reading_level — How advanced the vocabulary and sentence structure is. Folds in how much context the author assumes.
2. concreteness — Abstract/theoretical claims vs grounded in specifics — examples, numbers, named cases, analogies.
3. narrativity — How story-like the delivery is — characters, sequence, scene, or arc.
4. hedging — How certain the delivery sounds. About epistemic stance, NOT about whether the claim is true.
5. tone — INTENSITY of feeling — how heated vs measured. NOT the opinion/stance itself.
6. warmth — DIRECTION of affect toward the reader/subject — distinct from tone (intensity).
7. self_disclosure — How much the author opens up about their OWN self/experience/feelings.
8. casualness — Polished formal register vs casual internet register (slang, lowercase, fragments, emoji).
9. humor — Presence and intensity of comedy, wit, or playful tone.
""".strip()


def build_prompt(feat: dict, top_posts: list[dict]) -> str:
    r = feat.get("correlations", {})
    lifts = feat.get("lifts", {})

    # Format correlations and lifts
    corr_lines = "\n".join(
        f"  {ax:<20} {r.get(ax, 0):+.3f}" for ax in ALL_AXIS_NAMES
    )
    lift_lines = "\n".join(
        f"  {ax:<20} {lifts.get(ax, 0):+.3f}" for ax in ALL_AXIS_NAMES
    )

    # Axes confirmed at >= 0.2 (both directions)
    confirmed = [
        ax for ax in ALL_AXIS_NAMES
        if max(abs(r.get(ax, 0)), abs(lifts.get(ax, 0))) >= 0.2
    ]
    confirmed_str = ", ".join(
        f"{ax}({'r='+str(round(r.get(ax,0),3)) if abs(r.get(ax,0)) >= abs(lifts.get(ax,0)) else 'lift='+str(round(lifts.get(ax,0),3))})"
        for ax in confirmed
    ) or "none"

    # Format posts
    posts_str = "\n".join(
        f"{i+1}. act={p['activation']:.4f} | {p['text']}"
        for i, p in enumerate(top_posts)
    )

    return f"""You are analyzing SAE Feature {feat['feature']} (F{feat['feature']}) from a Sparse Autoencoder trained on Bluesky social media post embeddings. Figure out what latent concept this feature is detecting using both metrics and top {TOP_K} posts.

## Axis definitions (scale 0.0–1.0 each)
{AXIS_DEFINITIONS}

## F{feat['feature']} statistical metrics
- category: {feat.get('category', '?')}
- best_axis: {feat.get('best_axis', '?')} ({'NEGATIVE' if r.get(feat.get('best_axis',''), 0) < 0 else 'positive'})
- confirmed axes (score >= 0.2): {confirmed_str}
- density: {feat.get('density', 0):.4f}

Pearson r (correlation with axis label scores):
{corr_lines}

Lift (mean axis score when feature fires vs. when silent):
{lift_lines}

## Top {TOP_K} posts by F{feat['feature']} activation score
{posts_str}

## Your task
Analyze what concept F{feat['feature']} is detecting. Use BOTH the metrics and the posts — the metrics tell you the statistical shape, the posts tell you what it looks like in practice. If the best_axis correlation is NEGATIVE, the feature detects the LOW end of that axis, not the high end. If multiple axes are confirmed, explain what unifies them.

Respond in exactly 3 sentences. Be precise — propose a sharper label than just the axis name if the posts support it."""


def get_top_posts(acts: np.ndarray, dataset: pd.DataFrame, fid: int) -> list[dict]:
    scores = acts[:, fid]
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    posts = []
    for idx in top_idx:
        if scores[idx] <= 0:
            break
        row = dataset.iloc[int(idx)]
        posts.append({
            "activation": round(float(scores[idx]), 4),
            "text": str(row["text"]).replace("\n", " "),
        })
    return posts


def analyze_feature(client: anthropic.Anthropic, feat: dict, top_posts: list[dict]) -> str:
    prompt = build_prompt(feat, top_posts)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def print_result(feat: dict, analysis: str):
    fid = feat["feature"]
    r = feat.get("correlations", {})
    lifts = feat.get("lifts", {})
    confirmed = [
        ax for ax in ALL_AXIS_NAMES
        if max(abs(r.get(ax, 0)), abs(lifts.get(ax, 0))) >= 0.2
    ]
    axes_str = ", ".join(
        f"{ax}({'+' if r.get(ax,0)>=0 else ''}{r.get(ax,0):.3f})"
        for ax in confirmed
    )
    print(f"\n{'='*70}")
    print(f"F{fid} | {feat.get('category','?')} | density={feat.get('density',0):.3f}")
    print(f"Confirmed axes: {axes_str or 'none'}")
    print(f"{'-'*70}")
    print(analysis)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", nargs="*", type=int, help="Feature IDs to analyze (default: all confirmed)")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--all", action="store_true", help="Analyze all features including partial/novel")
    args = parser.parse_args()

    vdir = VARIANTS_DIR / args.variant
    corrs = json.loads((vdir / "correlations.json").read_text())
    acts = np.load(vdir / "feature_activations.npy")
    dataset = pd.read_parquet(APP_ROOT / "data/sae2/dataset.parquet")

    # Select features to analyze
    if args.features:
        feat_ids = args.features
    elif args.all:
        feat_ids = [f["feature"] for f in corrs if f.get("category") != "dead"]
    else:
        # Default: confirmed features only (score >= 0.2)
        feat_ids = []
        for f in corrs:
            if f.get("category") == "dead":
                continue
            r = f.get("correlations", {})
            lifts = f.get("lifts", {})
            if any(max(abs(r.get(ax, 0)), abs(lifts.get(ax, 0))) >= 0.2 for ax in ALL_AXIS_NAMES):
                feat_ids.append(f["feature"])

    print(f"Variant: {args.variant}")
    print(f"Analyzing {len(feat_ids)} features: {feat_ids}")

    client = anthropic.Anthropic()

    for fid in feat_ids:
        feat = corrs[fid]
        top_posts = get_top_posts(acts, dataset, fid)
        if not top_posts:
            print(f"\nF{fid}: dead — skipping")
            continue
        analysis = analyze_feature(client, feat, top_posts)
        print_result(feat, analysis)


if __name__ == "__main__":
    main()
