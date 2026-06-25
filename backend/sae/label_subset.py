"""
SAE2 label subset — LLM axis scores for the validation reference set.

Two functions:
  stratified_subset(df, n, seed)  sample n posts stratified by topic × substance
  label_axes(df)                  score each post on all 9 axes with Claude Haiku

Scores are a reference (not fully trusted); they are NOT training data.
Every result is cached to axis_label_cache.jsonl (keyed by post_id, append-only)
so reruns skip already-scored posts and never re-pay.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2]

_CACHE_FILE = APP_ROOT / "data/sae2/axis_label_cache.jsonl"

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.axes import ALL_AXIS_NAMES as _AXES
from prompts import annotation_system as _annotation_system, annotation_user as _annotation_user


# ── Stratified sampling ────────────────────────────────────────────────────────

def stratified_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample n posts from df stratified by (topic_name, substance).
    Proportional allocation; groups smaller than their share contribute all they have.
    """
    rng = np.random.default_rng(seed)
    total = len(df)
    parts = []
    for _, grp in df.groupby(["topic_name", "substance"]):
        alloc = max(1, round(n * len(grp) / total))
        take = min(alloc, len(grp))
        idx = rng.choice(len(grp), size=take, replace=False)
        parts.append(grp.iloc[idx])
    subset = pd.concat(parts).sample(frac=1, random_state=int(seed)).reset_index(drop=True)
    return subset.head(n)


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, dict]:
    if not _CACHE_FILE.exists():
        return {}
    cache: dict[str, dict] = {}
    with open(_CACHE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[str(rec["post_id"])] = rec["scores"]
    return cache


def _append_cache(post_id: str, scores: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "a") as f:
        f.write(json.dumps({"post_id": post_id, "scores": scores}) + "\n")


# ── Async scoring ──────────────────────────────────────────────────────────────

async def _score_one(
    client,
    post_id: str,
    text: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict | None]:
    import anthropic as _ant
    async with semaphore:
        for attempt in range(5):
            try:
                resp = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=512,
                    system=_annotation_system(),
                    messages=[{"role": "user", "content": _annotation_user(text)}],
                )
                raw = resp.content[0].text.strip()
                if not raw:
                    raise ValueError("empty response from model")
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw.strip())
                # official prompt returns {"score": float} or {"score": float, "subtype": ...}
                # extract just the score for each axis
                scores = {}
                for ax in _AXES:
                    val = parsed.get(ax, 0.5)
                    scores[ax] = float(val["score"] if isinstance(val, dict) else val)
                return post_id, scores
            except (_ant.RateLimitError, _ant.APIStatusError) as e:
                wait = 10 * (2 ** attempt)
                print(f"  [rate limit] attempt {attempt+1}, waiting {wait}s…")
                await asyncio.sleep(wait)
            except json.JSONDecodeError as e:
                if attempt == 4:
                    print(f"  [warn] bad JSON for post {post_id}: {e}")
                    return post_id, None
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 4:
                    print(f"  [warn] failed post {post_id}: {e}")
                    return post_id, None
                await asyncio.sleep(2 ** attempt)
    return post_id, None


async def _run_all(
    posts: list[tuple[str, str]],
    concurrency: int = 20,
) -> dict[str, dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — add it to .env or export it.")

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    tasks = [_score_one(client, pid, text, sem) for pid, text in posts]

    results: dict[str, dict] = {}
    done = 0
    t0 = time.time()
    for coro in asyncio.as_completed(tasks):
        pid, scores = await coro
        if scores is not None:
            results[pid] = scores
            _append_cache(pid, scores)
        done += 1
        if done % 100 == 0 or done == len(tasks):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = max(len(tasks) - done, 0) / max(rate, 1e-6)
            print(f"  {done}/{len(tasks)} scored | {rate:.1f} posts/s | eta {eta:.0f}s")
    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def label_axes(df: pd.DataFrame, concurrency: int = 20) -> pd.DataFrame:
    """
    Score every post in df on all 9 axes. Returns df with 9 new float columns.
    Already-cached posts are never re-scored.
    """
    cache = _load_cache()
    todo = [
        (str(r.post_id), str(r.text))
        for r in df.itertuples()
        if str(r.post_id) not in cache
    ]
    print(f"  {len(cache)} already cached, {len(todo)} to score.")

    if todo:
        new = asyncio.run(_run_all(todo, concurrency=concurrency))
        cache.update(new)

    rows = [cache.get(str(pid), {ax: float("nan") for ax in _AXES})
            for pid in df["post_id"].astype(str)]

    labeled = df.copy().reset_index(drop=True)
    for ax in _AXES:
        labeled[ax] = [r[ax] for r in rows]
    return labeled


# ── Driver ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(APP_ROOT))

    # Load .env
    env_file = APP_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    from backend.sae.dataset import load_dataset
    from config.settings import SAE2_LABEL_SUBSET_N, SAE2_LABELS_FILE
    from config.axes import ALL_AXIS_NAMES

    df, _ = load_dataset()
    print(f"Corpus: {len(df)} posts. Sampling {SAE2_LABEL_SUBSET_N} stratified…")
    subset = stratified_subset(df, n=SAE2_LABEL_SUBSET_N)
    print(f"Subset: {len(subset)} posts, {subset['topic_name'].nunique()} topics, "
          f"substance: {subset['substance'].value_counts().to_dict()}")

    print("Scoring with Claude Haiku 4.5…")
    labeled = label_axes(subset, concurrency=20)

    out = APP_ROOT / SAE2_LABELS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out)
    print(f"\nSaved {len(labeled)} labeled posts → {out}")

    print("\nAxis score means (± std):")
    for ax in ALL_AXIS_NAMES:
        col = labeled[ax].dropna()
        print(f"  {ax:20s}: {col.mean():.3f} ± {col.std():.3f}")
