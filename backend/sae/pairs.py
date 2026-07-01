"""
SAE2 synthetic pairs — controlled single-axis probe set for feature validation.

For each of the 9 axes, rewrites a sample of base posts shifting ONLY that axis,
then runs three quality gates:
  1. Substance gate  — reuse transform.verify_system/user_prompt: did the
                       core claim/facts survive?
  2. Direction gate  — new: blind forced choice, which post is "more <axis>"?
  3. Purity gate     — reuse prompts.annotation_system/user: score both posts
                       on all 9 axes; drop if a non-target axis moved ≥ 0.15.

Prompts reused from RecMod:
  - transform_system_prompt() / transform_user_prompt()  → rewriter
  - verify_system_prompt() / verify_user_prompt()        → substance gate
  - annotation_system() / annotation_user()              → purity gate

Run: python backend/sae/pairs.py
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

import sys
sys.path.insert(0, str(APP_ROOT))

from config.axes import ALL_AXIS_NAMES, AXES
from config.settings import (
    SAE2_DATASET_FILE, SAE2_LABELS_FILE, SAE2_PAIRS_DIR,
    SAE2_SYNTH_PAIRS_PER_AXIS,
)
from backend.transform import (
    transform_system_prompt, transform_user_prompt,
    verify_system_prompt, verify_user_prompt,
)
from prompts import annotation_system, annotation_user

CACHE_FILE  = APP_ROOT / SAE2_PAIRS_DIR / "pair_cache.jsonl"
OUTPUT_FILE = APP_ROOT / SAE2_PAIRS_DIR / "synthetic_pairs.parquet"
PURITY_THRESHOLD = 0.15   # max allowed shift on non-target axes
CONCURRENCY      = 6      # ~48 pairs/min, saturates 200K TPM ceiling


# ── Direction gate prompts ─────────────────────────────────────────────────────

def _direction_system() -> str:
    return (
        "You compare two social media posts and judge which one better exhibits "
        "a given writing style dimension. Reply with valid JSON only — no markdown."
    )


def _direction_user(axis_name: str, axis_def: str, text_a: str, text_b: str) -> str:
    return (
        f"Axis: {axis_name.replace('_', ' ')}\n"
        f"Definition: {axis_def}\n\n"
        f"POST A:\n\"\"\"{text_a}\"\"\"\n\n"
        f"POST B:\n\"\"\"{text_b}\"\"\"\n\n"
        f"Which post scores HIGHER on {axis_name.replace('_', ' ')}? "
        f"Reply JSON: {{\"winner\": \"A\" or \"B\", \"confidence\": 0.0-1.0}}"
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    cache = {}
    with open(CACHE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cache[rec["key"]] = rec["value"]
    return cache


def _save_cache(key: str, value: dict, cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache[key] = value
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


# ── Single pair pipeline ───────────────────────────────────────────────────────

async def _llm_json(client, system: str, user: str, max_tokens: int = 512) -> dict:
    for attempt in range(4):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, Exception) as e:
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt)
    return {}


async def _process_pair(
    client,
    semaphore: asyncio.Semaphore,
    pair_id: str,
    axis: dict,
    base_text: str,
    base_post_id: str,
    direction: str,
    cache: dict,
) -> dict:
    axis_name = axis["name"]
    axis_def  = axis["definition"]

    # Build deltas in transform.py format
    if direction == "up":
        current, target = 0.2, 0.85
    else:
        current, target = 0.8, 0.15

    deltas = {
        axis_name: {
            "current":     current,
            "target":      target,
            "direction":   "increase" if direction == "up" else "decrease",
            "gap":         round(target - current, 2),
            "is_additive": axis.get("additive", False),
        }
    }

    async with semaphore:
        # ── Step 1: Rewrite ────────────────────────────────────────────────
        rewrite_key = f"rewrite:{pair_id}"
        if rewrite_key in cache:
            rewrite_result = cache[rewrite_key]
        else:
            rewrite_result = await _llm_json(
                client,
                transform_system_prompt(),
                transform_user_prompt(base_text, deltas),
                max_tokens=600,
            )
            _save_cache(rewrite_key, rewrite_result, cache)

        shifted_text = rewrite_result.get("rewritten_text", "")
        if not shifted_text or shifted_text == base_text:
            return {"pair_id": pair_id, "kept": False, "fail_reason": "empty_rewrite"}

        # ── Step 2: Substance gate ─────────────────────────────────────────
        subst_key = f"substance:{pair_id}"
        if subst_key in cache:
            subst = cache[subst_key]
        else:
            subst = await _llm_json(
                client,
                verify_system_prompt(),
                verify_user_prompt(base_text, shifted_text),
                max_tokens=256,
            )
            _save_cache(subst_key, subst, cache)

        if not subst.get("substance_preserved", True):
            return {"pair_id": pair_id, "kept": False, "fail_reason": "substance_violation"}

        # ── Step 3: Direction gate ─────────────────────────────────────────
        dir_key = f"direction:{pair_id}"
        if dir_key in cache:
            dir_result = cache[dir_key]
        else:
            rng = np.random.default_rng(abs(hash(pair_id)) % (2**32))
            a_is_shifted = bool(rng.integers(0, 2))
            text_a = shifted_text if a_is_shifted else base_text
            text_b = base_text   if a_is_shifted else shifted_text

            dir_result = await _llm_json(
                client,
                _direction_system(),
                _direction_user(axis_name, axis_def, text_a, text_b),
                max_tokens=64,
            )
            dir_result["a_is_shifted"] = a_is_shifted
            _save_cache(dir_key, dir_result, cache)

        winner      = dir_result.get("winner", "")
        a_is_shifted = dir_result.get("a_is_shifted", True)
        expected    = "A" if a_is_shifted else "B"
        if direction == "down":
            expected = "B" if a_is_shifted else "A"
        passed_direction = (winner == expected)

        # ── Step 4: Purity gate — score both posts on all 9 axes ──────────
        purity_key = f"purity:{pair_id}"
        if purity_key in cache:
            purity = cache[purity_key]
        else:
            base_scores_raw, shift_scores_raw = await asyncio.gather(
                _llm_json(client, annotation_system(), annotation_user(base_text),   max_tokens=512),
                _llm_json(client, annotation_system(), annotation_user(shifted_text), max_tokens=512),
            )
            def _extract(raw):
                out = {}
                for ax in ALL_AXIS_NAMES:
                    v = raw.get(ax, 0.5)
                    out[ax] = float(v["score"] if isinstance(v, dict) else v)
                return out
            purity = {
                "base_scores":  _extract(base_scores_raw),
                "shift_scores": _extract(shift_scores_raw),
            }
            _save_cache(purity_key, purity, cache)

        base_scores  = purity["base_scores"]
        shift_scores = purity["shift_scores"]
        non_target_drifts = {
            ax: abs(shift_scores[ax] - base_scores[ax])
            for ax in ALL_AXIS_NAMES if ax != axis_name
        }
        passed_purity = all(d < PURITY_THRESHOLD for d in non_target_drifts.values())
        target_shift  = shift_scores[axis_name] - base_scores[axis_name]

        return {
            "pair_id":           pair_id,
            "axis":              axis_name,
            "direction":         direction,
            "base_post_id":      base_post_id,
            "base_text":         base_text,
            "shifted_text":      shifted_text,
            "passed_substance":  True,
            "passed_direction":  passed_direction,
            "passed_purity":     passed_purity,
            "target_shift":      round(target_shift, 3),
            "non_target_drifts": {k: round(v, 3) for k, v in non_target_drifts.items()},
            "kept":              passed_direction and passed_purity,
            "fail_reason":       None if (passed_direction and passed_purity) else
                                 ("direction" if not passed_direction else "purity"),
        }


# ── Public API ─────────────────────────────────────────────────────────────────

async def _run_all(tasks_args: list, concurrency: int) -> list[dict]:
    api_key = os.environ.get("OPENAI_API_KEY_UMICH_DYIMOD")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY_UMICH_DYIMOD not set — add it to .env")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, max_retries=6)
    sem    = asyncio.Semaphore(concurrency)
    cache  = _load_cache()

    coros = [_process_pair(client, sem, cache=cache, **kw) for kw in tasks_args]
    results, done, t0 = [], 0, time.time()
    for coro in asyncio.as_completed(coros):
        result = await coro
        results.append(result)
        done += 1
        if done % 20 == 0 or done == len(coros):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            print(f"  {done}/{len(coros)} pairs processed | {rate:.1f}/s")
    return results


def synthetic_pairs(
    df: pd.DataFrame,
    per_axis: int | None = None,
    seed: int = 42,
    concurrency: int = CONCURRENCY,
) -> pd.DataFrame:
    """
    Generate and quality-gate single-axis synthetic pairs.
    df must have post_id and text columns. Uses the labeled subset if available.
    """
    if per_axis is None:
        per_axis = SAE2_SYNTH_PAIRS_PER_AXIS

    rng = np.random.default_rng(seed)
    axes_meta = {a["name"]: a for a in AXES}

    tasks = []
    for ax_name in ALL_AXIS_NAMES:
        ax = axes_meta[ax_name]
        pool = df.sample(frac=1, random_state=int(rng.integers(0, 2**31)))
        selected = pool.head(per_axis)
        for _, row in selected.iterrows():
            pid  = str(row["post_id"])
            text = str(row["text"])
            # alternate up/down
            direction = "up" if rng.random() > 0.5 else "down"
            pair_id   = f"{ax_name}_{pid}_{direction}"
            tasks.append({
                "pair_id":      pair_id,
                "axis":         ax,
                "base_text":    text,
                "base_post_id": pid,
                "direction":    direction,
            })

    print(f"Processing {len(tasks)} pairs across {len(ALL_AXIS_NAMES)} axes…")
    results = asyncio.run(_run_all(tasks, concurrency=concurrency))

    df_out = pd.DataFrame([r for r in results if "axis" in r])
    return df_out


# ── Driver ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env_file = APP_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # Use labeled subset (we have axis scores for those posts)
    labels  = pd.read_parquet(APP_ROOT / SAE2_LABELS_FILE)
    dataset = pd.read_parquet(APP_ROOT / SAE2_DATASET_FILE)
    labeled_ids = set(labels["post_id"].astype(str))
    pool = dataset[dataset["post_id"].astype(str).isin(labeled_ids)].copy()
    print(f"Base pool: {len(pool)} labeled posts")

    per_axis = int(sys.argv[1]) if len(sys.argv) > 1 else SAE2_SYNTH_PAIRS_PER_AXIS
    print(f"Generating {per_axis} pairs/axis × {len(ALL_AXIS_NAMES)} axes = {per_axis * len(ALL_AXIS_NAMES)} total\n")

    pairs_df = synthetic_pairs(pool, per_axis=per_axis)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_parquet(OUTPUT_FILE, index=False)

    kept    = pairs_df["kept"].sum()
    total   = len(pairs_df)
    print(f"\nResults: {kept}/{total} pairs passed all gates ({100*kept/max(total,1):.1f}%)")

    print("\nPer-axis summary:")
    print(f"  {'axis':20s}  {'total':>6}  {'kept':>6}  {'pass%':>6}  {'dir%':>6}  {'pur%':>6}")
    for ax in ALL_AXIS_NAMES:
        sub = pairs_df[pairs_df["axis"] == ax]
        if len(sub) == 0:
            continue
        k  = sub["kept"].sum()
        d  = sub["passed_direction"].sum()
        p  = sub["passed_purity"].sum()
        print(f"  {ax:20s}  {len(sub):>6}  {k:>6}  {100*k/len(sub):>5.1f}%  "
              f"{100*d/len(sub):>5.1f}%  {100*p/len(sub):>5.1f}%")

    print(f"\nSaved → {OUTPUT_FILE}")
