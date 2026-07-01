"""
Independent evaluator for agent_hedging.jsonl pairs.
Applies 3 quality gates using EXACT prompts from source files.
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

# Load .env
env_file = APP_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from openai import AsyncOpenAI

# Import EXACT prompts from source files
from backend.transform import verify_system_prompt, verify_user_prompt
from backend.sae.pairs import _direction_system, _direction_user
from prompts import annotation_system, annotation_user
from config.axes import ALL_AXIS_NAMES, AXES

# hedging axis definition
HEDGING_DEF = next(a["definition"] for a in AXES if a["name"] == "hedging")

INPUT_FILE  = APP_ROOT / "data/sae2/pairs/agent_hedging.jsonl"
OUTPUT_FILE = APP_ROOT / "data/sae2/pairs/eval_hedging.jsonl"
PURITY_THRESHOLD = 0.15
CONCURRENCY = 3


async def llm_json(client, system: str, user: str, max_tokens: int = 512) -> dict:
    for attempt in range(8):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = min(30, 2 ** attempt)
                await asyncio.sleep(wait)
            elif attempt == 7:
                raise
            else:
                await asyncio.sleep(2 ** attempt)
    return {}


def _extract_scores(raw: dict) -> dict:
    out = {}
    for ax in ALL_AXIS_NAMES:
        v = raw.get(ax, 0.5)
        out[ax] = float(v["score"] if isinstance(v, dict) else v)
    return out


async def evaluate_pair(client, sem: asyncio.Semaphore, pair: dict, rng: random.Random) -> dict:
    pair_id      = pair["pair_id"]
    direction    = pair["direction"]
    base_text    = pair["base_text"]
    shifted_text = pair["shifted_text"]

    async with sem:
        # ── Gate 1: Substance ─────────────────────────────────────────────
        subst = await llm_json(
            client,
            verify_system_prompt(),
            verify_user_prompt(base_text, shifted_text),
            max_tokens=256,
        )
        passed_substance = bool(subst.get("substance_preserved", True))

        if not passed_substance:
            return {
                "pair_id":           pair_id,
                "axis":              "hedging",
                "direction":         direction,
                "base_post_id":      pair["base_post_id"],
                "base_text":         base_text,
                "shifted_text":      shifted_text,
                "passed_substance":  False,
                "passed_direction":  False,
                "passed_purity":     False,
                "target_shift":      None,
                "non_target_drifts": {},
                "kept":              False,
                "fail_reason":       "substance",
            }

        # ── Gate 2: Direction ─────────────────────────────────────────────
        a_is_shifted = rng.random() > 0.5
        text_a = shifted_text if a_is_shifted else base_text
        text_b = base_text   if a_is_shifted else shifted_text

        dir_result = await llm_json(
            client,
            _direction_system(),
            _direction_user("hedging", HEDGING_DEF, text_a, text_b),
            max_tokens=64,
        )
        winner = dir_result.get("winner", "")

        # "up" direction means shifted is more hedged → should score higher
        # "down" direction means shifted is more direct → should score lower
        if direction == "up":
            # shifted should be "more hedging" → winner should be whichever slot shifted is in
            expected = "A" if a_is_shifted else "B"
        else:
            # shifted should be "less hedging" → winner should be whichever slot base is in
            expected = "B" if a_is_shifted else "A"

        passed_direction = (winner == expected)

        # ── Gate 3: Purity — annotate both posts on all 9 axes ───────────
        base_raw, shift_raw = await asyncio.gather(
            llm_json(client, annotation_system(), annotation_user(base_text),    max_tokens=512),
            llm_json(client, annotation_system(), annotation_user(shifted_text), max_tokens=512),
        )
        base_scores  = _extract_scores(base_raw)
        shift_scores = _extract_scores(shift_raw)

        non_target_drifts = {
            ax: abs(shift_scores[ax] - base_scores[ax])
            for ax in ALL_AXIS_NAMES if ax != "hedging"
        }
        passed_purity = all(d < PURITY_THRESHOLD for d in non_target_drifts.values())
        target_shift  = shift_scores["hedging"] - base_scores["hedging"]

        kept = passed_substance and passed_direction and passed_purity
        if not kept:
            if not passed_direction:
                fail_reason = "direction"
            elif not passed_purity:
                fail_reason = "purity"
            else:
                fail_reason = "substance"
        else:
            fail_reason = None

        return {
            "pair_id":           pair_id,
            "axis":              "hedging",
            "direction":         direction,
            "base_post_id":      pair["base_post_id"],
            "base_text":         base_text,
            "shifted_text":      shifted_text,
            "passed_substance":  passed_substance,
            "passed_direction":  passed_direction,
            "passed_purity":     passed_purity,
            "target_shift":      round(target_shift, 3),
            "non_target_drifts": {k: round(v, 3) for k, v in non_target_drifts.items()},
            "kept":              kept,
            "fail_reason":       fail_reason,
        }


async def main():
    api_key = os.environ.get("OPENAI_API_KEY_UMICH_DYIMOD")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY_UMICH_DYIMOD not set")

    client = AsyncOpenAI(api_key=api_key, max_retries=6)
    sem    = asyncio.Semaphore(CONCURRENCY)
    rng    = random.Random(12345)  # fixed seed for reproducibility

    pairs = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    print(f"Evaluating {len(pairs)} hedging pairs through 3 gates...")

    # Load already-evaluated pairs (checkpoint/resume)
    already_done = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    already_done[rec["pair_id"]] = rec
        print(f"  Resuming: {len(already_done)} pairs already evaluated, skipping them.")

    pending = [p for p in pairs if p["pair_id"] not in already_done]
    print(f"  {len(pending)} pairs to evaluate now.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = list(already_done.values())
    coros = [evaluate_pair(client, sem, p, rng) for p in pending]
    done = 0
    with open(OUTPUT_FILE, "a") as out_f:
        for coro in asyncio.as_completed(coros):
            result = await coro
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            done += 1
            if done % 5 == 0 or done == len(pending):
                print(f"  {done}/{len(pending)} new pairs done (total {len(results)})")

    # Re-read all and re-sort for final clean output
    results.sort(key=lambda r: r["pair_id"])
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    total          = len(results)
    n_substance    = sum(1 for r in results if r["passed_substance"])
    n_direction    = sum(1 for r in results if r["passed_direction"])
    n_purity       = sum(1 for r in results if r["passed_purity"])
    n_kept         = sum(1 for r in results if r["kept"])

    fail_substance = sum(1 for r in results if r["fail_reason"] == "substance")
    fail_direction = sum(1 for r in results if r["fail_reason"] == "direction")
    fail_purity    = sum(1 for r in results if r["fail_reason"] == "purity")

    print(f"\n{'='*55}")
    print(f"EVALUATION SUMMARY — hedging ({total} pairs)")
    print(f"{'='*55}")
    print(f"  Gate 1 substance:  {n_substance}/{total} passed ({100*n_substance/total:.1f}%)")
    print(f"  Gate 2 direction:  {n_direction}/{total} passed ({100*n_direction/total:.1f}%)")
    print(f"  Gate 3 purity:     {n_purity}/{total} passed ({100*n_purity/total:.1f}%)")
    print(f"  KEPT (all 3):      {n_kept}/{total} ({100*n_kept/total:.1f}%)")
    print(f"\nFail breakdown:")
    print(f"  substance: {fail_substance}")
    print(f"  direction: {fail_direction}")
    print(f"  purity:    {fail_purity}")

    if n_kept > 0:
        kept_results = [r for r in results if r["kept"]]
        avg_shift    = sum(r["target_shift"] for r in kept_results) / len(kept_results)
        up_kept   = [r for r in kept_results if r["direction"] == "up"]
        down_kept = [r for r in kept_results if r["direction"] == "down"]
        print(f"\nKept pairs:")
        print(f"  avg target_shift:  {avg_shift:+.3f}")
        print(f"  direction=up:      {len(up_kept)}")
        print(f"  direction=down:    {len(down_kept)}")

        # non-target drift stats for kept pairs
        drift_sums = {ax: 0.0 for ax in ALL_AXIS_NAMES if ax != "hedging"}
        for r in kept_results:
            for ax, v in r["non_target_drifts"].items():
                drift_sums[ax] += v
        print(f"\n  Avg non-target drift (kept pairs):")
        for ax in ALL_AXIS_NAMES:
            if ax == "hedging":
                continue
            avg = drift_sums[ax] / len(kept_results)
            print(f"    {ax:20s}: {avg:.3f}")

    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
