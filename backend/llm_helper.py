"""
LLM helper: single interface for all LLM calls.
Uses OPENAI_API_KEY_UMICH_DYIMOD from .env.
"""

import json
import os
from pathlib import Path
from openai import AsyncOpenAI, AuthenticationError

_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY_UMICH_DYIMOD")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY_UMICH_DYIMOD not set in environment")
        _client = AsyncOpenAI(api_key=api_key, timeout=30.0)
    return _client


async def llm_json(system: str, user: str, model: str = "gpt-5.4-mini",
                   temperature: float | None = None) -> dict:
    """
    Call LLM and return parsed JSON dict.
    Retries up to 3 times on parse failure.

    temperature: explicit override for callers that need it (e.g. a
    regeneration retry loop that wants a genuinely different rewrite each
    time, not just re-asking an already near-deterministic prompt). Leave as
    None to keep the original behavior (0.0 on first attempt, 0.1 on
    parse-failure retries) unchanged for every existing caller.
    """
    client = get_client()

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature if temperature is not None else (0.1 if attempt > 0 else 0.0),
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except AuthenticationError:
            # Not retryable -- an invalid/expired key will fail identically
            # on every attempt, so retrying just wastes time and buries the
            # real cause inside a generic "JSON parse failed" message.
            # Surface it immediately and distinctly instead.
            raise
        except (json.JSONDecodeError, Exception) as e:
            if attempt == 2:
                raise ValueError(f"LLM JSON parse failed after 3 attempts: {e}")
            continue
