"""
reward/judge.py

Gemini judge via Vertex AI using API key auth.
Endpoint: https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key=API_KEY

Set GEMINI_API_KEY env var before training:
    export GEMINI_API_KEY=your_key_here

All batch calls are concurrent via asyncio.gather.
"""

import asyncio
import json
import logging
from typing import Optional

import httpx

from config import JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS, JUDGE_TIMEOUT, GEMINI_API_KEY

logger = logging.getLogger(__name__)

VERTEX_URL = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{JUDGE_MODEL}:generateContent"


def _build_request_body(problem: str, think_block: str, difficulty: str) -> dict:
    user_prompt = (
        f"Problem ({difficulty} difficulty):\n{problem}\n\n"
        f"Model reasoning trace:\n{think_block}\n\n"
        f"Score the 6 steps and provide an overall score."
    )
    return {
        "system_instruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": JUDGE_TEMPERATURE,
            "maxOutputTokens": JUDGE_MAX_TOKENS,
        },
    }


def parse_judge_response(text: str) -> Optional[dict]:
    """
    Parse Gemini judge response.
    Handles two formats:
      - Plain float: "0.85"
      - JSON: {"step_scores": [...], "overall": 0.85}
    Returns dict with "overall" key, or None on failure.
    """
    try:
        text = text.strip().strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

        # Try plain float first (current prompt format)
        try:
            overall = max(0.0, min(1.0, float(text)))
            return {"overall": overall, "step_scores": []}
        except ValueError:
            pass

        # Try JSON format
        data = json.loads(text)
        overall = max(0.0, min(1.0, float(data.get("overall", 0.5))))
        step_scores = [max(0.0, min(1.0, float(s))) for s in data.get("step_scores", [])[:6]]
        return {"overall": overall, "step_scores": step_scores}
    except Exception as e:
        logger.warning(f"Failed to parse judge response: {e} | text: {text[:200]}")
        return None


async def _call_single(
    client: httpx.AsyncClient,
    api_key: str,
    problem: str,
    think_block: str,
    difficulty: str,
) -> float:
    """Single async Gemini call. Returns overall score 0.0-1.0. Falls back to 0.5 on error."""
    body = _build_request_body(problem, think_block, difficulty)
    url = f"{VERTEX_URL}?key={api_key}"

    try:
        resp = await client.post(url, json=body, timeout=JUDGE_TIMEOUT)

        if resp.status_code == 429:
            await asyncio.sleep(2.0)
            resp = await client.post(url, json=body, timeout=JUDGE_TIMEOUT)

        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = parse_judge_response(text)
        return parsed["overall"] if parsed is not None else 0.5

    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        return 0.5


async def _score_batch_async(
    problems: list[str],
    think_blocks: list[str],
    difficulties: list[str],
    api_key: str,
) -> list[float]:
    async with httpx.AsyncClient() as client:
        tasks = [
            _call_single(client, api_key, prob, think, diff)
            for prob, think, diff in zip(problems, think_blocks, difficulties)
        ]
        return await asyncio.gather(*tasks)


def score_batch(
    problems: list[str],
    think_blocks: list[str],
    difficulties: list[str],
) -> list[float]:
    """
    Score reasoning traces via Gemini. All calls concurrent.
    Reads API key from GEMINI_API_KEY env var (set in config.py).
    Returns list of float scores 0.0-1.0 (0.5 on error).
    """
    api_key = GEMINI_API_KEY
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — returning 0.5 fallback for all")
        return [0.5] * len(problems)

    return asyncio.run(_score_batch_async(problems, think_blocks, difficulties, api_key))
