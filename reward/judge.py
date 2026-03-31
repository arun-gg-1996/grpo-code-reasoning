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
import re
from typing import Optional

import httpx

from config import JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS, JUDGE_TIMEOUT, GEMINI_API_KEY

logger = logging.getLogger(__name__)

VERTEX_URL = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{JUDGE_MODEL}:generateContent"

_last_batch_stats = {
    "total_calls": 0,
    "fallback_count": 0,
    "fallback_fraction": 0.0,
}


def _build_request_body(problem: str, think_block: str, difficulty: str) -> dict:
    user_prompt = (
        "Evaluate this reasoning trace for a competitive programming problem.\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\"step_scores\": [s1, s2, s3, s4, s5, s6], \"overall\": o}\n"
        "Rules:\n"
        "- step_scores must contain exactly 6 floats in [0.0, 1.0]\n"
        "- overall must be a float in [0.0, 1.0]\n"
        "- no markdown, no prose, no extra keys\n\n"
        f"Difficulty: {difficulty}\n\n"
        "Problem:\n"
        f"{problem}\n\n"
        "Reasoning trace:\n"
        f"{think_block}"
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
        try:
            data = json.loads(text)
        except Exception:
            # Try extracting the first JSON object if model added extra text.
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise
            data = json.loads(m.group(0))

        overall = max(0.0, min(1.0, float(data.get("overall", 0.5))))
        raw_steps = data.get("step_scores", [])
        step_scores = [max(0.0, min(1.0, float(s))) for s in raw_steps[:6]]
        if len(step_scores) < 6:
            step_scores.extend([overall] * (6 - len(step_scores)))
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
) -> tuple[float, bool, bool]:
    """Single async Gemini call. Returns (score, used_fallback, used_step_scores)."""
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
        if parsed is not None:
            steps = parsed.get("step_scores", [])
            if steps:
                # Judge score is derived from per-step quality.
                return float(sum(steps) / len(steps)), False, True
            return parsed["overall"], False, False
        return 0.5, True, False

    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        return 0.5, True, False


async def _score_batch_async(
    problems: list[str],
    think_blocks: list[str],
    difficulties: list[str],
    api_key: str,
) -> list[tuple[float, bool, bool]]:
    async with httpx.AsyncClient() as client:
        tasks = [
            _call_single(client, api_key, prob, think, diff)
            for prob, think, diff in zip(problems, think_blocks, difficulties)
        ]
        return await asyncio.gather(*tasks)


def get_last_batch_stats() -> dict:
    return dict(_last_batch_stats)


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
    global _last_batch_stats
    api_key = GEMINI_API_KEY
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — returning 0.5 fallback for all")
        n = len(problems)
        _last_batch_stats = {
            "total_calls": n,
            "fallback_count": n,
            "fallback_fraction": 1.0 if n > 0 else 0.0,
            "step_json_count": 0,
            "step_json_fraction": 0.0,
        }
        return [0.5] * len(problems)

    triples = asyncio.run(_score_batch_async(problems, think_blocks, difficulties, api_key))
    scores = [s for s, _, _ in triples]
    fallback_count = sum(1 for _, used_fallback, _ in triples if used_fallback)
    step_json_count = sum(1 for _, _, used_steps in triples if used_steps)
    total = len(triples)
    _last_batch_stats = {
        "total_calls": total,
        "fallback_count": fallback_count,
        "fallback_fraction": (fallback_count / total) if total > 0 else 0.0,
        "step_json_count": step_json_count,
        "step_json_fraction": (step_json_count / total) if total > 0 else 0.0,
    }
    return scores
