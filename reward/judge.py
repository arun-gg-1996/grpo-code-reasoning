"""
reward/judge.py

vLLM HTTP client for the judge model (Qwen2.5-Coder-7B-Instruct).
All requests are sent concurrently via asyncio + AsyncOpenAI so vLLM's
internal scheduler can batch them — no sequential blocking.

The judge server is started separately before training:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen2.5-Coder-7B-Instruct \
        --port 8000 \
        --enable-prefix-caching

Usage:
    from reward.judge import JudgeClient
    judge = JudgeClient(base_url="http://localhost:8000/v1")
    results = judge.score_reasoning_batch(prompts)  # list of dicts
"""

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from config import REASONING_SYSTEM_PROMPT, JUDGE_TIMEOUT, JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────

def build_reasoning_prompt(
        problem: str,
        reference_reasoning: str,
        model_reasoning: str,
) -> str:
    return f"""Problem:
{problem}

Reference reasoning:
{reference_reasoning}

Model's reasoning:
{model_reasoning}"""


# ─────────────────────────────────────────
# JSON parsing with fallbacks
# ─────────────────────────────────────────

def parse_reasoning_response(text: str) -> Optional[dict]:
    """
    Parse judge JSON response for reasoning scoring.
    Returns dict with step_1..step_6 and overall, or None on failure.
    """
    try:
        text = text.strip().strip("```json").strip("```").strip()
        data = json.loads(text)
        required = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6", "overall"]
        for key in required:
            if key not in data:
                data[key] = 0.0
        for key in required:
            data[key] = max(0.0, min(1.0, float(data[key])))
        # recompute overall as mean of steps to guard against judge arithmetic errors
        steps = [data[f"step_{i}"] for i in range(1, 7)]
        data["overall"] = sum(steps) / len(steps)
        return data
    except Exception as e:
        logger.warning(f"Failed to parse reasoning response: {e} | text: {text[:200]}")
        return None


# ─────────────────────────────────────────
# Judge client
# ─────────────────────────────────────────

class JudgeClient:
    """
    Async-first judge client for the vLLM OpenAI-compatible endpoint.
    All batch calls are sent concurrently via asyncio.gather so vLLM
    receives all requests at once and can schedule them as a single batch.
    """

    def __init__(
            self,
            base_url: str = "http://localhost:8000/v1",
            model: str = "Qwen2.5-Coder-7B-Instruct",
            timeout: int = JUDGE_TIMEOUT,
            temperature: float = JUDGE_TEMPERATURE,
            max_tokens: int = JUDGE_MAX_TOKENS,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _call_batch_async(
            self,
            system_prompt: str,
            prompts: list[str],
    ) -> list[str]:
        """
        Send all prompts concurrently to the vLLM endpoint.
        Returns a list of raw response strings, one per prompt.
        Exceptions are caught per-request and returned as empty strings.
        """
        client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY")
        tasks = [
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            for prompt in prompts
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for r in responses:
            if isinstance(r, Exception):
                logger.warning(f"Judge async call failed: {r}")
                results.append("")
            else:
                results.append(r.choices[0].message.content or "")
        return results

    def call_batch(self, system_prompt: str, prompts: list[str]) -> list[str]:
        """
        Synchronous wrapper around _call_batch_async.
        Fires all prompts concurrently and blocks until all responses arrive.
        """
        return asyncio.run(self._call_batch_async(system_prompt, prompts))

    def score_reasoning_batch(
            self,
            prompts: list[str],
    ) -> list[dict]:
        """
        Score a batch of reasoning traces concurrently.
        All HTTP requests are sent at once; vLLM batches them internally.

        Args:
            prompts: list of user-message strings (built by build_reasoning_prompt)

        Returns:
            list of dicts with step_1..step_6 + overall (float 0.0-1.0 each).
            Failed parses return a fallback dict of all zeros.
        """
        fallback = {f"step_{i}": 0.0 for i in range(1, 7)}
        fallback["overall"] = 0.0

        raw_responses = self.call_batch(REASONING_SYSTEM_PROMPT, prompts)
        results = []
        for raw in raw_responses:
            parsed = parse_reasoning_response(raw)
            results.append(parsed if parsed is not None else fallback.copy())
        return results
