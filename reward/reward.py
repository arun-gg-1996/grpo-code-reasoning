"""
reward/reward.py

Multi-signal reward function for GRPO code generation training.

Reward structure:
    final_reward = exec_weight * execution_score + reasoning_weight * reasoning_score

    reasoning_score = 0.7 * gemini_score + 0.3 * presence_score
        (flat weighting across all difficulties — no tiering)
        (falls back to presence_score only if Gemini API errors)

Signal coverage tiers (tracked per batch):
    exec_only       — valid execution score, no structured reasoning trace found
    exec_presence   — execution + presence score (Gemini failed or skipped)
    exec_full       — execution + presence + Gemini score

Data flow contract (DO NOT shuffle inside this function):
    completions: List[str], length = batch_size * G
    kwargs["difficulty"]: List[str], same length, repeated G times per problem
    kwargs["test_cases"]: List[Any], same length, repeated G times per problem
    kwargs["problem_ids"]: List[str], same length, for dataset coverage tracking

    ordering: problem 1 → [0:G], problem 2 → [G:2G], etc.
    GRPO advantage computation depends on this ordering.
"""

import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import wandb

from sandbox.testing_util import run_test
from config import (
    EXEC_WEIGHT,
    REASONING_WEIGHT,
    GEMINI_WEIGHT,
    PRESENCE_WEIGHT,
    MIN_STEPS,
    GEMINI_MAX_WORKERS,
    SANDBOX_MAX_WORKERS,
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    GEMINI_CORRELATION_INTERVAL,
    GROUP_SIZE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level persistent thread pools — do NOT create inside reward_fn
# ---------------------------------------------------------------------------
_sandbox_pool = ThreadPoolExecutor(max_workers=SANDBOX_MAX_WORKERS)
_gemini_pool = ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS)

# Dataset coverage tracking — stores problem IDs seen across all steps
_seen_problem_ids: set[str] = set()

# Rolling buffers for Gemini-presence correlation (last GEMINI_CORRELATION_INTERVAL steps)
_gemini_score_buffer: list[float] = []
_presence_score_buffer: list[float] = []

# Step counter for correlation logging
_reward_fn_call_count = 0


# ---------------------------------------------------------------------------
# Gemini judge
# ---------------------------------------------------------------------------

def _call_gemini(reasoning_text: str, problem_description: str) -> float | None:
    """
    Call Gemini to score reasoning quality of an already-extracted reasoning trace.
    Returns float in [0, 1] or None on any error.

    reasoning_text: content extracted from inside <think> tags (pre-extracted by caller)
    problem_description: the problem statement, for context

    Returns None on API error → caller falls back to presence score.
    """
    if not reasoning_text:
        return None

    try:
        import google.generativeai as genai
        from config import GEMINI_API_KEY

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=JUDGE_MODEL,
            system_instruction=JUDGE_SYSTEM_PROMPT,
        )

        prompt = f"Problem:\n{problem_description}\n\nReasoning trace:\n{reasoning_text}"

        response = model.generate_content(prompt)
        score_text = response.text.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))

    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Presence heuristic
# ---------------------------------------------------------------------------

def _presence_score(completion: str) -> tuple[float, int]:
    """
    Score based on presence and count of [STEP] blocks inside <think> tags.
    Returns (score, steps_found).

    Scoring:
        steps < MIN_STEPS  → partial credit: steps / MIN_STEPS
        steps >= MIN_STEPS → 1.0  (never penalize more steps)
    """
    # Search inside <think> block if present, else fall back to full completion
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL | re.IGNORECASE)
    search_text = think_match.group(1) if think_match else completion

    steps = re.findall(r'\[STEP\]', search_text, re.IGNORECASE)
    steps_found = len(steps)

    score = steps_found / MIN_STEPS if steps_found < MIN_STEPS else 1.0
    return score, steps_found


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_reasoning(completion: str) -> str:
    """Extract content inside <think> tags. Returns empty string if not found."""
    match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_code(completion: str) -> str | None:
    """Extract content inside <code> tags. Returns None if not found."""
    match = re.search(r'<code>(.*?)</code>', completion, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _count_tokens(text: str, tokenizer) -> int:
    """Count tokens using the actual tokenizer, not whitespace split."""
    if tokenizer is None:
        return len(text.split())  # fallback
    return len(tokenizer(text, return_tensors=None)["input_ids"])


def _is_truncated(completion: str, max_new_tokens: int, tokenizer) -> bool:
    """Check if completion likely hit MAX_NEW_TOKENS cutoff."""
    token_count = _count_tokens(completion, tokenizer)
    return token_count >= (max_new_tokens - 10)  # small buffer for off-by-one


# ---------------------------------------------------------------------------
# Per-completion reward computation
# ---------------------------------------------------------------------------

def _compute_single_reward(
        completion: str,
        test_cases: Any,
        problem_description: str,
        difficulty: str,
        tokenizer,
        max_new_tokens: int,
) -> dict:
    """
    Compute all reward components for a single completion.
    Returns a dict with all scores and diagnostic info.
    Does NOT call Gemini — that's handled in parallel separately.
    """
    code = _extract_code(completion)
    reasoning = _extract_reasoning(completion)
    presence, steps_found = _presence_score(completion)

    # Execution score
    if code is not None:
        try:
            exec_result = run_test(code, test_cases)
            exec_score = exec_result.get("passed", 0) / max(exec_result.get("total", 1), 1)
        except Exception as e:
            logger.warning(f"Sandbox error: {e}")
            exec_score = 0.0
    else:
        exec_score = 0.0

    # Token counts
    reasoning_tokens = _count_tokens(reasoning, tokenizer)
    code_tokens = _count_tokens(code, tokenizer) if code else 0
    truncated = _is_truncated(completion, max_new_tokens, tokenizer)

    return {
        "exec_score": exec_score,
        "presence_score": presence,
        "steps_found": steps_found,
        "has_code": code is not None,
        "has_reasoning": len(reasoning) > 0,
        "reasoning_tokens": reasoning_tokens,
        "code_tokens": code_tokens,
        "truncated": truncated,
        "difficulty": difficulty,
        "reasoning_text": reasoning,  # pre-extracted, passed directly to Gemini
        "problem_description": problem_description,
    }


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def reward_fn(
        completions: list[str],
        prompts: list[str],
        tokenizer=None,
        max_new_tokens: int = 4096,
        **kwargs,
) -> list[float]:
    """
    Main reward function called by GRPOTrainer.

    Args:
        completions: List of model completions, length = batch_size * G
        prompts: List of prompts, same length
        tokenizer: HuggingFace tokenizer for token counting
        max_new_tokens: Training MAX_NEW_TOKENS for truncation detection
        **kwargs:
            difficulty: List[str] — difficulty per completion
            test_cases: List[Any] — test cases per completion
            problem_ids: List[str] — problem IDs for coverage tracking
            problem_descriptions: List[str] — for Gemini judge

    Returns:
        List[float] of final rewards, same length as completions.
        ORDER IS PRESERVED — critical for GRPO advantage computation.
    """
    global _reward_fn_call_count, _gemini_score_buffer, _presence_score_buffer

    _reward_fn_call_count += 1
    n = len(completions)

    difficulties = kwargs.get("difficulty", ["unknown"] * n)
    test_cases_list = kwargs.get("test_cases", [None] * n)
    problem_ids = kwargs.get("problem_ids", [f"unknown_{i}" for i in range(n)])
    problem_descriptions = kwargs.get("problem_descriptions", [""] * n)

    # Update dataset coverage
    _seen_problem_ids.update(problem_ids)

    # ------------------------------------------------------------------
    # Step 1: Run sandbox in parallel (CPU-bound)
    # ------------------------------------------------------------------
    sandbox_futures = {}
    for i, completion in enumerate(completions):
        future = _sandbox_pool.submit(
            _compute_single_reward,
            completion,
            test_cases_list[i],
            problem_descriptions[i],
            difficulties[i],
            tokenizer,
            max_new_tokens,
        )
        sandbox_futures[future] = i

    sandbox_results = [None] * n
    for future in as_completed(sandbox_futures):
        i = sandbox_futures[future]
        try:
            sandbox_results[i] = future.result()
        except Exception as e:
            logger.error(f"Sandbox future error at index {i}: {e}")
            sandbox_results[i] = {
                "exec_score": 0.0, "presence_score": 0.0, "steps_found": 0,
                "has_code": False, "has_reasoning": False,
                "reasoning_tokens": 0, "code_tokens": 0, "truncated": False,
                "difficulty": difficulties[i], "reasoning_text": "",
                "problem_description": problem_descriptions[i],
            }

    # ------------------------------------------------------------------
    # Step 2: Run Gemini in parallel (I/O-bound)
    # ------------------------------------------------------------------
    gemini_futures = {}
    for i, result in enumerate(sandbox_results):
        if result["has_reasoning"]:
            future = _gemini_pool.submit(
                _call_gemini,
                result["reasoning_text"],
                result["problem_description"],
            )
            gemini_futures[future] = i

    gemini_results = [None] * n  # None = API error or no reasoning
    gemini_errors = 0
    gemini_latencies = []

    for future in as_completed(gemini_futures):
        i = gemini_futures[future]
        t0 = time.time()
        try:
            score = future.result()
            gemini_results[i] = score
            if score is None:
                gemini_errors += 1
        except Exception as e:
            logger.warning(f"Gemini future error at index {i}: {e}")
            gemini_errors += 1
        gemini_latencies.append((time.time() - t0) * 1000)

    # ------------------------------------------------------------------
    # Step 3: Combine scores
    # ------------------------------------------------------------------
    final_rewards = []
    signal_tier = []  # "exec_only", "exec_presence", "exec_full"

    batch_gemini_scores = []
    batch_presence_scores = []

    for i in range(n):
        r = sandbox_results[i]
        exec_score = r["exec_score"]
        presence_score = r["presence_score"]
        gemini_score = gemini_results[i]

        if not r["has_reasoning"]:
            # No <think> block found at all
            reasoning_score = 0.0
            tier = "exec_only"
        elif gemini_score is None:
            # Gemini failed — fall back to presence only
            reasoning_score = presence_score
            tier = "exec_presence"
        else:
            # Full signal
            reasoning_score = GEMINI_WEIGHT * gemini_score + PRESENCE_WEIGHT * presence_score
            tier = "exec_full"
            batch_gemini_scores.append(gemini_score)
            batch_presence_scores.append(presence_score)

        final_reward = EXEC_WEIGHT * exec_score + REASONING_WEIGHT * reasoning_score
        final_rewards.append(final_reward)
        signal_tier.append(tier)

    # ------------------------------------------------------------------
    # Step 4: Compute diagnostics
    # ------------------------------------------------------------------
    diff_array = np.array(difficulties)
    reward_array = np.array(final_rewards)
    exec_array = np.array([r["exec_score"] for r in sandbox_results])

    # Per-difficulty masks
    easy_mask = diff_array == "easy"
    medium_mask = diff_array == "medium"
    hard_mask = diff_array == "hard"

    # Degenerate group detection — reshape into (n_problems, G) groups
    n_problems = n // GROUP_SIZE
    reward_groups = reward_array.reshape(n_problems, GROUP_SIZE)
    group_stds = reward_groups.std(axis=1)
    all_zero = (reward_groups == 0.0).all(axis=1)
    all_correct = (reward_groups >= 0.99).all(axis=1)

    # Signal tier fractions
    tier_array = np.array(signal_tier)
    exec_only_frac = (tier_array == "exec_only").mean()
    exec_presence_frac = (tier_array == "exec_presence").mean()
    exec_full_frac = (tier_array == "exec_full").mean()

    # Steps and tokens
    steps_all = np.array([r["steps_found"] for r in sandbox_results])
    reasoning_tokens_all = np.array([r["reasoning_tokens"] for r in sandbox_results])
    code_tokens_all = np.array([r["code_tokens"] for r in sandbox_results])

    # Update rolling correlation buffers
    _gemini_score_buffer.extend(batch_gemini_scores)
    _presence_score_buffer.extend(batch_presence_scores)
    max_buffer = GEMINI_CORRELATION_INTERVAL * n
    _gemini_score_buffer = _gemini_score_buffer[-max_buffer:]
    _presence_score_buffer = _presence_score_buffer[-max_buffer:]

    # ------------------------------------------------------------------
    # Step 5: WandB logging
    # ------------------------------------------------------------------
    log_dict = {}

    # --- Reward signals ---
    log_dict["reward/mean"] = reward_array.mean()
    log_dict["reward/std"] = reward_array.std()
    log_dict["reward/non_zero_fraction"] = (reward_array > 0).mean()
    log_dict["reward/execution_mean"] = exec_array.mean()

    reasoning_scores_all = np.array([
        GEMINI_WEIGHT * (gemini_results[i] or 0) + PRESENCE_WEIGHT * sandbox_results[i]["presence_score"]
        if gemini_results[i] is not None
        else sandbox_results[i]["presence_score"]
        for i in range(n)
    ])
    log_dict["reward/reasoning_mean"] = reasoning_scores_all.mean()

    for mask, name in [(easy_mask, "easy"), (medium_mask, "medium"), (hard_mask, "hard")]:
        if mask.sum() > 0:
            log_dict[f"reward/execution_{name}"] = exec_array[mask].mean()
            log_dict[f"reward/reasoning_{name}"] = reasoning_scores_all[mask].mean()

    # --- Signal coverage tiers ---
    log_dict["reward/exec_only_fraction"] = exec_only_frac
    log_dict["reward/exec_presence_fraction"] = exec_presence_frac
    log_dict["reward/exec_full_fraction"] = exec_full_frac

    # --- GRPO degenerate groups ---
    log_dict["grpo/all_zero_fraction"] = all_zero.mean()
    log_dict["grpo/all_correct_fraction"] = all_correct.mean()
    log_dict["grpo/degenerate_fraction"] = (all_zero | all_correct).mean()
    log_dict["grpo/reward_std_mean"] = group_stds.mean()

    # --- Generation quality ---
    log_dict["gen/valid_code_fraction"] = np.array([r["has_code"] for r in sandbox_results]).mean()
    log_dict["gen/truncated_fraction"] = np.array([r["truncated"] for r in sandbox_results]).mean()
    log_dict["gen/empty_completion_fraction"] = np.array([
        len(c.strip()) == 0 for c in completions
    ]).mean()
    log_dict["gen/steps_count_mean"] = steps_all.mean()
    log_dict["gen/reasoning_tokens_mean"] = reasoning_tokens_all.mean()
    log_dict["gen/code_tokens_mean"] = code_tokens_all.mean()

    for mask, name in [(easy_mask, "easy"), (medium_mask, "medium"), (hard_mask, "hard")]:
        if mask.sum() > 0:
            log_dict[f"gen/steps_count_{name}"] = steps_all[mask].mean()
            log_dict[f"gen/valid_code_{name}"] = np.array([r["has_code"] for r in sandbox_results])[mask].mean()
            log_dict[f"gen/reasoning_tokens_{name}"] = reasoning_tokens_all[mask].mean()
            log_dict[f"gen/code_tokens_{name}"] = code_tokens_all[mask].mean()

    # --- Gemini judge reliability ---
    total_gemini_calls = len(gemini_futures)
    log_dict["judge/gemini_error_rate"] = gemini_errors / max(total_gemini_calls, 1)
    log_dict["judge/gemini_latency_mean"] = np.mean(gemini_latencies) if gemini_latencies else 0.0

    for mask, name in [(easy_mask, "easy"), (medium_mask, "medium"), (hard_mask, "hard")]:
        if mask.sum() > 0:
            errors_in_diff = sum(
                1 for i in range(n)
                if diff_array[i] == name and i in gemini_futures and gemini_results[i] is None
            )
            calls_in_diff = sum(1 for i in range(n) if diff_array[i] == name and i in gemini_futures)
            log_dict[f"judge/fallback_fraction_{name}"] = errors_in_diff / max(calls_in_diff, 1)

    # --- Dataset coverage ---
    log_dict["data/unique_problems_seen"] = len(_seen_problem_ids)
    hard_problem_ids = set(problem_ids[i] for i in range(n) if difficulties[i] == "hard")
    log_dict["data/hard_problems_attempted_this_batch"] = len(hard_problem_ids)

    # --- Gemini-presence correlation (every N steps) ---
    if (
            _reward_fn_call_count % GEMINI_CORRELATION_INTERVAL == 0
            and len(_gemini_score_buffer) > 10
    ):
        corr = np.corrcoef(_gemini_score_buffer, _presence_score_buffer)[0, 1]
        log_dict["judge/gemini_presence_correlation"] = corr if not np.isnan(corr) else 0.0

    try:
        wandb.log(log_dict)
    except Exception as e:
        logger.warning(f"WandB logging failed: {e}")

    return final_rewards
