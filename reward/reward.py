"""
reward/reward.py

Multi-signal reward function for GRPO training.

Reward structure (per spec):
    APPS:  final = 0.75 * exec + 0.25 * reasoning
    LCB:   final = 0.60 * exec + 0.40 * reasoning

Reasoning scoring policy:
    - Gemini-first for all tiers (when <think> exists and judge succeeds)
    - simple presence fallback only when judge fails
    - no <think> => reasoning score 0.0

Data flow contract (DO NOT shuffle):
    completions[0:G] → problem 1, completions[G:2G] → problem 2, etc.
    GRPO advantage computation depends on this ordering.
"""

import re
import logging
import time
import os
import json
from datetime import datetime

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from config import (
    # Per-source weights
    APPS_EXEC_WEIGHT, APPS_REASONING_WEIGHT,
    LCB_EXEC_WEIGHT, LCB_REASONING_WEIGHT,
    # Misc
    GROUP_SIZE,
    MAX_NEW_TOKENS,
    WARMUP_STEPS,
    DYNAMIC_BASELINE_WINDOW_STEPS,
    REWARD_STD_WARNING_THRESHOLD,
    GRPO_ALL_ZERO_REF_MAX,
    GRPO_ALL_PERFECT_REF_MAX,
    FORMAT_PENALTY_STRICT,
    FORMAT_PENALTY_FENCE,
    normalize_difficulty,
)
from reward.execution import score_batch as exec_score_batch, extract_code, score_single

logger = logging.getLogger(__name__)

# Dataset coverage tracking
_seen_problem_ids: set[str] = set()
_seen_by_source: dict = {"apps": set(), "lcb_seen": set()}
_seen_by_diff: dict = {"easy": set(), "medium": set(), "hard": set()}
_processed_total = 0
_processed_by_source: dict = {"apps": 0, "lcb_seen": 0}
_processed_by_diff: dict = {"easy": 0, "medium": 0, "hard": 0}

# Optional per-completion training debug stream (JSONL).
_train_debug_inited = False
_train_debug_enabled = False
_train_debug_path = ""
_train_debug_fh = None
_train_debug_call_idx = 0

# Dynamic operational baselines from post-warmup early window.
_metrics_step_idx = 0
_reward_judge_baseline_s: float | None = None
_failed_to_run_baseline_frac: float | None = None
_reward_judge_samples: list[float] = []
_failed_to_run_samples: list[float] = []


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_think_block(completion: str) -> str:
    """Extract content inside <think> tags. Returns empty string if not found."""
    match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _unwrap_completion(completion) -> str:
    """
    Normalize TRL completion payloads into plain assistant text.

    TRL may pass completions as:
      - plain string
      - list[{"role": "...", "content": "..."}] chat messages
      - {"role": "...", "content": "..."} dict

    Using str(payload) on list/dict chat outputs produces repr()-escaped text
    (e.g., "\\n"), which corrupts extraction/execution. We must unwrap content.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                return content if isinstance(content, str) else str(content)

    if isinstance(completion, dict) and completion.get("role") == "assistant":
        content = completion.get("content", "")
        return content if isinstance(content, str) else str(content)

    return str(completion)


def _presence_score(think_block: str) -> float:
    """
    Simple fallback score from <think> when Gemini fails.
    Intentionally conservative to reduce reward hacking:
      - reward: several meaningful [STEP] blocks
      - penalty: very long traces and heavy repetition
    """
    if not think_block:
        return 0.0

    parts = re.split(r"\[STEP\]", think_block, flags=re.IGNORECASE)
    raw_steps = [s.strip() for s in (parts[1:] if len(parts) > 1 else []) if s.strip()]
    # If the model does not use [STEP] markers, fall back to line/sentence chunks.
    if not raw_steps:
        raw_steps = [
            s.strip()
            for s in re.split(r"\n+|(?<=[.!?])\s+", think_block)
            if s and s.strip()
        ]
    if not raw_steps:
        return 0.0

    meaningful = [s for s in raw_steps if 20 <= len(s) <= 1200]
    base = min(len(meaningful) / 6.0, 1.0)

    penalty = 0.0
    total_chars = len(think_block)
    if total_chars > 2600:
        penalty += min(0.30, (total_chars - 2600) / 5000.0)

    # Adjacent lexical overlap as simple repetition signal.
    overlaps = []
    for a_step, b_step in zip(raw_steps[:20], raw_steps[1:21]):
        a = set(re.findall(r"[a-zA-Z]{4,}", a_step.lower()))
        b = set(re.findall(r"[a-zA-Z]{4,}", b_step.lower()))
        if not a or not b:
            continue
        union = len(a | b)
        overlaps.append(len(a & b) / union)

    if overlaps:
        mean_overlap = float(np.mean(overlaps))
        if mean_overlap > 0.75:
            penalty += min(0.20, (mean_overlap - 0.75) * 0.8)

    score = base - penalty
    return float(max(0.0, min(1.0, score)))


def _get_tier_weights(difficulty: str) -> tuple[float, float]:
    """Deprecated: kept for backward compatibility with smoke tests."""
    if difficulty == "easy":
        return 1.0, 0.0
    elif difficulty == "medium":
        return 1.0, 0.0
    else:  # hard
        return 1.0, 0.0


def _get_source_weights(source: str) -> tuple[float, float]:
    """Return (exec_weight, reasoning_weight) for a data source."""
    if source == "lcb" or source == "lcb_seen":
        return LCB_EXEC_WEIGHT, LCB_REASONING_WEIGHT
    else:  # apps
        return APPS_EXEC_WEIGHT, APPS_REASONING_WEIGHT


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def reward_fn(
    completions: list[str],
    prompts: list[str],
    **kwargs,
) -> list[float]:
    """
    Main reward function called by GRPOTrainer.

    Args:
        completions: model completions, length = batch_size * G
        prompts: prompts, same length (unused but required by TRL)
        **kwargs:
            problems: list[dict] — full problem dicts from JSONL
            source: list[str] — "apps" or "lcb"/"lcb_seen" per completion

    Returns:
        list[float] of final rewards, ORDER PRESERVED.
    """
    n = len(completions)
    if n == 0:
        return []

    # Defensive alignment: preserve order and length expected by GRPO.
    if len(prompts) != n:
        logger.warning("reward_fn alignment: prompts=%d completions=%d", len(prompts), n)
    problems = list(kwargs.get("problems", []))
    sources = list(kwargs.get("source", []))
    if len(problems) < n:
        problems = problems + ([{}] * (n - len(problems)))
    if len(sources) < n:
        sources = sources + (["apps"] * (n - len(sources)))
    problems = problems[:n]
    sources = sources[:n]

    # GRPO expects contiguous groups of G rollouts per problem.
    if GROUP_SIZE > 0 and (n % GROUP_SIZE) != 0:
        logger.warning(
            "reward_fn group integrity warning: completions=%d not divisible by GROUP_SIZE=%d",
            n, GROUP_SIZE
        )

    completions = [_unwrap_completion(c) for c in completions]

    # Normalize difficulties
    difficulties = [
        normalize_difficulty(p.get("difficulty", "medium"))
        for p in problems
    ]

    # Track dataset coverage
    global _processed_total
    for p, src, diff in zip(problems, sources, difficulties):
        _processed_total += 1
        src_key = "lcb_seen" if src in ("lcb", "lcb_seen") else "apps"
        _processed_by_source[src_key] = _processed_by_source.get(src_key, 0) + 1
        _processed_by_diff[diff] = _processed_by_diff.get(diff, 0) + 1

        pid = p.get("problem_id") or p.get("question_id", "")
        if pid:
            pid_str = str(pid)
            _seen_problem_ids.add(pid_str)
            _seen_by_source.setdefault(src_key, set()).add(pid_str)
            _seen_by_diff.setdefault(diff, set()).add(pid_str)

    t_reward_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Extract code and think blocks
    # ------------------------------------------------------------------
    extraction_results = [extract_code(c) for c in completions]
    codes = [code or "" for code, _mode in extraction_results]
    extract_modes = [mode for _code, mode in extraction_results]
    format_multipliers = [
        FORMAT_PENALTY_STRICT if mode == "strict"
        else FORMAT_PENALTY_FENCE if mode == "fence"
        else 0.0
        for mode in extract_modes
    ]
    think_blocks = [_extract_think_block(c) for c in completions]

    # ------------------------------------------------------------------
    # Step 2: Execution scoring (parallel via subprocess pool)
    # ------------------------------------------------------------------
    t_exec_start = time.perf_counter()
    try:
        exec_scores, exec_stats = exec_score_batch(codes=codes, problems=problems)
    except Exception:
        # Fallback to sequential scoring if batch path fails for any reason.
        logger.warning("score_batch failed, falling back to sequential")
        exec_scores = [score_single(c, p) for c, p in zip(codes, problems)]
        exec_stats = {
            "mean_score": sum(exec_scores) / len(exec_scores) if exec_scores else 0,
            "zero_scores": sum(1 for s in exec_scores if s == 0.0),
            "perfect_scores": sum(1 for s in exec_scores if s == 1.0),
            "timeout_count": 0,
            "timeout_fraction": 0.0,
        }
    exec_time_s = time.perf_counter() - t_exec_start

    # ------------------------------------------------------------------
    # Step 3: Presence heuristic for all completions
    # ------------------------------------------------------------------
    presence_scores = [_presence_score(tb) for tb in think_blocks]

    # ------------------------------------------------------------------
    # Step 4: Gemini judge for all completions with think blocks
    # ------------------------------------------------------------------
    # Collect indices that need Gemini calls
    gemini_indices = []
    gemini_problems = []
    gemini_thinks = []
    gemini_diffs = []

    for i in range(n):
        if think_blocks[i]:
            gemini_indices.append(i)
            gemini_problems.append(problems[i].get("question", ""))
            gemini_thinks.append(think_blocks[i])
            gemini_diffs.append(difficulties[i])

    gemini_scores = [None] * n  # None = not called or no think block

    judge_time_s = 0.0
    if gemini_indices:
        t_judge_start = time.perf_counter()
        try:
            from reward.judge import score_batch as judge_score_batch, get_last_batch_stats
            raw_scores = judge_score_batch(
                problems=gemini_problems,
                think_blocks=gemini_thinks,
                difficulties=gemini_diffs,
            )
            judge_stats = get_last_batch_stats()
            fallback_mask = judge_stats.get("fallback_mask", [])
            for j, (idx, score) in enumerate(zip(gemini_indices, raw_scores)):
                if j < len(fallback_mask) and fallback_mask[j]:
                    gemini_scores[idx] = None
                else:
                    gemini_scores[idx] = score
        except Exception as e:
            logger.warning(f"Gemini batch call failed, falling back to presence: {e}")
        judge_time_s = time.perf_counter() - t_judge_start

    # ------------------------------------------------------------------
    # Step 5: Combine scores per spec
    # ------------------------------------------------------------------
    final_rewards = []
    reasoning_scores = []
    penalized_exec_scores = []
    gemini_used_count = 0
    presence_used_count = 0

    for i in range(n):
        source = sources[i]

        # Reasoning score: Gemini-first; fallback to presence only on judge failure.
        if not think_blocks[i]:
            reasoning_score = 0.0
        elif gemini_scores[i] is not None:
            reasoning_score = gemini_scores[i]
            gemini_used_count += 1
        else:
            reasoning_score = presence_scores[i]
            presence_used_count += 1

        reasoning_scores.append(reasoning_score)

        # Per-source weighting
        exec_w, reas_w = _get_source_weights(source)
        penalized_exec = exec_scores[i] * format_multipliers[i]
        penalized_exec_scores.append(penalized_exec)
        reward = exec_w * penalized_exec + reas_w * reasoning_score
        final_rewards.append(reward)

    # ------------------------------------------------------------------
    # Step 6: WandB logging
    # ------------------------------------------------------------------
    _log_train_debug_rows(
        rewards=final_rewards,
        exec_scores=exec_scores,
        presence_scores=presence_scores,
        gemini_scores=gemini_scores,
        reasoning_scores=reasoning_scores,
        difficulties=difficulties,
        sources=sources,
        problems=problems,
        prompts=prompts,
        codes=codes,
        extract_modes=extract_modes,
        penalized_exec_scores=penalized_exec_scores,
        completions=completions,
    )

    _log_metrics(
        final_rewards, exec_scores, presence_scores, gemini_scores,
        reasoning_scores, penalized_exec_scores, extract_modes,
        difficulties, sources, think_blocks, codes, completions, exec_stats,
        timing={
            "timing/reward_total_s": time.perf_counter() - t_reward_start,
            "timing/reward_execution_s": exec_time_s,
            "timing/reward_judge_s": judge_time_s,
        },
        gemini_used_count=gemini_used_count,
        presence_used_count=presence_used_count,
    )

    return final_rewards


def _init_train_debug_logger() -> None:
    """Lazy-init JSONL debug file handle for per-completion training traces."""
    global _train_debug_inited, _train_debug_enabled, _train_debug_path, _train_debug_fh
    if _train_debug_inited:
        return
    _train_debug_inited = True

    enabled_raw = os.environ.get("TRAIN_SAVE_DEBUG_DETAILS", "0").strip().lower()
    _train_debug_enabled = enabled_raw in ("1", "true", "yes", "on")
    _train_debug_path = os.environ.get("TRAIN_DEBUG_DETAILS_PATH", "").strip()
    if not _train_debug_enabled or not _train_debug_path:
        _train_debug_enabled = False
        return

    os.makedirs(os.path.dirname(os.path.abspath(_train_debug_path)), exist_ok=True)
    _train_debug_fh = open(_train_debug_path, "a", buffering=1)
    logger.info(f"Streaming training completion details to {_train_debug_path}")


def _log_train_debug_rows(
    rewards: list[float],
    exec_scores: list[float],
    presence_scores: list[float],
    gemini_scores: list[float | None],
    reasoning_scores: list[float],
    penalized_exec_scores: list[float],
    difficulties: list[str],
    sources: list[str],
    problems: list[dict],
    prompts: list[str],
    codes: list[str],
    extract_modes: list[str],
    completions: list[str],
) -> None:
    """Append one JSONL row per completion for post-hoc debugging."""
    global _train_debug_call_idx
    try:
        _init_train_debug_logger()
        if not _train_debug_enabled or _train_debug_fh is None:
            return

        _train_debug_call_idx += 1
        call_idx = _train_debug_call_idx
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for i in range(len(completions)):
            p = problems[i] if i < len(problems) else {}
            row = {
                "timestamp": ts,
                "reward_call_index": call_idx,
                "completion_index_in_call": i,
                "problem_batch_index": i // GROUP_SIZE,
                "rollout_index": i % GROUP_SIZE,
                "problem_id": p.get("problem_id"),
                "question_id": p.get("question_id"),
                "difficulty": difficulties[i] if i < len(difficulties) else "unknown",
                "source": sources[i] if i < len(sources) else "unknown",
                "execution_score": float(exec_scores[i]) if i < len(exec_scores) else 0.0,
                "presence_score": float(presence_scores[i]) if i < len(presence_scores) else 0.0,
                "gemini_score": (
                    None if i >= len(gemini_scores) or gemini_scores[i] is None
                    else float(gemini_scores[i])
                ),
                "reasoning_score": float(reasoning_scores[i]) if i < len(reasoning_scores) else 0.0,
                "penalized_execution_score": (
                    float(penalized_exec_scores[i]) if i < len(penalized_exec_scores) else 0.0
                ),
                "final_reward": float(rewards[i]) if i < len(rewards) else 0.0,
                "prompt": prompts[i] if i < len(prompts) else "",
                "problem_statement": p.get("question", ""),
                "extracted_code": codes[i] if i < len(codes) else "",
                "extract_mode": extract_modes[i] if i < len(extract_modes) else "none",
                "completion_text": completions[i],
                "has_code_block": bool(codes[i].strip()) if i < len(codes) else False,
            }
            _train_debug_fh.write(json.dumps(row) + "\n")
        _train_debug_fh.flush()
    except Exception as e:
        logger.warning(f"Failed to write training debug rows: {e}")


def _log_metrics(
    rewards, exec_scores, presence_scores, gemini_scores,
    reasoning_scores, penalized_exec_scores, extract_modes,
    difficulties, sources, think_blocks, codes, completions, exec_stats,
    timing=None,
    gemini_used_count=0,
    presence_used_count=0,
):
    """Log comprehensive metrics to WandB and emit console warnings on threshold breaches."""
    global _metrics_step_idx
    global _reward_judge_baseline_s, _failed_to_run_baseline_frac
    global _reward_judge_samples, _failed_to_run_samples
    _metrics_step_idx += 1
    try:
        n = len(rewards)
        reward_arr = np.array(rewards)
        exec_arr = np.array(exec_scores)
        presence_arr = np.array(presence_scores)
        reasoning_arr = np.array(reasoning_scores)
        penalized_exec_arr = np.array(penalized_exec_scores)
        diff_arr = np.array(difficulties)

        log_dict = {
            # Reward signals
            "reward/mean": reward_arr.mean(),
            "reward/std": reward_arr.std(),
            "reward/non_zero_fraction": (reward_arr > 0).mean(),
            "reward/execution_mean": exec_arr.mean(),
            "exec/penalized_execution_mean": penalized_exec_arr.mean(),
            "reward/reasoning_mean": reasoning_arr.mean(),
            # Execution stats
            "exec/mean_score": exec_stats.get("mean_score", 0),
            "exec/zero_scores": exec_stats.get("zero_scores", 0),
            "exec/perfect_scores": exec_stats.get("perfect_scores", 0),
            "exec/ok_count": exec_stats.get("ok_count", 0),
            "exec/error_count": exec_stats.get("error_count", 0),
            "exec/empty_count": exec_stats.get("empty_count", 0),
            "exec/timeout_count": exec_stats.get("timeout_count", 0),
            "exec/zero_ok_count": exec_stats.get("zero_ok_count", 0),
            "exec/timeout_fraction": exec_stats.get("timeout_fraction", 0.0),
            # Presence / reasoning quality
            "presence/mean": presence_arr.mean(),
            # Generation quality
            "gen/valid_code_fraction": sum(1 for c in codes if c) / n,
            "gen/has_reasoning_fraction": sum(1 for t in think_blocks if t) / n,
            "gen/empty_completion_fraction": sum(1 for c in completions if not c.strip()) / n,
            "exec/extract_strict_fraction": sum(1 for m in extract_modes if m == "strict") / n,
            "exec/extract_fence_fraction": sum(1 for m in extract_modes if m == "fence") / n,
            "exec/extract_none_fraction": sum(1 for m in extract_modes if m == "none") / n,
            # Completion length (chars; ~4 chars per token as rough proxy)
            # Watch these over training: think_chars should grow (more reasoning),
            # truncated_fraction should stay near 0 (if high, increase MAX_NEW_TOKENS)
            "gen/mean_completion_chars": float(np.mean([len(c) for c in completions])),
            "gen/mean_think_chars": float(np.mean([len(t) for t in think_blocks])),
            "gen/mean_code_chars": float(np.mean([len(c) for c in codes if c])) if any(codes) else 0.0,
            "gen/likely_truncated_fraction": sum(
                1 for c in completions if len(c) > MAX_NEW_TOKENS * 4 * 0.9
            ) / n,
            # Dataset coverage
            "data/unique_problems_seen": len(_seen_problem_ids),
            "data/apps_seen": len(_seen_by_source.get("apps", set())),
            "data/lcb_seen": len(_seen_by_source.get("lcb_seen", set())),
            "data/easy_seen": len(_seen_by_diff.get("easy", set())),
            "data/medium_seen": len(_seen_by_diff.get("medium", set())),
            "data/hard_seen": len(_seen_by_diff.get("hard", set())),
            "data/processed_total": _processed_total,
            "data/apps_processed": _processed_by_source.get("apps", 0),
            "data/lcb_processed": _processed_by_source.get("lcb_seen", 0),
            "data/easy_processed": _processed_by_diff.get("easy", 0),
            "data/medium_processed": _processed_by_diff.get("medium", 0),
            "data/hard_processed": _processed_by_diff.get("hard", 0),
            # Gemini stats (default zeros so charts exist even when no judge calls happen)
            "judge/gemini_mean": 0.0,
            "judge/gemini_calls": 0,
            "judge/gemini_used_count": 0,
            "judge/presence_used_count": 0,
            "judge/total_calls": 0,
            "judge/fallback_count": 0,
            "judge/fallback_fraction": 0.0,
            "judge/json_count": 0,
            "judge/json_fraction": 0.0,
            "judge/retry_count": 0,
            "grpo/warn_reward_std_collapse": 0,
            "judge/rate_limit_count": 0,
            "judge/consecutive_rate_limit_steps": 0,
            # Universal GRPO-side reference lines (for chart overlays).
            "grpo/reward_std_ref_floor": float(REWARD_STD_WARNING_THRESHOLD),
            "grpo/all_zero_ref_max": float(GRPO_ALL_ZERO_REF_MAX),
            "grpo/all_perfect_ref_max": float(GRPO_ALL_PERFECT_REF_MAX),
        }
        if timing:
            log_dict.update(timing)
        if n > 0:
            log_dict["exec/ok_fraction"] = log_dict["exec/ok_count"] / n
            log_dict["exec/error_fraction"] = log_dict["exec/error_count"] / n
            log_dict["exec/empty_fraction"] = log_dict["exec/empty_count"] / n

        # Per-difficulty breakdown
        for diff_name in ("easy", "medium", "hard"):
            mask = diff_arr == diff_name
            if mask.sum() > 0:
                log_dict[f"reward/mean_{diff_name}"] = reward_arr[mask].mean()
                log_dict[f"reward/execution_{diff_name}"] = exec_arr[mask].mean()
                log_dict[f"reward/reasoning_{diff_name}"] = reasoning_arr[mask].mean()

        # Per-source reward breakdown
        source_arr = np.array(sources)
        apps_mask = source_arr == "apps"
        lcb_mask = (source_arr == "lcb_seen") | (source_arr == "lcb")
        if apps_mask.sum() > 0:
            log_dict["reward/apps_mean"] = reward_arr[apps_mask].mean()
            log_dict["exec/apps_mean"] = exec_arr[apps_mask].mean()
        if lcb_mask.sum() > 0:
            log_dict["reward/lcb_mean"] = reward_arr[lcb_mask].mean()
            log_dict["exec/lcb_mean"] = exec_arr[lcb_mask].mean()

        # Gemini stats
        gemini_called = [i for i in range(n) if gemini_scores[i] is not None]
        if gemini_called:
            g_scores = [gemini_scores[i] for i in gemini_called]
            log_dict["judge/gemini_mean"] = np.mean(g_scores)
            log_dict["judge/gemini_calls"] = len(gemini_called)
            try:
                from reward.judge import get_last_batch_stats
                judge_stats = get_last_batch_stats()
                log_dict["judge/total_calls"] = judge_stats.get("total_calls", 0)
                log_dict["judge/fallback_count"] = judge_stats.get("fallback_count", 0)
                log_dict["judge/fallback_fraction"] = judge_stats.get("fallback_fraction", 0.0)
                log_dict["judge/json_count"] = judge_stats.get(
                    "json_count", judge_stats.get("step_json_count", 0)
                )
                log_dict["judge/json_fraction"] = judge_stats.get(
                    "json_fraction", judge_stats.get("step_json_fraction", 0.0)
                )
                log_dict["judge/retry_count"] = judge_stats.get("retry_count", 0)
                log_dict["judge/rate_limit_count"] = judge_stats.get("rate_limit_count", 0)
                log_dict["judge/consecutive_rate_limit_steps"] = judge_stats.get(
                    "consecutive_rate_limit_steps", 0
                )
            except Exception:
                pass

        log_dict["judge/gemini_used_count"] = gemini_used_count
        log_dict["judge/presence_used_count"] = presence_used_count

        # Execution zero/low split
        exec_zero_frac = (exec_arr == 0.0).mean()
        nonzero_exec = exec_arr[exec_arr > 0.0]
        exec_nonzero_mean = float(nonzero_exec.mean()) if len(nonzero_exec) > 0 else None
        log_dict["exec/zero_fraction"] = exec_zero_frac
        if n > 0:
            infra_zero_count = (
                log_dict.get("exec/error_count", 0)
                + log_dict.get("exec/timeout_count", 0)
                + log_dict.get("exec/empty_count", 0)
            )
            log_dict["exec/failed_to_run_fraction"] = infra_zero_count / n   # errors + timeouts + empty
            log_dict["exec/wrong_solution_fraction"] = log_dict.get("exec/zero_ok_count", 0) / n  # ran but failed tests
        if exec_nonzero_mean is not None:
            log_dict["exec/nonzero_mean"] = exec_nonzero_mean

        # Dynamic operational baselines (post-warmup early window median).
        baseline_end_step = int(WARMUP_STEPS) + int(DYNAMIC_BASELINE_WINDOW_STEPS)
        if (
            _reward_judge_baseline_s is None
            and _metrics_step_idx > int(WARMUP_STEPS)
            and _metrics_step_idx <= baseline_end_step
            and "timing/reward_judge_s" in log_dict
        ):
            _reward_judge_samples.append(float(log_dict["timing/reward_judge_s"]))
        if (
            _failed_to_run_baseline_frac is None
            and _metrics_step_idx > int(WARMUP_STEPS)
            and _metrics_step_idx <= baseline_end_step
            and "exec/failed_to_run_fraction" in log_dict
        ):
            _failed_to_run_samples.append(float(log_dict["exec/failed_to_run_fraction"]))
        if _metrics_step_idx > baseline_end_step:
            if _reward_judge_baseline_s is None and _reward_judge_samples:
                _reward_judge_baseline_s = float(np.median(_reward_judge_samples))
            if _failed_to_run_baseline_frac is None and _failed_to_run_samples:
                _failed_to_run_baseline_frac = float(np.median(_failed_to_run_samples))
        if _reward_judge_baseline_s is not None:
            log_dict["timing/reward_judge_s_baseline"] = float(_reward_judge_baseline_s)
        if _failed_to_run_baseline_frac is not None:
            log_dict["exec/failed_to_run_fraction_baseline"] = float(_failed_to_run_baseline_frac)

        # GRPO group diagnostics
        n_problems = n // GROUP_SIZE if GROUP_SIZE > 0 else 0
        if n_problems > 0 and n == n_problems * GROUP_SIZE:
            groups = reward_arr.reshape(n_problems, GROUP_SIZE)
            group_stds = groups.std(axis=1)
            log_dict["grpo/reward_std_mean"] = group_stds.mean()
            log_dict["grpo/all_zero_fraction"] = (groups == 0.0).all(axis=1).mean()
            log_dict["grpo/all_perfect_fraction"] = (groups >= 0.99).all(axis=1).mean()

            exec_groups = exec_arr.reshape(n_problems, GROUP_SIZE)
            # pass@G proxy in training loop:
            # - pass_at_G: any rollout gets non-zero execution score
            # - perfect_at_G: any rollout gets full execution score
            log_dict["grpo/pass_at_G_fraction"] = (exec_groups.max(axis=1) > 0.0).mean()
            log_dict["grpo/perfect_at_G_fraction"] = (exec_groups.max(axis=1) >= 1.0).mean()

        # ── Early warning system ──────────────────────────────────────────
        # Evaluate all thresholds, collect fired warnings, then log to wandb
        warnings_fired = {}  # key → message

        reward_std_mean = log_dict.get("grpo/reward_std_mean")
        if reward_std_mean is not None and reward_std_mean < REWARD_STD_WARNING_THRESHOLD:
            warnings_fired["grpo/warn_reward_std_collapse"] = (
                f"GRPO signal collapsing — reward_std_mean={reward_std_mean:.4f} < "
                f"{REWARD_STD_WARNING_THRESHOLD:.2f}. "
                "All rollouts getting similar rewards; advantage estimates are noise."
            )
            log_dict["grpo/warn_reward_std_collapse"] = 1

        non_zero_frac = log_dict.get("reward/non_zero_fraction", 1.0)
        if non_zero_frac < 0.2:
            warnings_fired["warn/format_failure"] = (
                f"Format failure — only {non_zero_frac:.1%} of completions have non-zero reward. "
                "Model may not be following output format (missing/invalid code blocks)."
            )

        all_zero_frac = log_dict.get("grpo/all_zero_fraction")
        if all_zero_frac is not None and all_zero_frac > GRPO_ALL_ZERO_REF_MAX:
            warnings_fired["grpo/warn_all_zero_collapse"] = (
                f"GRPO collapse — {all_zero_frac:.1%} of groups are all-zero reward. "
                "Advantage is undefined for these groups; learning has stalled."
            )

        if exec_zero_frac > 0.8:
            valid_code_frac = log_dict.get("gen/valid_code_fraction", 1.0)
            if valid_code_frac < 0.5:
                warnings_fired["warn/exec_formatting_breakdown"] = (
                    f"Formatting breakdown — {exec_zero_frac:.1%} execution zero, "
                    f"only {valid_code_frac:.1%} completions have valid code blocks. "
                    "Model has stopped producing parseable code blocks."
                )
            else:
                warnings_fired["warn/exec_zero_sandbox"] = (
                    f"Execution zero rate critical — {exec_zero_frac:.1%} zero, "
                    "but code blocks are present. Check test case format or sandbox errors."
                )

        if exec_nonzero_mean is not None and exec_nonzero_mean < 0.15 and exec_zero_frac < 0.8:
            warnings_fired["warn/exec_low_nonzero"] = (
                f"Model capability issue — execution scores low but non-zero "
                f"(nonzero mean={exec_nonzero_mean:.3f}). Code runs but fails most test cases."
            )

        has_reasoning_frac = log_dict.get("gen/has_reasoning_fraction", 1.0)
        if has_reasoning_frac < 0.2:
            warnings_fired["warn/reasoning_collapse"] = (
                f"Reasoning collapse — only {has_reasoning_frac:.1%} of completions "
                "have <think> blocks. Model has stopped reasoning; reasoning reward signal is dead."
            )

        fallback_fraction = log_dict.get("judge/fallback_fraction", 0.0)
        total_calls = log_dict.get("judge/total_calls", 0)
        if total_calls > 4 and fallback_fraction > 0.5:
            warnings_fired["warn/gemini_silent_failure"] = (
                f"Gemini judge failing — {fallback_fraction:.1%} of calls fell back to presence score "
                f"({int(fallback_fraction * total_calls)}/{total_calls} calls). "
                "Check API key, quota, and network connectivity."
            )

        all_perfect_frac = log_dict.get("grpo/all_perfect_fraction")
        if all_perfect_frac is not None and all_perfect_frac > GRPO_ALL_PERFECT_REF_MAX:
            warnings_fired["warn/problems_too_easy"] = (
                f"Problems too easy — {all_perfect_frac:.1%} of groups are all-perfect. "
                "No contrast within groups; GRPO has no learning signal."
            )

        empty_frac = log_dict.get("gen/empty_completion_fraction", 0.0)
        if empty_frac > 0.3:
            warnings_fired["warn/empty_completions"] = (
                f"Empty completions — {empty_frac:.1%} of completions are empty strings. "
                "Model has stopped generating output entirely."
            )

        timeout_frac = log_dict.get("exec/timeout_fraction", 0.0)
        if timeout_frac > 0.3:
            warnings_fired["warn/high_timeout_rate"] = (
                f"High timeout rate — {timeout_frac:.1%} of executions timed out. "
                "Consider increasing EXEC_TIMEOUT in config or reducing MAX_TEST_CASES."
            )

        # Observer roll-up metrics: single glance indicators for dashboard monitoring.
        observer_critical_keys = {
            "grpo/warn_reward_std_collapse",
            "grpo/warn_all_zero_collapse",
            "warn/format_failure",
            "warn/exec_formatting_breakdown",
            "warn/exec_zero_sandbox",
            "warn/reasoning_collapse",
            "warn/gemini_silent_failure",
        }
        critical_fired = [k for k in warnings_fired if k in observer_critical_keys]
        log_dict["observer/attention_count"] = len(warnings_fired)
        log_dict["observer/attention_flag"] = 1 if warnings_fired else 0
        log_dict["observer/critical_count"] = len(critical_fired)
        log_dict["observer/critical_flag"] = 1 if critical_fired else 0

        # Log to wandb in a single row: metrics + binary warning flags
        if wandb is not None and wandb.run is not None:
            # Binary flags (1 = fired, 0 = clear) so warnings align with main metrics.
            flag_dict = {k: 1 for k in warnings_fired}
            for key in [
                "grpo/warn_reward_std_collapse",
                "grpo/warn_all_zero_collapse",
                "warn/format_failure",
                "warn/exec_formatting_breakdown",
                "warn/exec_zero_sandbox",
                "warn/exec_low_nonzero",
                "warn/reasoning_collapse",
                "warn/gemini_silent_failure",
                "warn/problems_too_easy",
                "warn/empty_completions",
                "warn/high_timeout_rate",
            ]:
                if key not in flag_dict:
                    flag_dict[key] = 0
            log_dict.update(flag_dict)
            wandb.log(log_dict)
            # Send push alert for each fired warning
            for key, msg in warnings_fired.items():
                try:
                    wandb.alert(title=f"OBSERVER/{key}", text=msg, level=wandb.AlertLevel.WARN)
                except Exception:
                    pass  # alerts are best-effort

        # Always print to console regardless of wandb
        for msg in warnings_fired.values():
            logger.warning(f"OBSERVER_ALERT: {msg}")

    except Exception as e:
        logger.warning(f"WandB logging failed: {e}")
