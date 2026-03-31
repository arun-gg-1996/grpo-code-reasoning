"""
reward/reward.py

Multi-signal reward function for GRPO training.

Reward structure (per spec):
    APPS:  final = 0.75 * exec + 0.25 * reasoning
    LCB:   final = 0.60 * exec + 0.40 * reasoning

Reasoning scoring tiers:
    Easy:   presence heuristic only (Gemini std=0.076, no discrimination)
    Medium: 0.3 * presence + 0.7 * gemini
    Hard:   0.7 * presence + 0.3 * gemini

Data flow contract (DO NOT shuffle):
    completions[0:G] → problem 1, completions[G:2G] → problem 2, etc.
    GRPO advantage computation depends on this ordering.
"""

import re
import logging

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from config import (
    # Per-source weights
    APPS_EXEC_WEIGHT, APPS_REASONING_WEIGHT,
    LCB_EXEC_WEIGHT, LCB_REASONING_WEIGHT,
    # Tier-weighted reasoning
    EASY_GEMINI_WEIGHT, EASY_PRESENCE_WEIGHT,
    MEDIUM_GEMINI_WEIGHT, MEDIUM_PRESENCE_WEIGHT,
    HARD_GEMINI_WEIGHT, HARD_PRESENCE_WEIGHT,
    # Misc
    GROUP_SIZE,
    MAX_NEW_TOKENS,
    normalize_difficulty,
)
from reward.execution import score_batch as exec_score_batch, extract_code, score_single

logger = logging.getLogger(__name__)

# Dataset coverage tracking
_seen_problem_ids: set[str] = set()
_seen_by_source: dict = {"apps": set(), "lcb_seen": set()}
_seen_by_diff: dict = {"easy": set(), "medium": set(), "hard": set()}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_think_block(completion: str) -> str:
    """Extract content inside <think> tags. Returns empty string if not found."""
    match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _presence_score(think_block: str) -> float:
    """
    Presence heuristic: count [STEP] markers with >=20 chars of content.
    Score = valid_steps / 6.0, capped at 1.0.
    """
    if not think_block:
        return 0.0

    # Split on [STEP] markers (case-insensitive)
    parts = re.split(r"\[STEP\]", think_block, flags=re.IGNORECASE)
    # First element is text before the first [STEP], skip it
    steps = parts[1:] if len(parts) > 1 else []

    valid = sum(1 for s in steps[:6] if len(s.strip()) >= 20)
    return min(valid / 6.0, 1.0)


def _get_tier_weights(difficulty: str) -> tuple[float, float]:
    """Return (gemini_weight, presence_weight) for a difficulty tier."""
    if difficulty == "easy":
        return EASY_GEMINI_WEIGHT, EASY_PRESENCE_WEIGHT
    elif difficulty == "medium":
        return MEDIUM_GEMINI_WEIGHT, MEDIUM_PRESENCE_WEIGHT
    else:  # hard
        return HARD_GEMINI_WEIGHT, HARD_PRESENCE_WEIGHT


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
    problems = kwargs.get("problems", [{}] * n)
    sources = kwargs.get("source", ["apps"] * n)

    # Normalize difficulties
    difficulties = [
        normalize_difficulty(p.get("difficulty", "medium"))
        for p in problems
    ]

    # Track dataset coverage
    for p, src, diff in zip(problems, sources, difficulties):
        pid = p.get("problem_id") or p.get("question_id", "")
        if pid:
            pid_str = str(pid)
            _seen_problem_ids.add(pid_str)
            src_key = "lcb_seen" if src in ("lcb", "lcb_seen") else "apps"
            _seen_by_source.setdefault(src_key, set()).add(pid_str)
            _seen_by_diff.setdefault(diff, set()).add(pid_str)

    # ------------------------------------------------------------------
    # Step 1: Extract code and think blocks
    # ------------------------------------------------------------------
    codes = [extract_code(c) or "" for c in completions]
    think_blocks = [_extract_think_block(c) for c in completions]

    # ------------------------------------------------------------------
    # Step 2: Execution scoring (parallel via subprocess pool)
    # ------------------------------------------------------------------
    try:
        exec_scores, exec_stats = exec_score_batch(codes=codes, problems=problems)
    except AssertionError:
        # Fallback to sequential scoring (e.g., when called from daemon process)
        logger.warning("score_batch failed (daemon process?), falling back to sequential")
        exec_scores = [score_single(c, p) for c, p in zip(codes, problems)]
        exec_stats = {
            "mean_score": sum(exec_scores) / len(exec_scores) if exec_scores else 0,
            "zero_scores": sum(1 for s in exec_scores if s == 0.0),
            "perfect_scores": sum(1 for s in exec_scores if s == 1.0),
            "timeout_count": 0,
            "timeout_fraction": 0.0,
        }

    # ------------------------------------------------------------------
    # Step 3: Presence heuristic for all completions
    # ------------------------------------------------------------------
    presence_scores = [_presence_score(tb) for tb in think_blocks]

    # ------------------------------------------------------------------
    # Step 4: Gemini judge for medium/hard with think blocks
    # ------------------------------------------------------------------
    # Collect indices that need Gemini calls
    gemini_indices = []
    gemini_problems = []
    gemini_thinks = []
    gemini_diffs = []

    for i in range(n):
        if think_blocks[i] and difficulties[i] in ("medium", "hard"):
            gemini_indices.append(i)
            gemini_problems.append(problems[i].get("question", ""))
            gemini_thinks.append(think_blocks[i])
            gemini_diffs.append(difficulties[i])

    gemini_scores = [None] * n  # None = not called or no think block

    if gemini_indices:
        try:
            from reward.judge import score_batch as judge_score_batch
            raw_scores = judge_score_batch(
                problems=gemini_problems,
                think_blocks=gemini_thinks,
                difficulties=gemini_diffs,
            )
            for idx, score in zip(gemini_indices, raw_scores):
                gemini_scores[idx] = score
        except Exception as e:
            logger.warning(f"Gemini batch call failed, falling back to presence: {e}")

    # ------------------------------------------------------------------
    # Step 5: Combine scores per spec
    # ------------------------------------------------------------------
    final_rewards = []
    reasoning_scores = []

    for i in range(n):
        diff = difficulties[i]
        source = sources[i]

        # Reasoning score: tiered
        if not think_blocks[i]:
            # Case A: no <think> block
            reasoning_score = 0.0
        else:
            gem_w, pres_w = _get_tier_weights(diff)
            if gem_w == 0.0 or gemini_scores[i] is None:
                # Easy tier, or Gemini failed
                reasoning_score = presence_scores[i]
            else:
                reasoning_score = (
                    pres_w * presence_scores[i] + gem_w * gemini_scores[i]
                )

        reasoning_scores.append(reasoning_score)

        # Per-source weighting
        exec_w, reas_w = _get_source_weights(source)
        reward = exec_w * exec_scores[i] + reas_w * reasoning_score
        final_rewards.append(reward)

    # ------------------------------------------------------------------
    # Step 6: WandB logging
    # ------------------------------------------------------------------
    _log_metrics(
        final_rewards, exec_scores, presence_scores, gemini_scores,
        reasoning_scores, difficulties, sources, think_blocks, codes, completions, exec_stats,
    )

    return final_rewards


def _log_metrics(
    rewards, exec_scores, presence_scores, gemini_scores,
    reasoning_scores, difficulties, sources, think_blocks, codes, completions, exec_stats,
):
    """Log comprehensive metrics to WandB and emit console warnings on threshold breaches."""
    try:
        n = len(rewards)
        reward_arr = np.array(rewards)
        exec_arr = np.array(exec_scores)
        presence_arr = np.array(presence_scores)
        reasoning_arr = np.array(reasoning_scores)
        diff_arr = np.array(difficulties)

        log_dict = {
            # Reward signals
            "reward/mean": reward_arr.mean(),
            "reward/std": reward_arr.std(),
            "reward/non_zero_fraction": (reward_arr > 0).mean(),
            "reward/execution_mean": exec_arr.mean(),
            "reward/reasoning_mean": reasoning_arr.mean(),
            # Execution stats
            "exec/mean_score": exec_stats.get("mean_score", 0),
            "exec/zero_scores": exec_stats.get("zero_scores", 0),
            "exec/perfect_scores": exec_stats.get("perfect_scores", 0),
            "exec/timeout_count": exec_stats.get("timeout_count", 0),
            "exec/timeout_fraction": exec_stats.get("timeout_fraction", 0.0),
            # Presence / reasoning quality
            "presence/mean": presence_arr.mean(),
            # Generation quality
            "gen/valid_code_fraction": sum(1 for c in codes if c) / n,
            "gen/has_reasoning_fraction": sum(1 for t in think_blocks if t) / n,
            "gen/empty_completion_fraction": sum(1 for c in completions if not c.strip()) / n,
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
        }

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

        # Execution zero/low split
        exec_zero_frac = (exec_arr == 0.0).mean()
        nonzero_exec = exec_arr[exec_arr > 0.0]
        exec_nonzero_mean = float(nonzero_exec.mean()) if len(nonzero_exec) > 0 else None
        log_dict["exec/zero_fraction"] = exec_zero_frac
        if exec_nonzero_mean is not None:
            log_dict["exec/nonzero_mean"] = exec_nonzero_mean

        # GRPO group diagnostics
        n_problems = n // GROUP_SIZE if GROUP_SIZE > 0 else 0
        if n_problems > 0 and n == n_problems * GROUP_SIZE:
            groups = reward_arr.reshape(n_problems, GROUP_SIZE)
            group_stds = groups.std(axis=1)
            log_dict["grpo/reward_std_mean"] = group_stds.mean()
            log_dict["grpo/all_zero_fraction"] = (groups == 0.0).all(axis=1).mean()
            log_dict["grpo/all_perfect_fraction"] = (groups >= 0.99).all(axis=1).mean()

        # ── Early warning system ──────────────────────────────────────────
        # Evaluate all thresholds, collect fired warnings, then log to wandb
        warnings_fired = {}  # key → message

        reward_std_mean = log_dict.get("grpo/reward_std_mean")
        if reward_std_mean is not None and reward_std_mean < 0.05:
            warnings_fired["grpo/warn_reward_std_collapse"] = (
                f"GRPO signal collapsing — reward_std_mean={reward_std_mean:.4f} < 0.05. "
                "All rollouts getting similar rewards; advantage estimates are noise."
            )

        non_zero_frac = log_dict.get("reward/non_zero_fraction", 1.0)
        if non_zero_frac < 0.2:
            warnings_fired["warn/format_failure"] = (
                f"Format failure — only {non_zero_frac:.1%} of completions have non-zero reward. "
                "Model may not be following output format (missing <code> tags)."
            )

        all_zero_frac = log_dict.get("grpo/all_zero_fraction")
        if all_zero_frac is not None and all_zero_frac > 0.5:
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
                    "Model has stopped producing <code> tags."
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

        gemini_mean = log_dict.get("judge/gemini_mean")
        gemini_calls = log_dict.get("judge/gemini_calls", 0)
        if gemini_calls > 4 and gemini_mean is not None and abs(gemini_mean - 0.5) < 0.02:
            warnings_fired["warn/gemini_silent_failure"] = (
                f"Gemini judge may be silently failing — mean score={gemini_mean:.3f} "
                "is suspiciously close to the 0.5 fallback value. Check API key and quota."
            )

        all_perfect_frac = log_dict.get("grpo/all_perfect_fraction")
        if all_perfect_frac is not None and all_perfect_frac > 0.5:
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

        # Log to wandb: metrics + binary warning flags
        if wandb is not None and wandb.run is not None:
            wandb.log(log_dict)
            # Binary flags (1 = fired, 0 = clear) so warnings are visible on the dashboard
            flag_dict = {k: 1 for k in warnings_fired}
            for key in [
                "grpo/warn_reward_std_collapse", "grpo/warn_all_zero_collapse",
                "warn/format_failure", "warn/exec_formatting_breakdown",
                "warn/exec_zero_sandbox", "warn/exec_low_nonzero",
                "warn/reasoning_collapse", "warn/gemini_silent_failure",
                "warn/problems_too_easy", "warn/empty_completions", "warn/high_timeout_rate",
            ]:
                if key not in flag_dict:
                    flag_dict[key] = 0
            wandb.log(flag_dict)
            # Send push alert for each fired warning
            for key, msg in warnings_fired.items():
                try:
                    wandb.alert(title=key, text=msg, level=wandb.AlertLevel.WARN)
                except Exception:
                    pass  # alerts are best-effort

        # Always print to console regardless of wandb
        for msg in warnings_fired.values():
            logger.warning(f"WARNING: {msg}")

    except Exception as e:
        logger.warning(f"WandB logging failed: {e}")
