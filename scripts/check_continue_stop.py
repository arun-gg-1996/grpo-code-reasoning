#!/usr/bin/env python3
"""
Check GRPO training health from a W&B run and print CONTINUE or STOP.

Usage:
  python scripts/check_continue_stop.py \
    --entity arun-gv-ghontale-suny-the-state-university-of-new-york \
    --project grpo-code-gen \
    --run z88tcq0i
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def _series(history, key: str) -> List[float]:
    vals = []
    for row in history:
        v = row.get(key)
        if _is_number(v):
            vals.append(float(v))
    return vals


def _series_any(history, keys: List[str]) -> List[float]:
    """Return first non-empty numeric series from candidate keys."""
    for key in keys:
        vals = _series(history, key)
        if vals:
            return vals
    return []


def _step_from_row(row: dict) -> float | None:
    for k in ("train/global_step", "global_step", "_step"):
        v = row.get(k)
        if _is_number(v):
            return float(v)
    return None


def _series_with_steps(history, key: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for row in history:
        step = _step_from_row(row)
        v = row.get(key)
        if step is not None and _is_number(v):
            out.append((step, float(v)))
    return out


def _rolling_mean(values: List[float], window: int) -> float | None:
    if len(values) < window or window <= 0:
        return None
    tail = values[-window:]
    return sum(tail) / len(tail)


def _window_trend(values: List[float], window: int) -> float | None:
    # Difference between latest rolling window and previous rolling window.
    if len(values) < 2 * window or window <= 0:
        return None
    prev = values[-2 * window : -window]
    curr = values[-window:]
    return (sum(curr) / len(curr)) - (sum(prev) / len(prev))


def _fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def _tail(values: List[float], window: int) -> List[float]:
    if window <= 0:
        return []
    if len(values) <= window:
        return values[:]
    return values[-window:]


def _fraction(values: List[float], pred) -> float | None:
    if not values:
        return None
    return sum(1 for v in values if pred(v)) / len(values)


def _range_mean(pairs: List[Tuple[float, float]], start: float, end: float) -> float | None:
    vals = [v for s, v in pairs if start <= s <= end]
    if not vals:
        return None
    return sum(vals) / len(vals)


def evaluate(
    run_path: str, window: int, baseline_start: float, baseline_end: float
) -> Tuple[bool, bool, Dict[str, float | None], Dict[str, int], List[str], List[str]]:
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "wandb is required for this script. Install dependencies and run from the project environment."
        ) from e

    api = wandb.Api()
    run = api.run(run_path)

    history = list(run.scan_history())
    if not history:
        raise RuntimeError("No history rows found for this run.")

    keys = {
        "exec_mean": "reward/execution_mean",
        "zero_frac": "exec/zero_fraction",
        "infra_zero_frac": "exec/failed_to_run_fraction",
        "model_zero_frac": "exec/wrong_solution_fraction",
        "valid_frac": "gen/valid_code_fraction",
        "fallback_frac": "judge/fallback_fraction",
        "retry_count": "judge/retry_count",
        "rate_limit_count": "judge/rate_limit_count",
        "judge_time_s": "timing/reward_judge_s",
        "reward_std": "grpo/reward_std_mean",
        "all_zero_frac": "grpo/all_zero_fraction",
    }

    series = {name: _series(history, k) for name, k in keys.items()}
    # KL key names vary across trainer/library versions.
    series["kl"] = _series_any(history, ["train/kl", "kl", "objective/kl"])
    exec_pairs = _series_with_steps(history, keys["exec_mean"])

    stats = {name: _rolling_mean(vals, window) for name, vals in series.items()}
    counts = {name: len(vals) for name, vals in series.items()}
    exec_trend = _window_trend(series["exec_mean"], window)
    zero_trend = _window_trend(series["zero_frac"], window)
    fallback_trend = _window_trend(series["fallback_frac"], window)
    retry_trend = _window_trend(series["retry_count"], window)
    rate_limit_trend = _window_trend(series["rate_limit_count"], window)
    judge_time_trend = _window_trend(series["judge_time_s"], window)
    kl_vals = series["kl"]

    exec_tail = _tail(series["exec_mean"], window)
    all_zero_tail = _tail(series["all_zero_frac"], window)
    retry_tail = _tail(series["retry_count"], window)
    rate_tail = _tail(series["rate_limit_count"], window)
    judge_time_tail = _tail(series["judge_time_s"], window)

    exec_low_share = _fraction(exec_tail, lambda v: v < 0.30)
    all_zero_spike_share = _fraction(all_zero_tail, lambda v: v > 0.0)
    retry_nonzero_share = _fraction(retry_tail, lambda v: v > 0.0)
    rate_nonzero_share = _fraction(rate_tail, lambda v: v > 0.0)

    # Judge timing instability: repeated spikes in the latest window.
    judge_spike_threshold = 40.0
    if len(series["judge_time_s"]) >= 2 * window and window > 0:
        prev_judge = series["judge_time_s"][-2 * window : -window]
        if prev_judge:
            prev_mean = sum(prev_judge) / len(prev_judge)
            judge_spike_threshold = max(40.0, 2.5 * prev_mean)
    judge_spike_count = sum(1 for v in judge_time_tail if v > judge_spike_threshold)
    judge_unstable = judge_spike_count >= 3

    baseline_exec = _range_mean(exec_pairs, baseline_start, baseline_end)
    exec_gain_vs_baseline = (
        (stats["exec_mean"] - baseline_exec)
        if (stats["exec_mean"] is not None and baseline_exec is not None)
        else None
    )

    continue_rules = []
    stop_rules = []
    limited_data = False

    # Continue rules
    if stats["exec_mean"] is not None and exec_trend is not None:
        continue_rules.append(stats["exec_mean"] >= 0.30 and exec_trend >= -0.01)
    else:
        limited_data = True

    if exec_gain_vs_baseline is not None:
        continue_rules.append(exec_gain_vs_baseline >= 0.05)
    else:
        limited_data = True

    if stats["zero_frac"] is not None:
        continue_rules.append(stats["zero_frac"] <= 0.55)
    else:
        limited_data = True
    if stats["infra_zero_frac"] is not None:
        continue_rules.append(stats["infra_zero_frac"] <= 0.25)
    else:
        limited_data = True
    if stats["valid_frac"] is not None:
        continue_rules.append(stats["valid_frac"] >= 0.75)
    else:
        limited_data = True
    if stats["fallback_frac"] is not None:
        continue_rules.append(stats["fallback_frac"] <= 0.20)
    else:
        limited_data = True
    if stats["retry_count"] is not None and retry_nonzero_share is not None:
        continue_rules.append(stats["retry_count"] <= 0.30 and retry_nonzero_share <= 0.20)
    else:
        limited_data = True
    if stats["rate_limit_count"] is not None and rate_nonzero_share is not None:
        continue_rules.append(stats["rate_limit_count"] <= 0.30 and rate_nonzero_share <= 0.20)
    else:
        limited_data = True
    if stats["judge_time_s"] is not None:
        continue_rules.append((not judge_unstable) and stats["judge_time_s"] <= 20.0)
    else:
        limited_data = True
    if stats["reward_std"] is not None:
        continue_rules.append(stats["reward_std"] >= 0.15)
    else:
        limited_data = True
    if stats["all_zero_frac"] is not None:
        continue_rules.append(stats["all_zero_frac"] <= 0.10)
    else:
        limited_data = True

    # KL "stable" check: latest window mean should not be >3x previous window mean.
    kl_stable = False
    if len(kl_vals) >= 2 * window:
        prev = sum(kl_vals[-2 * window : -window]) / window
        curr = sum(kl_vals[-window:]) / window
        kl_stable = curr <= (3.0 * max(prev, 1e-9))
    if len(kl_vals) >= 2 * window:
        continue_rules.append(kl_stable)
    else:
        limited_data = True

    # Stop/Tune triggers (persistent behavior approximated via rolling-window stats).
    stop_rules.append(stats["infra_zero_frac"] is not None and stats["infra_zero_frac"] > 0.30)
    stop_rules.append(
        stats["exec_mean"] is not None
        and stats["exec_mean"] < 0.30
        and exec_low_share is not None
        and exec_low_share >= 0.70
    )
    stop_rules.append(
        stats["fallback_frac"] is not None
        and stats["fallback_frac"] > 0.30
        and stats["retry_count"] is not None
        and stats["rate_limit_count"] is not None
        and (retry_trend is not None and retry_trend > 0.0)
        and (rate_limit_trend is not None and rate_limit_trend > 0.0)
    )
    stop_rules.append(
        stats["judge_time_s"] is not None
        and (judge_unstable or stats["judge_time_s"] > 25.0)
    )
    stop_rules.append(stats["reward_std"] is not None and stats["reward_std"] < 0.10)
    stop_rules.append(
        stats["all_zero_frac"] is not None
        and (
            stats["all_zero_frac"] > 0.20
            or (all_zero_spike_share is not None and all_zero_spike_share > 0.20)
        )
    )

    reasons_ok = []
    reasons_bad = []

    reasons_ok.append(f"window={window}")
    reasons_ok.append("sample_counts=" + ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)))
    reasons_ok.append(
        f"baseline_exec_mean[{int(baseline_start)}-{int(baseline_end)}]={_fmt(baseline_exec)} "
        f"gain={_fmt(exec_gain_vs_baseline)}"
    )
    reasons_ok.append(f"execution_mean={_fmt(stats['exec_mean'])} trend={_fmt(exec_trend)}")
    reasons_ok.append(f"zero_fraction={_fmt(stats['zero_frac'])} trend={_fmt(zero_trend)}")
    reasons_ok.append(f"failed_to_run_fraction={_fmt(stats['infra_zero_frac'])}")
    reasons_ok.append(f"wrong_solution_fraction={_fmt(stats['model_zero_frac'])}")
    reasons_ok.append(f"valid_code_fraction={_fmt(stats['valid_frac'])}")
    reasons_ok.append(f"fallback_fraction={_fmt(stats['fallback_frac'])} trend={_fmt(fallback_trend)}")
    reasons_ok.append(
        f"retry_count={_fmt(stats['retry_count'])} trend={_fmt(retry_trend)} nonzero_share={_fmt(retry_nonzero_share)}"
    )
    reasons_ok.append(
        f"rate_limit_count={_fmt(stats['rate_limit_count'])} trend={_fmt(rate_limit_trend)} nonzero_share={_fmt(rate_nonzero_share)}"
    )
    reasons_ok.append(
        f"reward_judge_s={_fmt(stats['judge_time_s'])} trend={_fmt(judge_time_trend)} spikes={judge_spike_count}"
    )
    reasons_ok.append(f"reward_std_mean={_fmt(stats['reward_std'])}")
    reasons_ok.append(
        f"all_zero_fraction={_fmt(stats['all_zero_frac'])} spike_share={_fmt(all_zero_spike_share)}"
    )
    reasons_ok.append(f"kl_mean={_fmt(stats['kl'])}")

    if stats["infra_zero_frac"] is not None and stats["infra_zero_frac"] > 0.30:
        reasons_bad.append("failed_to_run_fraction too high (>0.30)")
    if (
        stats["exec_mean"] is not None
        and stats["exec_mean"] < 0.30
        and exec_low_share is not None
        and exec_low_share >= 0.70
    ):
        reasons_bad.append("execution_mean mostly low (<0.30 for >=70% of recent window)")
    if (
        stats["fallback_frac"] is not None
        and stats["fallback_frac"] > 0.30
        and retry_trend is not None and retry_trend > 0.0
        and rate_limit_trend is not None and rate_limit_trend > 0.0
    ):
        reasons_bad.append("fallback high with rising retry/rate-limit pressure")
    if stats["judge_time_s"] is not None and (judge_unstable or stats["judge_time_s"] > 25.0):
        reasons_bad.append("reward_judge_s unstable (repeated spikes) or too high")
    if stats["reward_std"] is not None and stats["reward_std"] < 0.10:
        reasons_bad.append("reward_std_mean too low (<0.10)")
    if (
        stats["all_zero_frac"] is not None
        and (
            stats["all_zero_frac"] > 0.20
            or (all_zero_spike_share is not None and all_zero_spike_share > 0.20)
        )
    ):
        reasons_bad.append("all_zero_fraction persistently elevated")
    if len(kl_vals) >= 2 * window and not kl_stable:
        reasons_bad.append("KL is unstable or insufficient data to verify stability")

    hard_stop = any(stop_rules)
    should_continue = (not hard_stop) and all(continue_rules)
    return should_continue, limited_data, stats, counts, reasons_ok, reasons_bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", required=True, help="W&B run id")
    parser.add_argument("--window", type=int, default=50, help="Rolling window size (default: 50)")
    parser.add_argument(
        "--baseline-start",
        type=float,
        default=700.0,
        help="Baseline start step for execution_mean comparison (default: 700).",
    )
    parser.add_argument(
        "--baseline-end",
        type=float,
        default=750.0,
        help="Baseline end step for execution_mean comparison (default: 750).",
    )
    args = parser.parse_args()

    run_path = f"{args.entity}/{args.project}/{args.run}"
    should_continue, limited_data, _, _, reasons_ok, reasons_bad = evaluate(
        run_path=run_path,
        window=args.window,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )

    print(f"Run: {run_path}")
    print("Metrics (rolling window):")
    for line in reasons_ok:
        print(f"  - {line}")

    print("\nDecision:")
    if should_continue:
        print("  CONTINUE")
        print("  Reason: metrics are inside the healthy range.")
    else:
        if reasons_bad:
            print("  STOP")
            print("  Reasons:")
            for r in reasons_bad:
                print(f"  - {r}")
        else:
            if limited_data:
                print("  CONTINUE (limited trend data)")
                print("  Reason: no hard-stop conditions triggered, but some trend checks need more points.")
            else:
                print("  STOP")
                print("  Reasons: some continue checks failed; review run manually.")


if __name__ == "__main__":
    main()
