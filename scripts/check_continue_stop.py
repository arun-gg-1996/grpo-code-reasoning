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

import wandb


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def _series(history, key: str) -> List[float]:
    vals = []
    for row in history:
        v = row.get(key)
        if _is_number(v):
            vals.append(float(v))
    return vals


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


def evaluate(
    run_path: str, window: int
) -> Tuple[bool, bool, Dict[str, float | None], Dict[str, int], List[str], List[str]]:
    api = wandb.Api()
    run = api.run(run_path)

    history = list(run.scan_history())
    if not history:
        raise RuntimeError("No history rows found for this run.")

    keys = {
        "exec_mean": "reward/execution_mean",
        "zero_frac": "exec/zero_fraction",
        "valid_frac": "gen/valid_code_fraction",
        "trunc_frac": "gen/likely_truncated_fraction",
        "timeout_frac": "exec/timeout_fraction",
        "reward_std": "grpo/reward_std_mean",
        "kl": "train/kl",
    }

    series = {name: _series(history, k) for name, k in keys.items()}
    stats = {name: _rolling_mean(vals, window) for name, vals in series.items()}
    counts = {name: len(vals) for name, vals in series.items()}
    exec_trend = _window_trend(series["exec_mean"], window)
    zero_trend = _window_trend(series["zero_frac"], window)
    kl_vals = series["kl"]

    continue_rules = []
    stop_rules = []
    limited_data = False

    # Continue rules
    if stats["exec_mean"] is not None and exec_trend is not None:
        continue_rules.append(stats["exec_mean"] >= 0.22 and exec_trend >= -0.02)
    else:
        limited_data = True

    if stats["zero_frac"] is not None:
        continue_rules.append(stats["zero_frac"] <= 0.70)
    else:
        limited_data = True
    if stats["valid_frac"] is not None:
        continue_rules.append(stats["valid_frac"] >= 0.75)
    else:
        limited_data = True
    if stats["trunc_frac"] is not None:
        continue_rules.append(stats["trunc_frac"] <= 0.08)
    else:
        limited_data = True
    if stats["timeout_frac"] is not None:
        continue_rules.append(stats["timeout_frac"] <= 0.08)
    else:
        limited_data = True
    if stats["reward_std"] is not None:
        continue_rules.append(stats["reward_std"] >= 0.08)
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

    # Stop rules
    stop_rules.append(stats["zero_frac"] is not None and stats["zero_frac"] > 0.85)
    stop_rules.append(stats["exec_mean"] is not None and stats["exec_mean"] < 0.15)
    stop_rules.append(stats["valid_frac"] is not None and stats["valid_frac"] < 0.60)
    stop_rules.append(stats["trunc_frac"] is not None and stats["trunc_frac"] > 0.20)
    stop_rules.append(stats["timeout_frac"] is not None and stats["timeout_frac"] > 0.15)
    stop_rules.append(stats["reward_std"] is not None and stats["reward_std"] < 0.03)

    reasons_ok = []
    reasons_bad = []

    reasons_ok.append(f"window={window}")
    reasons_ok.append("sample_counts=" + ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)))
    reasons_ok.append(f"execution_mean={_fmt(stats['exec_mean'])} trend={_fmt(exec_trend)}")
    reasons_ok.append(f"zero_fraction={_fmt(stats['zero_frac'])} trend={_fmt(zero_trend)}")
    reasons_ok.append(f"valid_code_fraction={_fmt(stats['valid_frac'])}")
    reasons_ok.append(f"likely_truncated_fraction={_fmt(stats['trunc_frac'])}")
    reasons_ok.append(f"timeout_fraction={_fmt(stats['timeout_frac'])}")
    reasons_ok.append(f"reward_std_mean={_fmt(stats['reward_std'])}")
    reasons_ok.append(f"kl_mean={_fmt(stats['kl'])}")

    if stats["zero_frac"] is not None and stats["zero_frac"] > 0.85:
        reasons_bad.append("zero_fraction too high (>0.85)")
    if stats["exec_mean"] is not None and stats["exec_mean"] < 0.15:
        reasons_bad.append("execution_mean too low (<0.15)")
    if stats["valid_frac"] is not None and stats["valid_frac"] < 0.60:
        reasons_bad.append("valid_code_fraction too low (<0.60)")
    if stats["trunc_frac"] is not None and stats["trunc_frac"] > 0.20:
        reasons_bad.append("likely_truncated_fraction too high (>0.20)")
    if stats["timeout_frac"] is not None and stats["timeout_frac"] > 0.15:
        reasons_bad.append("timeout_fraction too high (>0.15)")
    if stats["reward_std"] is not None and stats["reward_std"] < 0.03:
        reasons_bad.append("reward_std_mean collapsed (<0.03)")
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
    args = parser.parse_args()

    run_path = f"{args.entity}/{args.project}/{args.run}"
    should_continue, limited_data, _, _, reasons_ok, reasons_bad = evaluate(
        run_path=run_path, window=args.window
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
