#!/usr/bin/env python3
"""
Analyze a W&B GRPO training run using full unsampled history.

Why this script:
- run.history() can be sampled and may hide spikes/trends.
- run.scan_history() iterates full logged rows.

Example:
  python scripts/analyze_wandb_run.py \
    --entity arun-gv-ghontale-suny-the-state-university-of-new-york \
    --project grpo-code-gen \
    --run 8x2y5wjg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pandas as pd
import wandb


METRICS = [
    "train/global_step",
    "reward/execution_mean",
    "exec/zero_fraction",
    "exec/infra_zero_fraction",
    "exec/model_zero_fraction",
    "exec/empty_fraction",
    "exec/timeout_fraction",
    "exec/error_fraction",
    "gen/valid_code_fraction",
    "gen/has_reasoning_fraction",
    "gen/likely_truncated_fraction",
    "gen/mean_completion_chars",
    "gen/mean_code_chars",
    "judge/fallback_fraction",
    "judge/retry_count",
    "judge/rate_limit_count",
    "timing/reward_judge_s",
    "data/easy_seen",
    "data/medium_seen",
    "data/hard_seen",
]


@dataclass
class Window:
    name: str
    start: int
    end: int


def _safe_mean(df: pd.DataFrame, key: str) -> float | None:
    if key not in df.columns:
        return None
    s = pd.to_numeric(df[key], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _safe_last(df: pd.DataFrame, key: str) -> float | None:
    if key not in df.columns:
        return None
    s = pd.to_numeric(df[key], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _get_step_col(df: pd.DataFrame) -> str:
    if "train/global_step" in df.columns and pd.to_numeric(df["train/global_step"], errors="coerce").notna().any():
        return "train/global_step"
    if "_step" in df.columns:
        return "_step"
    raise ValueError("No step column found (train/global_step or _step).")


def _window_df(df: pd.DataFrame, step_col: str, start: int, end: int) -> pd.DataFrame:
    s = pd.to_numeric(df[step_col], errors="coerce")
    return df[(s >= start) & (s <= end)]


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def fetch_history(entity: str, project: str, run_id: str, page_size: int = 2000) -> tuple[wandb.apis.public.Run, pd.DataFrame]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    rows: list[dict] = []
    for row in run.scan_history(keys=METRICS, page_size=page_size):
        rows.append(row)

    if not rows:
        raise RuntimeError("No history rows returned by W&B.")

    df = pd.DataFrame(rows)
    return run, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True, dest="run_id")
    ap.add_argument("--phase-a-start", type=int, default=600)
    ap.add_argument("--phase-a-end", type=int, default=800)
    ap.add_argument("--phase-b-start", type=int, default=801)
    ap.add_argument("--phase-b-end", type=int, default=960)
    args = ap.parse_args()

    run, df = fetch_history(args.entity, args.project, args.run_id)
    step_col = _get_step_col(df)
    df = df.sort_values(by=step_col).reset_index(drop=True)

    print(f"Run: {args.entity}/{args.project}/{args.run_id}")
    print(f"Name: {run.name}")
    print(f"URL: {run.url}")
    print(f"Rows fetched (unsampled): {len(df)}")
    print(f"Step column: {step_col}")
    print(f"Step range: {_fmt(_safe_last(df.head(1), step_col), 0)} -> {_fmt(_safe_last(df, step_col), 0)}")

    _print_section("Global Means")
    for key in [
        "reward/execution_mean",
        "exec/zero_fraction",
        "exec/infra_zero_fraction",
        "exec/model_zero_fraction",
        "exec/empty_fraction",
        "exec/timeout_fraction",
        "exec/error_fraction",
        "gen/valid_code_fraction",
        "gen/has_reasoning_fraction",
        "gen/likely_truncated_fraction",
        "judge/fallback_fraction",
        "judge/retry_count",
        "judge/rate_limit_count",
        "timing/reward_judge_s",
    ]:
        print(f"{key:28s} {_fmt(_safe_mean(df, key))}")

    _print_section("Coverage (latest logged value)")
    for key in ["data/easy_seen", "data/medium_seen", "data/hard_seen"]:
        print(f"{key:28s} {_fmt(_safe_last(df, key), 0)}")

    wa = Window("Phase A", args.phase_a_start, args.phase_a_end)
    wb = Window("Phase B", args.phase_b_start, args.phase_b_end)
    dfa = _window_df(df, step_col, wa.start, wa.end)
    dfb = _window_df(df, step_col, wb.start, wb.end)

    _print_section(f"Window Comparison: {wa.name} ({wa.start}-{wa.end}) vs {wb.name} ({wb.start}-{wb.end})")
    print(f"rows[{wa.name}]={len(dfa)} rows[{wb.name}]={len(dfb)}")

    compare_keys = [
        "reward/execution_mean",
        "exec/infra_zero_fraction",
        "exec/empty_fraction",
        "exec/timeout_fraction",
        "judge/fallback_fraction",
        "judge/retry_count",
        "judge/rate_limit_count",
        "timing/reward_judge_s",
    ]

    for key in compare_keys:
        a = _safe_mean(dfa, key)
        b = _safe_mean(dfb, key)
        delta = None if (a is None or b is None) else (b - a)
        dstr = "n/a" if delta is None else f"{delta:+.4f}"
        print(f"{key:28s} A={_fmt(a)}  B={_fmt(b)}  Δ={dstr}")

    _print_section("Notes")
    print("- Use this output for diagnosis; it is based on full scan_history (not sampled history).")
    print("- If coverage hard_seen stays 0 after phase introducing hard, inspect data difficulty distribution and normalization.")


if __name__ == "__main__":
    main()
