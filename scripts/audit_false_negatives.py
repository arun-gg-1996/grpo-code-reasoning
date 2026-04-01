#!/usr/bin/env python3
"""
Audit possible false negatives in eval details.

This script re-runs selected low-scored completions and reports mismatches:
- logged score vs re-run score (same timeout)
- logged score vs relaxed-timeout score

Use this to sanity-check whether "good" code is being penalized by harness/timeouts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure repo root is on sys.path when running as `python scripts/...`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import LCB_EVAL_PATH, EXEC_TIMEOUT, normalize_difficulty
from reward.execution import score_single


def _normalize_test_cases(problem: dict) -> dict:
    if problem.get("test_cases"):
        return problem
    io = problem.get("io")
    if io and isinstance(io, dict):
        inputs = io.get("inputs", [])
        outputs = io.get("outputs", [])
        tc = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]
        problem["test_cases"] = tc
        problem["stdin_tests"] = tc
    return problem


def load_eval_problems(path: str) -> dict[str, dict]:
    problems: dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            p["difficulty"] = normalize_difficulty(p.get("difficulty", "medium"))
            p = _normalize_test_cases(p)
            pid = p.get("question_id") or p.get("problem_id")
            if pid is None:
                continue
            problems[str(pid)] = p
    return problems


def load_details(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=str, required=True, help="Path to details.jsonl")
    ap.add_argument("--eval-data", type=str, default=LCB_EVAL_PATH, help="Eval JSONL path")
    ap.add_argument("--max-cases", type=int, default=50, help="Max low-score rows to audit")
    ap.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.99,
        help="Audit rows with execution_score < threshold",
    )
    ap.add_argument(
        "--relaxed-timeout",
        type=int,
        default=20,
        help="Timeout seconds for second pass",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path for full audit report",
    )
    args = ap.parse_args()

    details_path = Path(args.details)
    rows = load_details(str(details_path))
    problems = load_eval_problems(args.eval_data)

    # Pick candidate rows: low score + non-empty code
    candidates = [
        r for r in rows
        if float(r.get("execution_score", 0.0)) < args.low_score_threshold
        and bool((r.get("model_output") or "").strip())
    ][: max(1, args.max_cases)]

    results = []
    missing_problem = 0
    for r in candidates:
        pid = str(r.get("question_id"))
        problem = problems.get(pid)
        if problem is None:
            missing_problem += 1
            continue

        code = r.get("model_output", "")
        logged = float(r.get("execution_score", 0.0))
        rerun_default = score_single(code, problem, timeout=EXEC_TIMEOUT)
        rerun_relaxed = score_single(code, problem, timeout=args.relaxed_timeout)

        results.append(
            {
                "question_id": pid,
                "difficulty": r.get("difficulty", "unknown"),
                "platform": r.get("platform", "unknown"),
                "completion_index": r.get("completion_index"),
                "logged_score": logged,
                "rerun_default": rerun_default,
                "rerun_relaxed": rerun_relaxed,
                "delta_default": rerun_default - logged,
                "delta_relaxed": rerun_relaxed - logged,
                "possible_timeout_sensitivity": (rerun_relaxed > rerun_default + 1e-9),
                "possible_false_negative": (logged < 0.99 and rerun_default >= 0.99),
            }
        )

    # Summaries
    n = len(results)
    false_neg_like = sum(1 for x in results if x["possible_false_negative"])
    timeout_sensitive = sum(1 for x in results if x["possible_timeout_sensitivity"])
    changed_default = sum(1 for x in results if abs(x["delta_default"]) > 1e-9)
    changed_relaxed = sum(1 for x in results if abs(x["delta_relaxed"]) > 1e-9)

    by_diff = defaultdict(int)
    for x in results:
        by_diff[x["difficulty"]] += 1

    print("=== FALSE-NEGATIVE AUDIT ===")
    print(f"details rows total: {len(rows)}")
    print(f"audited candidates: {n}")
    print(f"missing problem metadata: {missing_problem}")
    print(f"changed on rerun (default timeout): {changed_default}")
    print(f"changed on rerun (relaxed timeout): {changed_relaxed}")
    print(f"possible timeout-sensitive: {timeout_sensitive}")
    print(f"possible false negatives: {false_neg_like}")
    print(f"audited by difficulty: {dict(by_diff)}")

    suspicious = [
        x for x in results
        if x["possible_false_negative"] or x["possible_timeout_sensitivity"] or abs(x["delta_default"]) > 1e-9
    ]
    suspicious = sorted(
        suspicious,
        key=lambda x: (
            not x["possible_false_negative"],
            not x["possible_timeout_sensitivity"],
            abs(x["delta_relaxed"]),
        ),
        reverse=False,
    )

    print("\nTop suspicious cases:")
    for x in suspicious[:20]:
        print(
            f"qid={x['question_id']} diff={x['difficulty']} idx={x['completion_index']} "
            f"logged={x['logged_score']:.3f} rerun={x['rerun_default']:.3f} "
            f"relaxed={x['rerun_relaxed']:.3f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "details_rows_total": len(rows),
                "audited_candidates": n,
                "missing_problem_metadata": missing_problem,
                "changed_on_rerun_default": changed_default,
                "changed_on_rerun_relaxed": changed_relaxed,
                "possible_timeout_sensitive": timeout_sensitive,
                "possible_false_negatives": false_neg_like,
                "audited_by_difficulty": dict(by_diff),
            },
            "suspicious_cases": suspicious,
            "all_results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
