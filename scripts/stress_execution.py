"""
Stress test for reward.execution.score_batch.

Runs repeated batch scoring on real APPS passing solutions, with optional
mixed noisy code (wrong output, syntax error, infinite loop).

Usage:
    python scripts/stress_execution.py
    python scripts/stress_execution.py --rounds-clean 100 --rounds-mixed 30
    python scripts/stress_execution.py --workers 16 --batch-size 32
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reward.execution import score_batch


def load_apps(dataset_size: int) -> list[tuple[str, dict]]:
    apps: list[tuple[str, dict]] = []
    with open("data/clean/apps_clean.jsonl") as f:
        for line in f:
            p = json.loads(line)
            sol = p.get("passing_solution")
            if not sol:
                continue
            io = p.get("io", {})
            tc = [{"input": i, "output": o} for i, o in zip(io.get("inputs", []), io.get("outputs", []))]
            if not tc:
                continue
            q = dict(p)
            q["test_cases"] = tc
            q["stdin_tests"] = tc
            q["is_leetcode"] = False
            apps.append((sol, q))
            if len(apps) >= dataset_size:
                break
    return apps


def summarize(name: str, times: list[float], means: list[float], zeros: list[int], timeouts: list[int], total_exec: int):
    if not times:
        print(f"\n=== {name} SUMMARY ===")
        print("No rounds run.")
        return
    p95 = statistics.quantiles(times, n=20)[-1] if len(times) >= 20 else max(times)
    print(f"\n=== {name} SUMMARY ===")
    print(f"rounds={len(times)} total_executions={total_exec}")
    print(f"time_per_round_mean={statistics.mean(times):.3f}s")
    print(f"time_per_round_p95={p95:.3f}s")
    print(f"time_per_round_max={max(times):.3f}s")
    print(f"mean_score_avg={statistics.mean(means):.4f}")
    print(f"mean_score_min={min(means):.4f}")
    print(f"zero_total={sum(zeros)} timeout_total={sum(timeouts)}")
    print(f"throughput_exec_per_sec={total_exec / sum(times):.2f}")


def run_clean(apps, rounds: int, batch_size: int, timeout: int, workers: int):
    times, means, zeros, timeouts = [], [], [], []
    idx = 0
    print(f"Clean test start: rounds={rounds}, batch_size={batch_size}")
    for r in range(rounds):
        chunk = []
        for _ in range(batch_size):
            chunk.append(apps[idx % len(apps)])
            idx += 1
        codes = [c for c, _ in chunk]
        probs = [p for _, p in chunk]

        t0 = time.time()
        _, stats = score_batch(codes=codes, problems=probs, timeout=timeout, n_workers=workers)
        dt = time.time() - t0

        times.append(dt)
        means.append(stats["mean_score"])
        zeros.append(stats["zero_scores"])
        timeouts.append(stats["timeout_count"])

        if (r + 1) % 10 == 0 or r == rounds - 1:
            print(
                f"C round {r+1:03d}: t={dt:.3f}s "
                f"mean={stats['mean_score']:.3f} zeros={stats['zero_scores']} to={stats['timeout_count']}"
            )
    summarize("CLEAN", times, means, zeros, timeouts, rounds * batch_size)


def run_mixed(
    apps,
    rounds: int,
    batch_size: int,
    timeout: int,
    workers: int,
    p_good: float,
    p_wrong: float,
    p_syntax: float,
):
    p_loop = 1.0 - (p_good + p_wrong + p_syntax)
    if p_loop < 0:
        raise ValueError("Probabilities must sum to <= 1.0")

    times, means, zeros, timeouts = [], [], [], []
    idx = 0
    print(
        f"Mixed test start: rounds={rounds}, batch_size={batch_size}, "
        f"mix=good:{p_good:.2f} wrong:{p_wrong:.2f} syntax:{p_syntax:.2f} loop:{p_loop:.2f}"
    )
    for r in range(rounds):
        codes, probs = [], []
        for _ in range(batch_size):
            sol, p = apps[idx % len(apps)]
            idx += 1
            x = random.random()
            if x < p_good:
                code = sol
            elif x < p_good + p_wrong:
                code = 'print("definitely_wrong_output_123")'
            elif x < p_good + p_wrong + p_syntax:
                code = "def bad(:\n    pass"
            else:
                code = "while True:\n    pass"
            codes.append(code)
            probs.append(p)

        t0 = time.time()
        _, stats = score_batch(codes=codes, problems=probs, timeout=timeout, n_workers=workers)
        dt = time.time() - t0

        times.append(dt)
        means.append(stats["mean_score"])
        zeros.append(stats["zero_scores"])
        timeouts.append(stats["timeout_count"])

        if (r + 1) % 5 == 0 or r == rounds - 1:
            print(
                f"M round {r+1:03d}: t={dt:.3f}s "
                f"mean={stats['mean_score']:.3f} zeros={stats['zero_scores']} to={stats['timeout_count']}"
            )
    summarize("MIXED", times, means, zeros, timeouts, rounds * batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--rounds-clean", type=int, default=20)
    parser.add_argument("--rounds-mixed", type=int, default=10)
    parser.add_argument("--p-good", type=float, default=0.70)
    parser.add_argument("--p-wrong", type=float, default=0.10)
    parser.add_argument("--p-syntax", type=float, default=0.10)
    args = parser.parse_args()

    random.seed(args.seed)
    apps = load_apps(args.dataset_size)
    if len(apps) < args.batch_size:
        raise SystemExit(
            f"Not enough validated APPS solutions: loaded={len(apps)} batch_size={args.batch_size}"
        )
    print(f"Loaded {len(apps)} validated APPS problems with passing solutions.")

    if args.rounds_clean > 0:
        run_clean(apps, args.rounds_clean, args.batch_size, args.timeout, args.workers)
    if args.rounds_mixed > 0:
        run_mixed(
            apps,
            args.rounds_mixed,
            args.batch_size,
            args.timeout,
            args.workers,
            args.p_good,
            args.p_wrong,
            args.p_syntax,
        )


if __name__ == "__main__":
    main()
