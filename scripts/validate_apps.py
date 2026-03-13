"""
validate_apps.py

Validates APPS introductory and competition problems using the official
Hendrycks testing_util.run_test harness, which handles both:
  - stdin/stdout problems  (fn_name absent)
  - function-call problems (fn_name present)

IMPORTANT: testing_util.reliability_guard() permanently nullifies os.kill,
subprocess.Popen, etc in whatever process calls it. So every run_test call
must happen in an isolated subprocess via multiprocessing.

Usage:
    python scripts/validate_apps.py
"""

import json
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict

# must be set before any json.loads on APPS data
sys.set_int_max_str_digits(100000)

from datasets import load_dataset

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

OUTPUT_PATH        = "../data/clean/apps_clean.jsonl"
FAILED_DIR         = "../data/failed/apps"
FAILED_SAMPLE_SIZE = 20

VALIDATION_THRESHOLD = 0.9
MAX_TESTS            = 10
MAX_SOLUTIONS_TO_TRY = 5
SUBPROCESS_TIMEOUT   = 15   # seconds per solution attempt

DIFFICULTY_CONFIG = {
    "introductory": {"weight": 1.0, "min_tests": 1},
    "competition":  {"weight": 2.0, "min_tests": 1},
}

# ─────────────────────────────────────────
# Subprocess worker
# ─────────────────────────────────────────

def _run_test_worker(solution: str, capped_io: dict, result_queue):
    """
    Worker run in an isolated subprocess.

    testing_util.reliability_guard() permanently nullifies os.kill,
    subprocess.Popen, signal handlers, etc in the calling process.
    Running in a subprocess prevents this from poisoning the parent.
    """
    # add project root to path so sandbox package is importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.set_int_max_str_digits(100000)

    try:
        from sandbox.testing_util import run_test
        problem_dict = {"input_output": capped_io}
        results      = run_test(problem=problem_dict, test=solution)
        passed       = sum(1 for r in results if r is True)
        score        = passed / len(results) if results else 0.0
        result_queue.put(score)
    except Exception:
        result_queue.put(0.0)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def parse_io(problem: dict) -> dict | None:
    """Parse input_output field. Returns dict or None if unusable."""
    try:
        io = json.loads(problem.get("input_output", "{}"))
        if not io.get("inputs"):
            return None
        return io
    except (json.JSONDecodeError, ValueError):
        return None


def parse_solutions(problem: dict) -> list:
    """Parse solutions field into list of strings."""
    try:
        sols = json.loads(problem.get("solutions", "[]"))
        return [s for s in sols if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


def score_solution(solution: str, io: dict, max_tests: int) -> float:
    """
    Run a single solution in an isolated subprocess.
    Returns 0.0-1.0 score (fraction of test cases passed).
    """
    capped_io = {
        "inputs":  io["inputs"][:max_tests],
        "outputs": io["outputs"][:max_tests],
    }
    if io.get("fn_name"):
        capped_io["fn_name"] = io["fn_name"]

    ctx   = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc  = ctx.Process(target=_run_test_worker, args=(solution, capped_io, queue))
    proc.start()
    proc.join(timeout=SUBPROCESS_TIMEOUT)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return 0.0

    return queue.get() if not queue.empty() else 0.0


def validate_problem(problem: dict, difficulty: str) -> tuple:
    """
    Validate a single APPS problem.
    Returns (clean_dict, failure_reason) — one is always None.
    failure_reason: "no_io" | "no_tests" | "no_solutions" | "score"
    """
    weight    = DIFFICULTY_CONFIG[difficulty]["weight"]
    min_tests = DIFFICULTY_CONFIG[difficulty]["min_tests"]

    io = parse_io(problem)
    if io is None:
        return None, "no_io"

    if len(io["inputs"]) < min_tests:
        return None, "no_tests"

    solutions = parse_solutions(problem)
    if not solutions:
        return None, "no_solutions"

    best_score       = 0.0
    passing_solution = None

    for sol in solutions[:MAX_SOLUTIONS_TO_TRY]:
        score = score_solution(sol, io, MAX_TESTS)
        if score > best_score:
            best_score       = score
            passing_solution = sol
        if score >= VALIDATION_THRESHOLD:
            break

    if best_score < VALIDATION_THRESHOLD:
        return None, "score"

    return {
        "problem_id":       problem.get("problem_id", ""),
        "question":         problem.get("question", ""),
        "difficulty":       difficulty,
        "sampling_weight":  weight,
        "fn_name":          io.get("fn_name"),      # None = stdin/stdout
        "io": {
            "inputs":  io["inputs"][:MAX_TESTS],
            "outputs": io["outputs"][:MAX_TESTS],
            **({"fn_name": io["fn_name"]} if io.get("fn_name") else {}),
        },
        "solutions":        solutions,
        "passing_solution": passing_solution,
        "validation_score": round(best_score, 4),
        "source":           "apps",
    }, None


def save_failed_sample(failed: list, difficulty: str, reason: str):
    if not failed:
        return
    os.makedirs(FAILED_DIR, exist_ok=True)
    sample = random.sample(failed, min(FAILED_SAMPLE_SIZE, len(failed)))
    path   = os.path.join(FAILED_DIR, f"{difficulty}_{reason}_failed.jsonl")
    with open(path, "w") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")
    print(f"    Saved {len(sample)} samples → {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("APPS DATASET VALIDATION (using testing_util.run_test)")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("\nLoading APPS dataset...")
    ds = load_dataset("codeparrot/apps", split="train")
    print(f"Total problems loaded: {len(ds)}")

    by_difficulty = defaultdict(list)
    for problem in ds:
        diff = problem.get("difficulty", "")
        if diff in DIFFICULTY_CONFIG:
            by_difficulty[diff].append(problem)

    for diff, problems in by_difficulty.items():
        print(f"  {diff}: {len(problems)} problems")

    all_clean = []
    stats     = {}

    for difficulty, problems in by_difficulty.items():
        print(f"\n{'─' * 60}")
        print(f"Validating {difficulty} ({len(problems)} problems)...")
        print(f"  weight={DIFFICULTY_CONFIG[difficulty]['weight']}, "
              f"min_tests={DIFFICULTY_CONFIG[difficulty]['min_tests']}")
        print(f"{'─' * 60}")

        clean             = []
        failed_score_list = []
        dropped_no_io     = 0
        dropped_no_tests  = 0
        dropped_no_sols   = 0
        dropped_score     = 0
        start_time        = time.time()

        for i, problem in enumerate(problems):
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate    = i / elapsed
                eta     = (len(problems) - i) / rate
                print(f"  [{i}/{len(problems)}] "
                      f"clean={len(clean)} "
                      f"dropped={dropped_no_io+dropped_no_tests+dropped_no_sols+dropped_score} "
                      f"ETA={eta:.0f}s")

            result, reason = validate_problem(problem, difficulty)

            if result is None:
                if reason == "no_io":
                    dropped_no_io += 1
                elif reason == "no_tests":
                    dropped_no_tests += 1
                elif reason == "no_solutions":
                    dropped_no_sols += 1
                elif reason == "score":
                    dropped_score += 1
                    sols = parse_solutions(problem)
                    failed_score_list.append({
                        "problem_id":     problem.get("problem_id", ""),
                        "question":       problem.get("question", "")[:400],
                        "fn_name":        parse_io(problem) and parse_io(problem).get("fn_name"),
                        "first_solution": sols[0][:600] if sols else "",
                        "num_solutions":  len(sols),
                    })
                continue

            clean.append(result)

        elapsed = time.time() - start_time

        print(f"\n  Saving failed samples...")
        save_failed_sample(failed_score_list, difficulty, "score")

        stats[difficulty] = {
            "total":            len(problems),
            "clean":            len(clean),
            "dropped_no_io":    dropped_no_io,
            "dropped_no_tests": dropped_no_tests,
            "dropped_no_sols":  dropped_no_sols,
            "dropped_score":    dropped_score,
            "time_seconds":     round(elapsed, 1),
        }

        fn_count    = sum(1 for p in clean if p.get("fn_name"))
        stdin_count = sum(1 for p in clean if not p.get("fn_name"))

        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Clean:              {len(clean)} ({len(clean)/len(problems)*100:.1f}%)")
        print(f"    stdin/stdout:     {stdin_count}")
        print(f"    function-call:    {fn_count}")
        print(f"  Dropped (no io):    {dropped_no_io}")
        print(f"  Dropped (no tests): {dropped_no_tests}")
        print(f"  Dropped (no sols):  {dropped_no_sols}")
        print(f"  Dropped (score):    {dropped_score}")

        all_clean.extend(clean)

    # save
    print(f"\n{'=' * 60}")
    print(f"SAVING CLEAN DATASET → {OUTPUT_PATH}")
    print(f"{'=' * 60}")

    with open(OUTPUT_PATH, "w") as f:
        for problem in all_clean:
            f.write(json.dumps(problem) + "\n")

    # summary
    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 60}")
    total_weighted = 0
    for diff, s in stats.items():
        weight   = DIFFICULTY_CONFIG[diff]["weight"]
        weighted = s["clean"] * weight
        total_weighted += weighted
        print(f"\n  {diff}:")
        print(f"    Total:    {s['total']}")
        print(f"    Clean:    {s['clean']} ({s['clean']/s['total']*100:.1f}%)")
        print(f"    Weighted: {weighted:.0f} (×{weight})")
        print(f"    Time:     {s['time_seconds']}s")

    print(f"\n  Total clean problems:   {len(all_clean)}")
    print(f"  Total weighted:         {total_weighted:.0f}")
    print(f"  Failed samples:         {FAILED_DIR}/")
    print(f"\n  Threshold: {VALIDATION_THRESHOLD}")
    print(f"  Ready for training: {'YES ✓' if len(all_clean) > 1500 else 'LOW — inspect failures'}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()