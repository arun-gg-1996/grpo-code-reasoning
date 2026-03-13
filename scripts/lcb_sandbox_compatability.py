"""
lcb_sandbox_compat_check.py

Sandbox compatibility check for all LCB seen problems.
Does NOT validate test case correctness — just confirms the sandbox
can execute code against every problem without crashing or hanging.

For each problem:
  - Run a trivially wrong solution ("print(0)" or "return 0")
  - Confirm sandbox returns a score (even 0.0) without crashing
  - Flag any problems where sandbox hangs, crashes, or errors

Usage:
    python scripts/lcb_sandbox_compat_check.py
"""

import json
import multiprocessing
import os
import sys
import time

sys.set_int_max_str_digits(100000)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LCB_PATH       = "../data/clean/lcb_seen_clean.jsonl"
FAILED_DIR     = "../data/failed/lcb"
SUBPROCESS_TIMEOUT = 15

# ─────────────────────────────────────────
# Trivial wrong solutions
# ─────────────────────────────────────────

STDIO_WRONG_SOLUTION = "print(0)"

def make_functional_wrong_solution(func_name: str, starter_code: str) -> str:
    """
    Build a trivially wrong solution for a LeetCode functional problem.
    Reads func_name from metadata and returns a dummy value.
    """
    if starter_code:
        # replace the function body with return None
        lines = starter_code.strip().split("\n")
        result = []
        in_func = False
        for line in lines:
            if line.strip().startswith("def "):
                in_func = True
                result.append(line)
                result.append("        return None")
            elif in_func and line.strip() and not line.startswith(" "):
                in_func = False
                result.append(line)
            elif not in_func:
                result.append(line)
        return "\n".join(result)
    # fallback
    return f"""
class Solution:
    def {func_name}(self, *args, **kwargs):
        return None
"""

# ─────────────────────────────────────────
# Subprocess workers
# ─────────────────────────────────────────

def _stdio_worker(solution: str, test_cases: list, result_queue):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.set_int_max_str_digits(100000)
    try:
        from sandbox.testing_util import run_test
        io = {
            "inputs":  [tc["input"] for tc in test_cases],
            "outputs": [tc["output"] for tc in test_cases],
        }
        results = run_test(problem={"input_output": io}, test=solution)
        score   = sum(1 for r in results if r is True) / len(results) if results else 0.0
        result_queue.put(("ok", score))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _functional_worker(solution: str, func_name: str, test_cases: list, result_queue):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.set_int_max_str_digits(100000)
    try:
        from sandbox.testing_util import run_test
        io = {
            "inputs":  [tc["input"] for tc in test_cases],
            "outputs": [tc["output"] for tc in test_cases],
            "fn_name": func_name,
        }
        results = run_test(problem={"input_output": io}, test=solution)
        score   = sum(1 for r in results if r is True) / len(results) if results else 0.0
        result_queue.put(("ok", score))
    except Exception as e:
        result_queue.put(("error", str(e)))


def run_in_subprocess(worker_fn, args, timeout) -> tuple:
    """
    Returns ("ok", score), ("error", msg), or ("timeout", None)
    """
    ctx   = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc  = ctx.Process(target=worker_fn, args=(*args, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return ("timeout", None)

    return queue.get() if not queue.empty() else ("error", "empty queue")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("LCB SANDBOX COMPATIBILITY CHECK")
    print("=" * 60)

    with open(LCB_PATH) as f:
        problems = [json.loads(l) for l in f]

    print(f"Loaded {len(problems)} LCB seen problems")
    print(f"Running trivially wrong solution through each sandbox...\n")

    ok_count      = 0
    error_count   = 0
    timeout_count = 0
    error_list    = []
    timeout_list  = []

    stdio_ok      = 0
    functional_ok = 0

    start_time = time.time()

    for i, problem in enumerate(problems):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_time
            eta     = (len(problems) - i) / (i / elapsed)
            print(f"  [{i}/{len(problems)}] ok={ok_count} errors={error_count} "
                  f"timeouts={timeout_count} ETA={eta:.0f}s")

        test_cases = problem.get("test_cases", [])
        if not test_cases:
            continue

        is_lc     = problem.get("is_leetcode", False)
        func_name = problem.get("func_name", "")

        if is_lc:
            starter = problem.get("starter_code", "")
            solution = make_functional_wrong_solution(func_name, starter)
            # functional test cases have JSON-string inputs
            func_tests = problem.get("functional_tests", test_cases[:3])
            status, result = run_in_subprocess(
                _functional_worker,
                (solution, func_name, func_tests[:3]),
                SUBPROCESS_TIMEOUT
            )
        else:
            solution = STDIO_WRONG_SOLUTION
            stdin_tests = problem.get("stdin_tests", test_cases[:3])
            status, result = run_in_subprocess(
                _stdio_worker,
                (solution, stdin_tests[:3]),
                SUBPROCESS_TIMEOUT
            )

        if status == "ok":
            ok_count += 1
            if is_lc:
                functional_ok += 1
            else:
                stdio_ok += 1
        elif status == "timeout":
            timeout_count += 1
            timeout_list.append({
                "question_id": problem.get("question_id"),
                "platform":    problem.get("platform"),
                "is_lc":       is_lc,
            })
        else:
            error_count += 1
            error_list.append({
                "question_id": problem.get("question_id"),
                "platform":    problem.get("platform"),
                "is_lc":       is_lc,
                "error":       str(result),
            })

    elapsed = time.time() - start_time

    # ── save failures ────────────────────────────────────────
    os.makedirs(FAILED_DIR, exist_ok=True)
    if error_list:
        path = os.path.join(FAILED_DIR, "compat_errors.jsonl")
        with open(path, "w") as f:
            for item in error_list:
                f.write(json.dumps(item) + "\n")
        print(f"\n  Saved {len(error_list)} errors → {path}")

    if timeout_list:
        path = os.path.join(FAILED_DIR, "compat_timeouts.jsonl")
        with open(path, "w") as f:
            for item in timeout_list:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(timeout_list)} timeouts → {path}")

    # ── summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total problems:    {len(problems)}")
    print(f"  OK:                {ok_count} ({ok_count/len(problems)*100:.1f}%)")
    print(f"    stdio OK:        {stdio_ok}")
    print(f"    functional OK:   {functional_ok}")
    print(f"  Errors:            {error_count}")
    print(f"  Timeouts:          {timeout_count}")
    print(f"  Time:              {elapsed:.1f}s")

    if error_count == 0 and timeout_count == 0:
        print(f"\n  Sandbox compatible with all LCB problems ✓")
    else:
        print(f"\n  Issues found — inspect data/failed/lcb/ for details")
        if error_count > 0:
            print(f"  First error: {error_list[0]}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()