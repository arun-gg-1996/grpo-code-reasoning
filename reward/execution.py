"""
reward/execution.py

Execution scoring for GRPO training.
Runs generated code against test cases in isolated subprocesses.
Returns a score 0.0 to 1.0 (fraction of test cases passed).

Usage:
    from reward.execution import score_batch
    scores = score_batch(problems, codes)  # list of floats
"""

import multiprocessing
import os
import sys
from typing import Optional
from config import EXEC_TIMEOUT as SUBPROCESS_TIMEOUT, EXEC_WORKERS as POOL_WORKERS, MAX_TEST_CASES

sys.set_int_max_str_digits(100000)


# ─────────────────────────────────────────
# Subprocess workers (run in fresh process)
# ─────────────────────────────────────────

def _stdio_worker(solution: str, test_cases: list, result_queue):
    sys.set_int_max_str_digits(100000)
    try:
        # local import — keeps reliability_guard damage inside this process
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from sandbox.testing_util import run_test

        io = {
            "inputs": [tc["input"] for tc in test_cases],
            "outputs": [tc["output"] for tc in test_cases],
        }
        results = run_test(problem={"input_output": io}, test=solution)
        score = sum(1 for r in results if r is True) / len(results) if results else 0.0
        result_queue.put(("ok", score))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _functional_worker(solution: str, func_name: str, test_cases: list, result_queue):
    sys.set_int_max_str_digits(100000)
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from sandbox.testing_util import run_test

        io = {
            "inputs": [tc["input"] for tc in test_cases],
            "outputs": [tc["output"] for tc in test_cases],
            "fn_name": func_name,
        }
        results = run_test(problem={"input_output": io}, test=solution)
        score = sum(1 for r in results if r is True) / len(results) if results else 0.0
        result_queue.put(("ok", score))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _run_subprocess(worker_fn, args: tuple, timeout: int) -> tuple:
    """
    Spawn a fresh subprocess, run worker_fn(*args, queue).
    Returns ("ok", score), ("error", msg), or ("timeout", None).
    """
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=worker_fn, args=(*args, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return ("timeout", None)

    return queue.get() if not queue.empty() else ("error", "empty queue")


# ─────────────────────────────────────────
# Per-response scoring
# ─────────────────────────────────────────

def score_single(
        code: str,
        problem: dict,
        timeout: int = SUBPROCESS_TIMEOUT,
) -> float:
    """
    Score a single generated solution against a problem's test cases.

    Args:
        code:     extracted code string from model response
        problem:  problem dict from JSONL (has test_cases, is_leetcode, func_name)
        timeout:  subprocess timeout in seconds

    Returns:
        float 0.0 to 1.0
    """
    if not code or not code.strip():
        return 0.0

    test_cases = problem.get("test_cases", [])
    if not test_cases:
        return 0.0

    # cap test cases
    test_cases = test_cases[:MAX_TEST_CASES]

    is_lc = problem.get("is_leetcode", False)
    func_name = problem.get("func_name", "")

    if is_lc and func_name:
        status, result = _run_subprocess(
            _functional_worker,
            (code, func_name, problem.get("functional_tests", test_cases)),
            timeout,
        )
    else:
        status, result = _run_subprocess(
            _stdio_worker,
            (code, problem.get("stdin_tests", test_cases)),
            timeout,
        )

    if status == "ok":
        return float(result)
    return 0.0  # error or timeout → 0.0


# ─────────────────────────────────────────
# Pool-based batch scoring
# ─────────────────────────────────────────

def _pool_worker_wrapper(args):
    code, problem, timeout = args
    return score_single(code, problem, timeout)


def score_batch(
        codes: list[str],
        problems: list[dict],
        timeout: int = SUBPROCESS_TIMEOUT,
        n_workers: int = POOL_WORKERS,
) -> tuple[list[float], dict]:
    """
    Score a batch of (code, problem) pairs in parallel.

    Args:
        codes:     list of extracted code strings (len = batch_size * G)
        problems:  list of corresponding problem dicts (same length)
        timeout:   subprocess timeout per worker
        n_workers: pool size

    Returns:
        tuple of (scores, stats) where scores is list of floats
        same length as codes, stats has mean/zero/perfect counts (to see if we should move to DAPO later)
    """
    assert len(codes) == len(problems), "codes and problems must have same length"

    args = [(code, problem, timeout) for code, problem in zip(codes, problems)]

    # use spawn context for pool to avoid nested fork issues
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        scores = pool.map(_pool_worker_wrapper, args)

    stats = {
        "mean_score": sum(scores) / len(scores),
        "zero_scores": sum(1 for s in scores if s == 0.0),
        "perfect_scores": sum(1 for s in scores if s == 1.0),
    }
    return scores, stats


# ─────────────────────────────────────────
# Code extraction
# ─────────────────────────────────────────

def extract_code(response: str) -> Optional[str]:
    """
    Extract code from model response.
    Expects <code>...</code> tags.
    Returns None if no code block found.
    """
    import re
    match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: try ```python blocks
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
