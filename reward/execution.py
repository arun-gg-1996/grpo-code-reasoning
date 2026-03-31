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
import multiprocessing.pool
import os
import sys
from typing import Optional
from config import EXEC_TIMEOUT as SUBPROCESS_TIMEOUT, EXEC_WORKERS as POOL_WORKERS, MAX_TEST_CASES

sys.set_int_max_str_digits(100000)

# ─────────────────────────────────────────
# Persistent batch pool
# Created once on first score_batch call, reused for all subsequent calls.
# Avoids spawning POOL_WORKERS new processes per reward step — without this,
# pool creation/teardown alone costs ~3s per step × 2000 steps ≈ 1.5 hours.
# ─────────────────────────────────────────
_batch_pool: Optional["multiprocessing.pool.Pool"] = None


def _get_batch_pool(n_workers: int) -> multiprocessing.pool.Pool:
    """Return the module-level persistent spawn pool, creating it on first call."""
    global _batch_pool
    if _batch_pool is None:
        ctx = multiprocessing.get_context("spawn")
        _batch_pool = ctx.Pool(processes=n_workers)
    return _batch_pool


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


def _parse_functional_input(input_str: str) -> list:
    """
    Parse a functional test input string into a list of Python arguments.
    Handles two formats:
      - Single-line: '["a", "b"]'   → [["a", "b"]]   (one list argument)
      - Multi-line:  '[1,2]\n[3,4]' → [[1,2], [3,4]] (two arguments)
    """
    import ast, json as _json
    lines = [l.strip() for l in input_str.strip().split("\n") if l.strip()]
    args = []
    for line in lines:
        try:
            args.append(_json.loads(line))
        except Exception:
            try:
                args.append(ast.literal_eval(line))
            except Exception:
                args.append(line)
    return args


def _parse_functional_output(output_str: str):
    """Parse a functional test output string into a Python object."""
    import ast, json as _json
    s = output_str.strip()
    try:
        return _json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return s


def _functional_worker(solution: str, func_name: str, test_cases: list, result_queue):
    sys.set_int_max_str_digits(100000)
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from sandbox.testing_util import run_test

        io = {
            "inputs": [_parse_functional_input(tc["input"]) for tc in test_cases],
            "outputs": [_parse_functional_output(tc["output"]) for tc in test_cases],
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
    """Returns (score, status) where status is 'ok', 'timeout', 'error', or 'empty'."""
    code, problem, timeout = args

    if not code or not code.strip():
        return 0.0, "empty"

    test_cases = problem.get("test_cases", [])
    if not test_cases:
        return 0.0, "empty"

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
        return float(result), "ok"
    elif status == "timeout":
        return 0.0, "timeout"
    else:
        return 0.0, "error"


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
        same length as codes, stats has mean/zero/perfect/timeout counts
    """
    assert len(codes) == len(problems), "codes and problems must have same length"

    args = [(code, problem, timeout) for code, problem in zip(codes, problems)]

    # Use persistent pool — created once, reused across all training steps
    global _batch_pool
    try:
        pool = _get_batch_pool(n_workers)
        results = pool.map(_pool_worker_wrapper, args)
    except Exception:
        # Pool may have died (worker crash etc); recreate once and retry
        _batch_pool = None
        pool = _get_batch_pool(n_workers)
        results = pool.map(_pool_worker_wrapper, args)

    scores = [r[0] for r in results]
    statuses = [r[1] for r in results]
    n = len(scores)

    stats = {
        "mean_score": sum(scores) / n,
        "zero_scores": sum(1 for s in scores if s == 0.0),
        "perfect_scores": sum(1 for s in scores if s == 1.0),
        "timeout_count": sum(1 for s in statuses if s == "timeout"),
        "timeout_fraction": sum(1 for s in statuses if s == "timeout") / n,
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
    # fallback: ```python blocks
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: any ``` block
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
