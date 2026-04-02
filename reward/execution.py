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
import io
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from config import EXEC_TIMEOUT as SUBPROCESS_TIMEOUT, EXEC_WORKERS as POOL_WORKERS, MAX_TEST_CASES
from problem_format import get_function_name, is_function_style_problem

sys.set_int_max_str_digits(100000)

# ─────────────────────────────────────────
# Persistent thread pool for batch scoring.
# Each thread launches an isolated subprocess via _run_subprocess.
# This avoids daemon process nesting errors from multiprocessing.Pool workers.
# ─────────────────────────────────────────
_batch_pool: Optional[ThreadPoolExecutor] = None
_batch_pool_workers: Optional[int] = None


def _get_batch_pool(n_workers: int) -> ThreadPoolExecutor:
    """Return the module-level persistent thread pool, creating it on first call."""
    global _batch_pool, _batch_pool_workers
    if _batch_pool is None or _batch_pool_workers != n_workers:
        if _batch_pool is not None:
            _batch_pool.shutdown(wait=True, cancel_futures=False)
        _batch_pool = ThreadPoolExecutor(max_workers=n_workers)
        _batch_pool_workers = n_workers
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

        io_payload = {
            "inputs": [tc["input"] for tc in test_cases],
            "outputs": [tc["output"] for tc in test_cases],
        }
        # Suppress verbose checker prints (failed checks, runtime traces) from child process.
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                results = run_test(problem={"input_output": io_payload}, test=solution)
        score = sum(1 for r in results if r is True) / len(results) if results else 0.0
        result_queue.put(("ok", score))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _parse_functional_input(input_obj) -> list:
    """
    Parse functional-test input into a list of Python call arguments.
    Handles two formats:
      - Single-line: '["a", "b"]'   → [["a", "b"]]   (one list argument)
      - Multi-line:  '[1,2]\n[3,4]' → [[1,2], [3,4]] (two arguments)
      - Already-typed input:
          - list/tuple -> treated as argument list
          - scalar/dict -> treated as single argument
    """
    import ast, json as _json

    # Already-typed APPS inputs (common in function-style tasks).
    if not isinstance(input_obj, str):
        if isinstance(input_obj, tuple):
            return list(input_obj)
        if isinstance(input_obj, list):
            return input_obj
        return [input_obj]

    lines = [l.strip() for l in input_obj.strip().split("\n") if l.strip()]
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


def _parse_functional_output(output_obj):
    """Parse functional-test expected output into a Python object."""
    import ast, json as _json

    if not isinstance(output_obj, str):
        return output_obj

    s = output_obj.strip()
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

        parsed_inputs = []
        parsed_outputs = []
        for tc in test_cases:
            if isinstance(tc, dict):
                in_obj = tc.get("input")
                out_obj = tc.get("output")
            else:
                # Fallback for tuple/list test-case rows.
                in_obj = tc[0] if isinstance(tc, (list, tuple)) and len(tc) > 0 else tc
                out_obj = tc[1] if isinstance(tc, (list, tuple)) and len(tc) > 1 else None
            parsed_inputs.append(_parse_functional_input(in_obj))
            parsed_outputs.append(_parse_functional_output(out_obj))

        io_payload = {"inputs": parsed_inputs, "outputs": parsed_outputs, "fn_name": func_name}
        # Suppress verbose checker prints (failed checks, runtime traces) from child process.
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                results = run_test(problem={"input_output": io_payload}, test=solution)
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

    func_name = get_function_name(problem)
    is_function_style = is_function_style_problem(problem)

    if is_function_style:
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
    func_name = get_function_name(problem)
    is_function_style = is_function_style_problem(problem)

    if is_function_style:
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

    # Use persistent thread pool — each task still executes user code in
    # its own subprocess via _run_subprocess, preserving isolation.
    global _batch_pool
    try:
        pool = _get_batch_pool(n_workers)
        results = list(pool.map(_pool_worker_wrapper, args))
    except Exception:
        # Pool may have died; recreate once and retry
        _batch_pool = None
        pool = _get_batch_pool(n_workers)
        results = list(pool.map(_pool_worker_wrapper, args))

    scores = [r[0] for r in results]
    statuses = [r[1] for r in results]
    n = len(scores)

    stats = {
        "mean_score": sum(scores) / n,
        "zero_scores": sum(1 for s in scores if s == 0.0),
        "perfect_scores": sum(1 for s in scores if s == 1.0),
        "ok_count": sum(1 for s in statuses if s == "ok"),
        "error_count": sum(1 for s in statuses if s == "error"),
        "empty_count": sum(1 for s in statuses if s == "empty"),
        "timeout_count": sum(1 for s in statuses if s == "timeout"),
        "zero_ok_count": sum(1 for score, status in zip(scores, statuses) if status == "ok" and score == 0.0),
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
    def _strip_markdown_fences(code_text: str) -> str:
        """
        Remove wrapping markdown fences if the extracted code still includes them.
        Handles:
          ```python
          ...
          ```
        and:
          ```
          ...
          ```
        """
        text = code_text.strip()
        # If a fenced block appears anywhere inside, prefer its content.
        # This covers malformed cases where extra text leaks before the real fenced code.
        fenced_anywhere = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fenced_anywhere:
            return fenced_anywhere[-1].strip()

        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            # Drop opening fence line.
            lines = lines[1:]
            # Drop trailing fence line if present.
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        return "\n".join(lines).strip()

    match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
    if match:
        return _strip_markdown_fences(match.group(1))
    # fallback: ```python blocks
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return _strip_markdown_fences(match.group(1))
    # fallback: any ``` block
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return _strip_markdown_fences(match.group(1))
    return None
