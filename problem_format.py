"""
problem_format.py

Shared helpers for deciding whether a problem expects function-style output
or stdin/stdout program output.

Keeping this logic centralized prevents prompt/scoring mismatches.
"""

from __future__ import annotations


def get_function_name(problem: dict) -> str:
    """Return the expected function name if present, else empty string."""
    fn = problem.get("func_name") or problem.get("fn_name") or ""
    return fn.strip() if isinstance(fn, str) else ""


def is_function_style_problem(problem: dict) -> bool:
    """
    Determine whether a problem should be solved as a function/class task.

    Rules mirror execution scoring logic:
    - Any leetcode task with a function name is function-style.
    - Presence of `fn_name` in the schema indicates function-style APPS rows.
    - Explicit `functional_tests` also indicates function-style.
    """
    fn = get_function_name(problem)
    if not fn:
        return False
    if bool(problem.get("is_leetcode")):
        return True
    if "fn_name" in problem:
        return True
    if problem.get("functional_tests"):
        return True
    return False
