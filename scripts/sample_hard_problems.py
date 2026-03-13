"""
sample_hard_problems.py

Randomly samples 10 hard problems from lcb_seen_clean.jsonl and apps_clean.jsonl
for manual Gemini judge evaluation.

Outputs a clean JSON file with problem description, test cases, and expected outputs.

Usage:
    python sample_hard_problems.py
    python sample_hard_problems.py --source lcb       # LCB only
    python sample_hard_problems.py --source apps      # APPS only
    python sample_hard_problems.py --n 5              # sample 5 instead of 10
"""

import json
import random
import argparse
from pathlib import Path

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

DATA_DIR = Path("data/clean")
APPS_PATH = DATA_DIR / "apps_clean.jsonl"
LCB_PATH  = DATA_DIR / "lcb_seen_clean.jsonl"
OUTPUT_PATH = Path("data/judge_eval_sample.json")

random.seed(42)  # change seed for different sample

import sys
sys.set_int_max_str_digits(100000)

# ─────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ─────────────────────────────────────────
# Filters
# ─────────────────────────────────────────

def filter_hard(problems: list[dict], source: str) -> list[dict]:
    """
    Filter to hard problems only.
    APPS: difficulty == "competition"
    LCB:  difficulty == "hard"
    """
    if source == "apps":
        return [p for p in problems if p.get("difficulty") == "competition"]
    elif source == "lcb":
        return [p for p in problems if p.get("difficulty") == "hard"]
    return []


# ─────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────

def format_problem(problem: dict, source: str, idx: int) -> dict:
    """
    Extract relevant fields for judge evaluation.
    Keeps: question, test cases with inputs and expected outputs.
    """
    test_cases = problem.get("test_cases", [])[:5]  # cap at 5 for readability

    formatted_tests = []
    for i, tc in enumerate(test_cases):
        formatted_tests.append({
            "test_number": i + 1,
            "input": tc.get("input", ""),
            "expected_output": tc.get("output", ""),
        })

    return {
        "id": idx,
        "source": source,
        "platform": problem.get("platform", source),
        "difficulty": problem.get("difficulty", "hard"),
        "problem": problem.get("question", problem.get("problem_statement", "")),
        "test_cases": formatted_tests,
        "func_name": problem.get("func_name", None),      # None for stdin problems
        "starter_code": problem.get("starter_code", None), # LeetCode only
        "is_leetcode": problem.get("is_leetcode", False),
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["lcb", "apps", "both"], default="both")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    pool = []

    if args.source in ("apps", "both"):
        if APPS_PATH.exists():
            apps = load_jsonl(APPS_PATH)
            hard_apps = filter_hard(apps, "apps")
            print(f"APPS competition problems: {len(hard_apps)}")
            pool.extend([("apps", p) for p in hard_apps])
        else:
            print(f"WARNING: {APPS_PATH} not found, skipping APPS")

    if args.source in ("lcb", "both"):
        if LCB_PATH.exists():
            lcb = load_jsonl(LCB_PATH)
            hard_lcb = filter_hard(lcb, "lcb")
            print(f"LCB hard problems: {len(hard_lcb)}")
            pool.extend([("lcb", p) for p in hard_lcb])
        else:
            print(f"WARNING: {LCB_PATH} not found, skipping LCB")

    if not pool:
        print("ERROR: no problems found. Check data paths.")
        return

    # filter to problems that have at least one test case
    pool = [(src, p) for src, p in pool if p.get("test_cases")]
    print(f"Problems with test cases: {len(pool)}")

    if not pool:
        print("ERROR: no problems with test cases found.")
        return

    # sample n problems
    n = min(args.n, len(pool))
    sampled = random.sample(pool, n)

    # format
    output = []
    for idx, (source, problem) in enumerate(sampled):
        formatted = format_problem(problem, source, idx + 1)
        output.append(formatted)

    # save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSampled {n} problems → {OUTPUT_PATH}")
    print("\nBreakdown:")
    for p in output:
        print(f"  [{p['id']}] {p['platform']} | {p['difficulty']} | "
              f"{'LeetCode functional' if p['is_leetcode'] else 'stdin/stdout'} | "
              f"{len(p['test_cases'])} test cases")

    print(f"\nNext: open {OUTPUT_PATH} and feed problems to Gemini Flash manually")


if __name__ == "__main__":
    main()