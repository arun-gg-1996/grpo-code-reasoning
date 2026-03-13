"""
build_lcb_dataset.py

Parses LiveCodeBench seen problems (before 2024-01-01 cutoff) and saves
them to data/clean/lcb_seen_clean.jsonl.

Steps:
  1. Filter to seen problems (contest_date < 2024-01-01)
  2. Decompress private_test_cases (base64 + gzip)
  3. Split by testtype: "stdin" vs "functional"
  4. Save to lcb_seen_clean.jsonl

No execution or validation — LCB is a curated benchmark with verified
test cases. Reference solutions will be added later via generate_lcb_solutions.py.

Usage:
    python scripts/build_lcb_dataset.py
"""

import base64
import gzip
import json
import os
import sys
from datetime import datetime

sys.set_int_max_str_digits(100000)

from datasets import load_dataset

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

OUTPUT_PATH     = "../data/clean/lcb_seen_clean.jsonl"
CUTOFF_DATE     = datetime(2024, 1, 1)
SAMPLING_WEIGHT = 2.0

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def parse_date(date_str: str) -> datetime | None:
    """Parse ISO date string from LCB contest_date field."""
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except Exception:
        return None


def decompress_test_cases(compressed: str) -> list:
    """
    LCB private_test_cases are base64-encoded gzipped JSON.
    Returns list of {input, output, testtype} dicts.
    """
    try:
        decoded     = base64.b64decode(compressed)
        decompressed = gzip.decompress(decoded)
        return json.loads(decompressed)
    except Exception:
        return []


def parse_test_cases(problem: dict) -> list:
    """
    Combine public and private test cases.
    Each test case: {input, output, testtype}
    testtype is "stdin" or "functional"
    """
    public = []
    try:
        raw = problem.get("public_test_cases", [])
        if isinstance(raw, str):
            raw = json.loads(raw)
        public = raw or []
    except Exception:
        pass

    private = []
    try:
        raw = problem.get("private_test_cases", "")
        if raw:
            private = decompress_test_cases(raw)
    except Exception:
        pass

    all_tests = public + private
    # normalize — ensure each has input/output/testtype
    result = []
    for tc in all_tests:
        if tc.get("input") is not None and tc.get("output") is not None:
            result.append({
                "input":    tc["input"],
                "output":   tc["output"],
                "testtype": tc.get("testtype", "stdin"),
            })
    return result


def get_func_name(problem: dict) -> str | None:
    """Extract function name from metadata field for LeetCode problems."""
    try:
        meta = problem.get("metadata", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        return meta.get("func_name")
    except Exception:
        return None


def format_problem(problem: dict, test_cases: list) -> dict:
    """Format a LCB problem into our training format."""
    func_name = get_func_name(problem)
    is_lc     = problem.get("platform", "").lower() == "leetcode"

    return {
        "question_id":      problem.get("question_id", ""),
        "question_title":   problem.get("question_title", ""),
        "question":         problem.get("question_content", ""),
        "platform":         problem.get("platform", ""),
        "difficulty":       problem.get("difficulty", ""),
        "contest_date":     problem.get("contest_date", ""),
        "starter_code":     problem.get("starter_code", ""),
        "func_name":        func_name,
        "is_leetcode":      is_lc,
        "sampling_weight":  SAMPLING_WEIGHT,
        "test_cases":       test_cases,
        "stdin_tests":      [tc for tc in test_cases if tc["testtype"] == "stdin"],
        "functional_tests": [tc for tc in test_cases if tc["testtype"] == "functional"],
        "reference_solution": None,   # populated later by generate_lcb_solutions.py
        "source":           "lcb_seen",
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("BUILD LCB SEEN DATASET")
    print(f"Cutoff: before {CUTOFF_DATE.date()}")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("\nLoading LiveCodeBench...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_v2"
    )
    print(f"Total LCB problems: {len(ds)}")

    # ── filter to seen (before cutoff) ──────────────────────
    seen = []
    skipped_no_date = 0
    for p in ds:
        date = parse_date(p.get("contest_date", ""))
        if date is None:
            skipped_no_date += 1
            continue
        if date < CUTOFF_DATE:
            seen.append(p)

    print(f"Seen (before {CUTOFF_DATE.date()}): {len(seen)}")
    print(f"Unseen (after cutoff):              {len(ds) - len(seen) - skipped_no_date}")
    if skipped_no_date:
        print(f"Skipped (no date):                  {skipped_no_date}")

    # ── platform breakdown ───────────────────────────────────
    platforms = {}
    for p in seen:
        plat = p.get("platform", "unknown")
        platforms[plat] = platforms.get(plat, 0) + 1
    print(f"\nPlatform breakdown:")
    for plat, count in sorted(platforms.items()):
        print(f"  {plat}: {count}")

    # ── parse and format ─────────────────────────────────────
    print(f"\nParsing test cases...")
    results         = []
    no_tests        = 0
    public_only     = 0
    private_failed  = 0
    stdin_count     = 0
    functional_count = 0

    for problem in seen:
        test_cases = parse_test_cases(problem)

        if not test_cases:
            no_tests += 1
            continue

        # count private decompress failures
        raw_private = problem.get("private_test_cases", "")
        if raw_private and not decompress_test_cases(raw_private):
            private_failed += 1

        has_private = bool(raw_private)
        if not has_private:
            public_only += 1

        # count by type
        stdin_count      += sum(1 for tc in test_cases if tc["testtype"] == "stdin")
        functional_count += sum(1 for tc in test_cases if tc["testtype"] == "functional")

        results.append(format_problem(problem, test_cases))

    # ── save ─────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        for p in results:
            f.write(json.dumps(p) + "\n")

    # ── summary ──────────────────────────────────────────────
    lc_count    = sum(1 for p in results if p["is_leetcode"])
    other_count = len(results) - lc_count

    avg_tests = sum(len(p["test_cases"]) for p in results) / len(results) if results else 0

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Saved:              {len(results)} problems → {OUTPUT_PATH}")
    print(f"  Skipped (no tests): {no_tests}")
    print(f"  LeetCode:           {lc_count}")
    print(f"  Other (stdio):      {other_count}")
    print(f"  Avg test cases:     {avg_tests:.1f}")
    print(f"  Total stdin tests:      {stdin_count}")
    print(f"  Total functional tests: {functional_count}")
    print(f"  Sampling weight:    {SAMPLING_WEIGHT}x")
    print(f"  Weighted total:     {len(results) * SAMPLING_WEIGHT:.0f}")

    print(f"\n{'=' * 60}")
    print(f"FULL TRAINING DATASET")
    print(f"{'=' * 60}")

    apps_path = "../data/clean/apps_clean.jsonl"
    if os.path.exists(apps_path):
        with open(apps_path) as f:
            lines = f.readlines()
        apps_count    = len(lines)
        apps_weighted = sum(json.loads(l).get("sampling_weight", 1.0) for l in lines)
        print(f"  APPS:     {apps_count} problems  weighted: {apps_weighted:.0f}")
    else:
        print(f"  APPS:     not found — run validate_apps.py first")
        apps_weighted = 0

    lcb_weighted = len(results) * SAMPLING_WEIGHT
    print(f"  LCB seen: {len(results)} problems  weighted: {lcb_weighted:.0f}")
    print(f"  TOTAL:                       weighted: {apps_weighted + lcb_weighted:.0f}")
    print(f"\n  Note: LCB reference solutions pending generate_lcb_solutions.py")


if __name__ == "__main__":
    main()