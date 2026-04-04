#!/usr/bin/env python3
"""
Regression test for fence sanitization in extract_code().

Uses real training completions from latest results/train/*/train_details.jsonl and
checks that completions containing markdown fences produce extracted code without
fence markers after sanitization.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward.execution import extract_code  # noqa: E402


def main() -> int:
    paths = sorted(glob.glob("results/train/*/train_details.jsonl"))
    if not paths:
        print("SKIP: no results/train/*/train_details.jsonl found")
        return 0

    path = paths[-1]
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    fenced = [r for r in rows if "```" in (r.get("completion_text") or "")]
    if not fenced:
        print(f"SKIP: no fenced completions found in {path}")
        return 0

    checked = 0
    bad = []
    for r in fenced:
        completion = r.get("completion_text") or ""
        code, _extract_mode = extract_code(completion)
        if code is None:
            continue
        checked += 1
        if "```" in code:
            bad.append(r)

    print(f"file: {path}")
    print(f"rows: {len(rows)}")
    print(f"fenced_completions: {len(fenced)}")
    print(f"checked_with_extractable_code: {checked}")
    print(f"still_contains_fence_after_extract: {len(bad)}")

    if bad:
        ex = bad[0]
        print("Example bad row:")
        print(f"  question_id={ex.get('question_id')} source={ex.get('source')} diff={ex.get('difficulty')}")
        return 1

    print("PASS: extracted code is fence-free for all checked fenced completions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
