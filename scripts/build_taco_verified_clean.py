"""
build_taco_verified_clean.py

Build a Python-only, sandbox-validated TACO-verified dataset with
fuzzy dedup against existing train/eval corpora.

Pipeline:
1) Stream rows from TACO-verified JSON
2) Keep rows matching quality gates (difficulty, min tests, python solutions)
3) Dedup against:
   - data/clean/apps_clean.jsonl
   - data/clean/lcb_seen_clean.jsonl
   - data/clean/lcb_unseen_clean.jsonl
   using exact normalized text + fuzzy token-overlap/Jaccard
4) Validate by executing candidate solutions in isolated subprocess sandbox
5) Save clean rows + failed samples + summary

Usage:
  python scripts/build_taco_verified_clean.py
  python scripts/build_taco_verified_clean.py --min-tests 5 --max-solutions 3
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from json import JSONDecoder
from pathlib import Path

# Must be set before json parsing some long numeric strings.
sys.set_int_max_str_digits(100000)

from huggingface_hub import hf_hub_download


# ─────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "clean" / "taco_verified_clean.jsonl"
DEFAULT_FAILED_DIR = REPO_ROOT / "data" / "failed" / "taco_verified"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "results" / "analysis" / "taco_verified_build_summary.json"

APPS_CLEAN_PATH = REPO_ROOT / "data" / "clean" / "apps_clean.jsonl"
LCB_SEEN_PATH = REPO_ROOT / "data" / "clean" / "lcb_seen_clean.jsonl"
LCB_UNSEEN_PATH = REPO_ROOT / "data" / "clean" / "lcb_unseen_clean.jsonl"

FAILED_SAMPLE_SIZE = 30

# Training-shape defaults for high-value subset (can be overridden by flags)
DEFAULT_MIN_TESTS = 5
DEFAULT_MAX_TESTS = 10
DEFAULT_VALIDATION_THRESHOLD = 1.0
DEFAULT_MAX_SOLUTIONS_TO_TRY = 3
DEFAULT_SUBPROCESS_TIMEOUT = 15

DEFAULT_ALLOWED_SOURCES = {"codeforces", "atcoder"}
DEFAULT_ALLOWED_DIFFICULTIES = {"medium", "medium_hard", "hard", "very_hard"}

# Fuzzy dedup knobs
MAX_TOKEN_DF = 800
MIN_SHARED_TOKENS = 8
JACCARD_DUP_THRESHOLD = 0.82

# Keep source reward blend similar to LCB-ish weighting style
DEFAULT_SAMPLING_WEIGHT = 1.8


# ─────────────────────────────────────────
# Helpers: normalization + fuzzy dedup
# ─────────────────────────────────────────

WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return t.strip()


def tokenize_for_dedup(text: str) -> set[str]:
    toks = set(WORD_RE.findall(text))
    # Drop tiny tokens and common boilerplate tokens.
    toks = {t for t in toks if len(t) >= 3 and t not in {"input", "output", "examples", "example", "constraints"}}
    return toks


class FuzzyIndex:
    def __init__(self):
        self.texts: list[str] = []
        self.tokens: list[set[str]] = []
        self.exact_hashes: set[str] = set()
        self.token_to_ids: dict[str, list[int]] = defaultdict(list)
        self.token_df: Counter[str] = Counter()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def add(self, norm_text: str):
        if not norm_text:
            return
        h = self._hash_text(norm_text)
        if h in self.exact_hashes:
            return
        idx = len(self.texts)
        self.texts.append(norm_text)
        tok = tokenize_for_dedup(norm_text)
        self.tokens.append(tok)
        self.exact_hashes.add(h)
        for t in tok:
            self.token_to_ids[t].append(idx)
            self.token_df[t] += 1

    def is_exact_dup(self, norm_text: str) -> bool:
        return self._hash_text(norm_text) in self.exact_hashes

    def is_fuzzy_dup(self, norm_text: str) -> tuple[bool, float]:
        c_toks = tokenize_for_dedup(norm_text)
        if len(c_toks) < MIN_SHARED_TOKENS:
            return False, 0.0

        overlap = Counter()
        for t in c_toks:
            if self.token_df.get(t, 0) <= MAX_TOKEN_DF:
                for idx in self.token_to_ids.get(t, []):
                    overlap[idx] += 1

        if not overlap:
            return False, 0.0

        best = 0.0
        for idx, shared in overlap.items():
            if shared < MIN_SHARED_TOKENS:
                continue
            t_ref = self.tokens[idx]
            inter = len(c_toks & t_ref)
            union = len(c_toks | t_ref)
            if union == 0:
                continue
            j = inter / union
            if j > best:
                best = j
            if j >= JACCARD_DUP_THRESHOLD:
                return True, j

        return False, best


# ─────────────────────────────────────────
# Helpers: streaming TACO JSON
# ─────────────────────────────────────────

def iter_json_array(fp, chunk_size: int = 1 << 20):
    dec = JSONDecoder()
    buf = ""
    pos = 0

    while True:
        chunk = fp.read(chunk_size)
        if not chunk:
            raise ValueError("Empty JSON")
        buf += chunk
        i = 0
        while i < len(buf) and buf[i].isspace():
            i += 1
        if i < len(buf):
            if buf[i] != "[":
                raise ValueError("Expected JSON array")
            pos = i + 1
            break

    while True:
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf):
                break
            chunk = fp.read(chunk_size)
            if not chunk:
                return
            buf += chunk

        if buf[pos] == "]":
            return
        if buf[pos] == ",":
            pos += 1
            continue

        while True:
            try:
                obj, end = dec.raw_decode(buf, pos)
                pos = end
                yield obj
                if pos > (1 << 20):
                    buf = buf[pos:]
                    pos = 0
                break
            except ValueError:
                chunk = fp.read(chunk_size)
                if not chunk:
                    raise
                buf += chunk


# ─────────────────────────────────────────
# Helpers: data parsing + validation
# ─────────────────────────────────────────

def normalize_difficulty(raw: str) -> str:
    d = (raw or "").strip().lower()
    if d in {"introductory", "easy"}:
        return "easy"
    if d in {"interview", "medium", "medium_hard"}:
        return "medium"
    if d in {"competition", "hard", "very_hard"}:
        return "hard"
    return "medium"


def parse_io(io_raw) -> dict | None:
    io = io_raw
    if isinstance(io, str):
        try:
            io = json.loads(io)
        except Exception:
            return None
    if not isinstance(io, dict):
        return None
    ins = io.get("inputs")
    outs = io.get("outputs")
    if not isinstance(ins, list) or not isinstance(outs, list):
        return None
    if not ins or not outs:
        return None
    n = min(len(ins), len(outs))
    if n <= 0:
        return None
    out = {
        "inputs": ins[:n],
        "outputs": outs[:n],
    }
    fn_name = io.get("fn_name")
    if isinstance(fn_name, str) and fn_name.strip():
        out["fn_name"] = fn_name.strip()
    return out


def parse_solutions(raw) -> list[str]:
    sols = raw
    if isinstance(sols, str):
        try:
            sols = json.loads(sols)
        except Exception:
            return []
    if not isinstance(sols, list):
        return []
    return [s for s in sols if isinstance(s, str) and s.strip()]


def looks_python(code: str) -> bool:
    if not isinstance(code, str):
        return False
    s = code.strip()
    if not s:
        return False

    low = s.lower()
    non_py_markers = (
        "#include", "using namespace std", "std::", "int main(", "scanf(", "printf(",
        "cin >>", "cout <<", "public class", "system.out.", "import java.",
        "func main()", "package main", "console.writeline"
    )
    if any(m in low for m in non_py_markers):
        return False

    try:
        ast.parse(s)
        return True
    except Exception:
        # lightweight fallback for odd but pythonic snippets
        hints = ("def ", "import ", "print(", "input(", "sys.stdin", "sys.stdout", "if __name__")
        return any(h in s for h in hints) and ("{" not in s)


def _run_test_worker(solution: str, capped_io: dict, result_queue):
    sys.path.insert(0, str(REPO_ROOT))
    sys.set_int_max_str_digits(100000)
    try:
        from sandbox.testing_util import run_test
        problem_dict = {"input_output": capped_io}
        results = run_test(problem=problem_dict, test=solution)
        passed = sum(1 for r in results if r is True)
        score = passed / len(results) if results else 0.0
        result_queue.put(score)
    except Exception:
        result_queue.put(0.0)


def score_solution(solution: str, io: dict, max_tests: int, timeout_s: int) -> float:
    capped_io = {
        "inputs": io["inputs"][:max_tests],
        "outputs": io["outputs"][:max_tests],
    }
    if io.get("fn_name"):
        capped_io["fn_name"] = io["fn_name"]

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_test_worker, args=(solution, capped_io, queue))
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return 0.0

    return queue.get() if not queue.empty() else 0.0


# ─────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────

def load_reference_index() -> FuzzyIndex:
    idx = FuzzyIndex()
    ref_paths = [APPS_CLEAN_PATH, LCB_SEEN_PATH, LCB_UNSEEN_PATH]

    for path in ref_paths:
        if not path.exists():
            print(f"WARNING: missing reference file: {path}")
            continue
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                q = normalize_text(row.get("question", ""))
                if q:
                    idx.add(q)

    print(f"Reference index loaded: {len(idx.texts)} unique questions")
    return idx


def save_failed_sample(items: list[dict], out_dir: Path, name: str):
    if not items:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = random.sample(items, min(FAILED_SAMPLE_SIZE, len(items)))
    path = out_dir / f"{name}_failed.jsonl"
    with open(path, "w") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")
    print(f"  saved {len(sample)} failed samples -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tests", type=int, default=DEFAULT_MIN_TESTS)
    ap.add_argument("--max-tests", type=int, default=DEFAULT_MAX_TESTS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_VALIDATION_THRESHOLD)
    ap.add_argument("--max-solutions", type=int, default=DEFAULT_MAX_SOLUTIONS_TO_TRY)
    ap.add_argument("--timeout-s", type=int, default=DEFAULT_SUBPROCESS_TIMEOUT)
    ap.add_argument("--sampling-weight", type=float, default=DEFAULT_SAMPLING_WEIGHT)
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    ap.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--allowed-sources", nargs="*", default=sorted(DEFAULT_ALLOWED_SOURCES))
    ap.add_argument("--allowed-difficulties", nargs="*", default=sorted(DEFAULT_ALLOWED_DIFFICULTIES))
    args = ap.parse_args()

    t0 = time.time()
    random.seed(42)

    allowed_sources = {s.lower() for s in args.allowed_sources}
    allowed_difficulties = {d.lower() for d in args.allowed_difficulties}

    output_path = Path(args.output)
    failed_dir = DEFAULT_FAILED_DIR
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILD TACO-VERIFIED CLEAN (python-only + fuzzy dedup + sandbox validation)")
    print("=" * 70)
    print(f"allowed_sources={sorted(allowed_sources)}")
    print(f"allowed_difficulties={sorted(allowed_difficulties)}")
    print(f"min_tests={args.min_tests}, max_tests={args.max_tests}, threshold={args.threshold}")

    print("\nDownloading TACO-verified metadata/file path...")
    taco_json_path = hf_hub_download(
        repo_id="likaixin/TACO-verified",
        repo_type="dataset",
        filename="taco_verified.json",
    )
    print(f"TACO json: {taco_json_path}")

    print("\nBuilding reference fuzzy index from existing train + eval files...")
    ref_index = load_reference_index()

    stats = Counter()
    by_source = Counter()
    by_diff = Counter()

    failed = defaultdict(list)
    clean_rows = []
    seen_candidate_hashes = set()

    print("\nStreaming TACO rows and validating...")
    with open(taco_json_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(iter_json_array(f), start=1):
            stats["total_rows"] += 1
            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                print(
                    f"  [{i}] clean={len(clean_rows)} dropped={stats['total_rows']-len(clean_rows)} "
                    f"rate={rate:.1f}/s"
                )

            src = (row.get("source") or "unknown").lower().strip()
            raw_diff = (row.get("difficulty") or "unknown").lower().strip()

            if src not in allowed_sources:
                stats["drop_source"] += 1
                continue
            if raw_diff not in allowed_difficulties:
                stats["drop_difficulty"] += 1
                continue

            io = parse_io(row.get("input_output"))
            if io is None:
                stats["drop_no_io"] += 1
                failed["no_io"].append({"id": row.get("id"), "source": src, "difficulty": raw_diff})
                continue

            n_tests = min(len(io["inputs"]), len(io["outputs"]))
            if n_tests < args.min_tests:
                stats["drop_low_tests"] += 1
                failed["low_tests"].append(
                    {"id": row.get("id"), "source": src, "difficulty": raw_diff, "n_tests": n_tests}
                )
                continue

            sols = parse_solutions(row.get("solutions"))
            if not sols:
                stats["drop_no_solutions"] += 1
                failed["no_solutions"].append({"id": row.get("id"), "source": src, "difficulty": raw_diff})
                continue

            py_solutions = [s for s in sols if looks_python(s)]
            if not py_solutions:
                stats["drop_no_python_solution"] += 1
                failed["no_python_solution"].append(
                    {"id": row.get("id"), "source": src, "difficulty": raw_diff, "num_solutions": len(sols)}
                )
                continue

            question = row.get("question", "")
            norm_q = normalize_text(question)
            if not norm_q:
                stats["drop_empty_question"] += 1
                continue

            # intra-candidate exact dedup
            qh = hashlib.sha1(norm_q.encode("utf-8")).hexdigest()
            if qh in seen_candidate_hashes:
                stats["drop_candidate_exact_dup"] += 1
                continue
            seen_candidate_hashes.add(qh)

            # external exact/fuzzy dedup
            if ref_index.is_exact_dup(norm_q):
                stats["drop_exact_dup_ref"] += 1
                failed["exact_dup_ref"].append(
                    {"id": row.get("id"), "source": src, "difficulty": raw_diff, "q": question[:200]}
                )
                continue

            is_dup, score = ref_index.is_fuzzy_dup(norm_q)
            if is_dup:
                stats["drop_fuzzy_dup_ref"] += 1
                failed["fuzzy_dup_ref"].append(
                    {
                        "id": row.get("id"),
                        "source": src,
                        "difficulty": raw_diff,
                        "fuzzy_score": round(score, 4),
                        "q": question[:200],
                    }
                )
                continue

            # sandbox validation
            best_score = 0.0
            passing_solution = None
            for sol in py_solutions[: args.max_solutions]:
                s = score_solution(sol, io, args.max_tests, args.timeout_s)
                if s > best_score:
                    best_score = s
                    passing_solution = sol
                if s >= args.threshold:
                    break

            if best_score < args.threshold:
                stats["drop_score"] += 1
                failed["score"].append(
                    {
                        "id": row.get("id"),
                        "source": src,
                        "difficulty": raw_diff,
                        "best_score": round(best_score, 4),
                        "q": question[:200],
                    }
                )
                continue

            norm_diff = normalize_difficulty(raw_diff)
            clean = {
                "problem_id": str(row.get("id", "")),
                "question": question,
                "difficulty": norm_diff,
                "sampling_weight": args.sampling_weight,
                "fn_name": io.get("fn_name"),
                "io": {
                    "inputs": io["inputs"][: args.max_tests],
                    "outputs": io["outputs"][: args.max_tests],
                    **({"fn_name": io["fn_name"]} if io.get("fn_name") else {}),
                },
                "solutions": py_solutions,
                "passing_solution": passing_solution,
                "validation_score": round(best_score, 4),
                "source": "taco_verified",
                "taco_source": src,
                "taco_difficulty": raw_diff,
                "num_tests": n_tests,
            }
            clean_rows.append(clean)
            stats["clean"] += 1
            by_source[src] += 1
            by_diff[norm_diff] += 1

            # Add accepted to reference index to avoid near-duplicates within TACO itself
            ref_index.add(norm_q)

    # Write clean output
    with open(output_path, "w") as f:
        for row in clean_rows:
            f.write(json.dumps(row) + "\n")

    # Write failed samples
    for reason, items in failed.items():
        save_failed_sample(items, failed_dir, reason)

    elapsed = time.time() - t0
    summary = {
        "config": {
            "allowed_sources": sorted(allowed_sources),
            "allowed_difficulties": sorted(allowed_difficulties),
            "min_tests": args.min_tests,
            "max_tests": args.max_tests,
            "threshold": args.threshold,
            "max_solutions": args.max_solutions,
            "timeout_s": args.timeout_s,
            "sampling_weight": args.sampling_weight,
            "jaccard_dup_threshold": JACCARD_DUP_THRESHOLD,
            "min_shared_tokens": MIN_SHARED_TOKENS,
        },
        "stats": dict(stats),
        "clean_by_taco_source": dict(by_source),
        "clean_by_normalized_difficulty": dict(by_diff),
        "output_path": str(output_path),
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Clean rows: {len(clean_rows)} -> {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    print("\nKey stats:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print("\nClean by source:")
    for k, v in by_source.most_common():
        print(f"  {k}: {v}")
    print("\nClean by normalized difficulty:")
    for k, v in by_diff.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
