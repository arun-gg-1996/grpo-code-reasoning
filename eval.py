"""
eval.py

LiveCodeBench v4 evaluation with pass@1 metric.
Uses vLLM for fast inference on the held-out LCB unseen set.

Usage:
    # Evaluate a checkpoint
    python eval.py --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo

    # Evaluate base model (baseline)
    python eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct

    # Smoke test
    python eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --smoke-test
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

sys.set_int_max_str_digits(0)  # APPS data contains very large integers in JSON fields

import numpy as np
from tqdm import tqdm

from config import (
    LCB_EVAL_PATH,
    LCB_SEEN_PATH,
    EVAL_SYSTEM_PROMPT_STDIO,
    EVAL_SYSTEM_PROMPT_LEETCODE,
    EVAL_TEMPERATURE,
    EVAL_BATCH_SIZE,
    EVAL_N_GENERATIONS,
    EVAL_K_VALUES,
    MAX_NEW_TOKENS,
    normalize_difficulty,
)
from reward.execution import score_batch as exec_score_batch, extract_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _slugify_model_name(model: str) -> str:
    """Turn model path/ID into a safe filename fragment."""
    s = model.strip().replace("\\", "/").split("/")[-1]
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("-") or "model"


def _fmt_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# pass@k estimator (unbiased, from Chen et al. 2021)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator.
    n = total generations, c = correct generations, k = k value.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_test_cases(problem: dict) -> dict:
    """Normalize APPS io format to standard test_cases/stdin_tests format."""
    if problem.get("test_cases"):
        return problem
    io = problem.get("io")
    if io and isinstance(io, dict):
        inputs = io.get("inputs", [])
        outputs = io.get("outputs", [])
        tc = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]
        problem["test_cases"] = tc
        problem["stdin_tests"] = tc
    return problem


def load_eval_problems(path: str) -> list[dict]:
    """Load evaluation problems from JSONL."""
    problems = []
    if not os.path.exists(path):
        logger.warning(f"Eval data not found at {path}, falling back to LCB seen set")
        path = LCB_SEEN_PATH

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            p["difficulty"] = normalize_difficulty(p.get("difficulty", "medium"))
            p = _normalize_test_cases(p)
            problems.append(p)
    return problems


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_solutions(
    model_path: str,
    problems: list[dict],
    n_generations: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, list[str]]:
    """
    Generate n_generations solutions per problem using vLLM.
    Returns: {problem_id: [code_1, code_2, ..., code_n]}
    """
    from vllm import LLM, SamplingParams

    logger.info(f"Loading model: {model_path}")
    llm = LLM(model=model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_generations,
    )

    # Build prompts
    prompts = []
    problem_ids = []
    for p in problems:
        pid = p.get("question_id") or p.get("problem_id", "unknown")
        question = p.get("question", "")
        is_lc = p.get("is_leetcode", False)

        if is_lc:
            prompt = f"{EVAL_SYSTEM_PROMPT_LEETCODE}\n\n{question}"
            starter = p.get("starter_code", "")
            if starter:
                prompt += f"\n\n{starter}"
        else:
            prompt = f"{EVAL_SYSTEM_PROMPT_STDIO}\n\n{question}"

        prompts.append(prompt)
        problem_ids.append(pid)

    results = {}
    total = len(prompts)
    logger.info(
        f"Generating {n_generations} solutions for {total} problems "
        f"(batch_size={batch_size})..."
    )

    gen_start = time.time()
    gen_pbar = tqdm(total=total, desc="Generating", unit="prompt")

    # Generate in user-configured mini-batches to control memory/throughput.
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]
        batch_ids = problem_ids[start:end]

        batch_outputs = llm.generate(batch_prompts, sampling_params)
        for pid, output in zip(batch_ids, batch_outputs):
            codes = []
            for gen in output.outputs:
                code = extract_code(gen.text)
                codes.append(code or "")
            results[pid] = codes

        gen_pbar.update(end - start)
        done = end
        elapsed = time.time() - gen_start
        eta = ((elapsed / done) * (total - done)) if done > 0 else 0.0
        logger.info(
            f"Generation progress: {done}/{total} prompts | "
            f"elapsed={_fmt_seconds(elapsed)} | eta={_fmt_seconds(eta)}"
        )

    gen_pbar.close()

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    problems: list[dict],
    solutions: dict[str, list[str]],
    k_values: list[int],
    include_details: bool = False,
    debug_output_path: str = "",
    score_chunk_size: int = 128,
) -> tuple[dict, dict, list[dict]]:
    """
    Evaluate solutions against test cases using parallel batch scoring.
    Returns pass@k scores broken down by: overall, difficulty, and platform.
    """
    # Flatten all (code, problem) pairs.
    flat_rows: list[dict] = []
    for p_idx, p in enumerate(problems):
        pid = p.get("question_id") or p.get("problem_id", "unknown")
        diff = p.get("difficulty", "medium")
        platform = p.get("platform", "unknown").lower()
        question = p.get("question", "")
        for completion_idx, code in enumerate(solutions.get(pid, []), start=1):
            flat_rows.append(
                {
                    "problem_idx": p_idx,
                    "question_id": pid,
                    "difficulty": diff,
                    "platform": platform,
                    "problem_statement": question,
                    "completion_index": completion_idx,
                    "model_output": code or "",
                    "problem_obj": p,
                }
            )

    total = len(flat_rows)
    all_scores: list[float] = []
    details: list[dict] = []

    details_fh = None
    if include_details and debug_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(debug_output_path)), exist_ok=True)
        details_fh = open(debug_output_path, "w")

    if total:
        logger.info(
            f"Scoring {total} completions in parallel (chunk_size={score_chunk_size})..."
        )
        score_start = time.time()
        score_pbar = tqdm(total=total, desc="Scoring", unit="completion")

        for start in range(0, total, score_chunk_size):
            end = min(start + score_chunk_size, total)
            chunk = flat_rows[start:end]
            chunk_codes = [r["model_output"] for r in chunk]
            chunk_problems = [r["problem_obj"] for r in chunk]
            chunk_scores, _ = exec_score_batch(codes=chunk_codes, problems=chunk_problems)
            all_scores.extend(chunk_scores)

            if include_details:
                for row, score in zip(chunk, chunk_scores):
                    detail_row = {
                        "question_id": row["question_id"],
                        "difficulty": row["difficulty"],
                        "platform": row["platform"],
                        "completion_index": row["completion_index"],
                        "execution_score": float(score),
                        "is_correct": bool(score >= 0.99),
                        "problem_statement": row["problem_statement"],
                        "model_output": row["model_output"],
                    }
                    if details_fh is not None:
                        details_fh.write(json.dumps(detail_row) + "\n")
                    else:
                        details.append(detail_row)

            score_pbar.update(end - start)
            done = end
            elapsed = time.time() - score_start
            eta = ((elapsed / done) * (total - done)) if done > 0 else 0.0
            logger.info(
                f"Scoring progress: {done}/{total} completions | "
                f"elapsed={_fmt_seconds(elapsed)} | eta={_fmt_seconds(eta)}"
            )

        score_pbar.close()

    if details_fh is not None:
        details_fh.close()

    # Reconstruct per-problem (n, c) — tracked by difficulty and platform.
    problem_stats: dict[int, dict] = {}
    for row, score in zip(flat_rows, all_scores):
        p_idx = row["problem_idx"]
        st = problem_stats.setdefault(
            p_idx,
            {
                "n": 0,
                "c": 0,
                "difficulty": row["difficulty"],
                "platform": row["platform"],
            },
        )
        st["n"] += 1
        if score >= 0.99:
            st["c"] += 1

    buckets: dict[str, list[dict]] = {
        "all": [],
        "easy": [],
        "medium": [],
        "hard": [],
        "leetcode": [],
        "atcoder": [],
    }
    for st in problem_stats.values():
        entry = {"n": st["n"], "c": st["c"]}
        buckets["all"].append(entry)
        if st["difficulty"] in buckets:
            buckets[st["difficulty"]].append(entry)
        if st["platform"] in buckets:
            buckets[st["platform"]].append(entry)

    # Compute pass@k for each bucket
    metrics = {}
    counts = {}
    for bucket_name, entries in buckets.items():
        valid = [e for e in entries if e["n"] > 0]
        counts[bucket_name] = len(valid)
        for k in k_values:
            eligible = [e for e in valid if e["n"] >= k]
            if eligible:
                scores = [pass_at_k(e["n"], e["c"], k) for e in eligible]
                metrics[f"pass@{k}/{bucket_name}"] = round(float(np.mean(scores)), 4)

    return metrics, counts, details


def save_eval_details_jsonl(details: list[dict], path: str) -> None:
    """Save per-completion eval details to JSONL for debugging."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--eval-data", type=str, default=LCB_EVAL_PATH,
                        help="Path to eval JSONL")
    parser.add_argument("--n-generations", type=int, default=EVAL_N_GENERATIONS)
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=EVAL_TEMPERATURE)
    parser.add_argument("--k-values", type=int, nargs="+", default=EVAL_K_VALUES)
    parser.add_argument("--score-chunk-size", type=int, default=128)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results/eval")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--save-debug-details",
        action="store_true",
        help="Save per-completion debug JSONL (problem statement, model output, execution score)",
    )
    parser.add_argument(
        "--debug-output",
        type=str,
        default="",
        help="Path for debug JSONL; default is results/eval/<timestamp_model>_details.jsonl",
    )
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = _slugify_model_name(args.model)
    run_name = args.run_name or f"{run_ts}_{model_slug}"
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = args.output or os.path.join(args.output_dir, f"{run_name}_summary.json")
    debug_path_default = os.path.join(args.output_dir, f"{run_name}_details.jsonl")
    debug_path = args.debug_output or debug_path_default

    logger.info(
        f"Eval artifacts directory: {os.path.abspath(args.output_dir)} | run_name={run_name}"
    )

    # Load problems
    problems = load_eval_problems(args.eval_data)
    logger.info(f"Loaded {len(problems)} evaluation problems")

    if args.smoke_test:
        problems = problems[:5]
        args.n_generations = 2
        args.max_tokens = 512
        logger.info("=== SMOKE TEST: 5 problems, 2 generations ===")

    eval_start = time.time()

    # Generate solutions
    solutions = generate_solutions(
        model_path=args.model,
        problems=problems,
        n_generations=args.n_generations,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Evaluate
    metrics, counts, details = evaluate(
        problems,
        solutions,
        args.k_values,
        include_details=args.save_debug_details,
        debug_output_path=debug_path if args.save_debug_details else "",
        score_chunk_size=max(1, int(args.score_chunk_size)),
    )

    # Pretty-print results table
    k_vals = sorted(set(int(k.split("@")[1].split("/")[0]) for k in metrics))
    splits = [
        ("Overall",   "all"),
        ("Easy",      "easy"),
        ("Medium",    "medium"),
        ("Hard",      "hard"),
        ("LeetCode",  "leetcode"),
        ("AtCoder",   "atcoder"),
    ]

    header = f"{'Split':<12} {'Problems':>9}" + "".join(f"  {'pass@'+str(k):>8}" for k in k_vals)
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print(f"  EVALUATION RESULTS — {args.model}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for label, key in splits:
        n = counts.get(key, 0)
        if n == 0:
            continue
        row = f"{label:<12} {n:>9}"
        for k in k_vals:
            val = metrics.get(f"pass@{k}/{key}")
            row += f"  {val:>8.4f}" if val is not None else f"  {'—':>8}"
        print(row)
    print(f"{'=' * len(header)}\n")

    # Save results — auto-create output directory
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    output = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "n_problems": len(problems),
        "n_generations": args.n_generations,
        "problem_counts": counts,
        "metrics": metrics,
        "runtime_s": round(time.time() - eval_start, 2),
    }
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {summary_path}")

    if args.save_debug_details:
        if details:
            # Fallback path when details were not streamed during evaluate().
            save_eval_details_jsonl(details, debug_path)
            logger.info(f"Debug details saved to {debug_path} ({len(details)} rows)")
        else:
            logger.info(f"Debug details streamed to {debug_path}")


if __name__ == "__main__":
    main()
