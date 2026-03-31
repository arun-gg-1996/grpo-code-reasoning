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
import sys
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

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    problems: list[dict],
    solutions: dict[str, list[str]],
    k_values: list[int],
) -> dict:
    """
    Evaluate solutions against test cases using parallel batch scoring.
    Returns pass@k scores broken down by: overall, difficulty, and platform.
    """
    # Flatten all (code, problem) pairs for a single parallel batch call.
    flat_codes: list[str] = []
    flat_problems: list[dict] = []
    for p in problems:
        pid = p.get("question_id") or p.get("problem_id", "unknown")
        for code in solutions.get(pid, []):
            flat_codes.append(code or "")
            flat_problems.append(p)

    if flat_codes:
        logger.info(f"Scoring {len(flat_codes)} completions in parallel...")
        all_scores, _ = exec_score_batch(codes=flat_codes, problems=flat_problems)
    else:
        all_scores = []

    # Reconstruct per-problem (n, c) — tracked by difficulty and platform
    buckets: dict[str, list[dict]] = {
        "all": [],
        "easy": [], "medium": [], "hard": [],
        "leetcode": [], "atcoder": [],
    }
    idx = 0
    for p in tqdm(problems, desc="Aggregating results"):
        pid = p.get("question_id") or p.get("problem_id", "unknown")
        diff = p.get("difficulty", "medium")
        platform = p.get("platform", "unknown").lower()
        codes = solutions.get(pid, [])
        n = len(codes)
        c = sum(1 for s in all_scores[idx:idx + n] if s >= 0.99)
        idx += n
        if n > 0:
            entry = {"n": n, "c": c}
            buckets["all"].append(entry)
            if diff in buckets:
                buckets[diff].append(entry)
            if platform in buckets:
                buckets[platform].append(entry)

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

    return metrics, counts


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
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    # Load problems
    problems = load_eval_problems(args.eval_data)
    logger.info(f"Loaded {len(problems)} evaluation problems")

    if args.smoke_test:
        problems = problems[:5]
        args.n_generations = 2
        args.max_tokens = 512
        logger.info("=== SMOKE TEST: 5 problems, 2 generations ===")

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
    metrics, counts = evaluate(problems, solutions, args.k_values)

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
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_problems": len(problems),
        "n_generations": args.n_generations,
        "problem_counts": counts,
        "metrics": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
