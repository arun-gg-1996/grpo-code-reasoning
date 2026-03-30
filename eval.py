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
from reward.execution import score_single, extract_code

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

    # Generate in batches
    logger.info(f"Generating {n_generations} solutions for {len(problems)} problems...")
    all_outputs = llm.generate(prompts, sampling_params)

    results = {}
    for pid, output in zip(problem_ids, all_outputs):
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
    Evaluate solutions against test cases.
    Returns pass@k scores per difficulty and overall.
    """
    results_by_diff = {"easy": [], "medium": [], "hard": [], "all": []}

    for p in tqdm(problems, desc="Evaluating"):
        pid = p.get("question_id") or p.get("problem_id", "unknown")
        diff = p.get("difficulty", "medium")
        codes = solutions.get(pid, [])

        if not codes:
            for k in k_values:
                results_by_diff[diff].append({"n": 0, "c": 0})
                results_by_diff["all"].append({"n": 0, "c": 0})
            continue

        # Score each generation
        n = len(codes)
        c = 0
        for code in codes:
            if not code:
                continue
            score = score_single(code, p)
            if score >= 0.99:  # full pass
                c += 1

        results_by_diff[diff].append({"n": n, "c": c})
        results_by_diff["all"].append({"n": n, "c": c})

    # Compute pass@k
    metrics = {}
    for diff, entries in results_by_diff.items():
        valid_entries = [e for e in entries if e["n"] > 0]
        if not valid_entries:
            continue

        for k in k_values:
            scores = [
                pass_at_k(e["n"], e["c"], k)
                for e in valid_entries
                if e["n"] >= k
            ]
            if scores:
                metrics[f"pass@{k}_{diff}"] = np.mean(scores)

    return metrics


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
    metrics = evaluate(problems, solutions, args.k_values)

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for key, val in sorted(metrics.items()):
        logger.info(f"  {key}: {val:.4f}")
    logger.info("=" * 60)

    # Save results
    output = {
        "model": args.model,
        "n_problems": len(problems),
        "n_generations": args.n_generations,
        "metrics": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
