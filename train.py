"""
train.py

GRPO training loop for Qwen2.5-Coder-7B-Instruct using TRL's GRPOTrainer.
Single GPU (A100 80GB). LoRA fine-tuning.

Usage:
    # Full training (cloud)
    python train.py

    # Local smoke test (1.5B model, 2 steps)
    python train.py --smoke-test
"""

import argparse
import json
import logging
import random
import sys
import os

sys.set_int_max_str_digits(0)  # APPS data contains very large integers in JSON fields

import torch

try:
    import wandb
except ImportError:
    wandb = None

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from config import (
    TRAINING_MODEL,
    LOCAL_TRAINING_MODEL,
    TRAINING_SYSTEM_PROMPT,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    GROUP_SIZE,
    BATCH_SIZE,
    MAX_NEW_TOKENS,
    MAX_PROMPT_LENGTH,
    ROLLOUT_TEMPERATURE,
    LEARNING_RATE,
    KL_COEFF,
    WARMUP_STEPS,
    MAX_TRAINING_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    VLLM_GPU_MEMORY_UTILIZATION,
    APPS_CLEAN_PATH,
    LCB_SEEN_PATH,
    SAVE_STEPS,
    LOGGING_STEPS,
    WANDB_PROJECT,
    CURRICULUM,
    get_curriculum_weights,
    normalize_difficulty,
)
from reward.reward import reward_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_test_cases(problem: dict) -> dict:
    """
    Normalize APPS io format to standard test_cases/stdin_tests format.
    APPS uses: {"io": {"inputs": [...], "outputs": [...]}}
    LCB uses:  {"test_cases": [{"input": ..., "output": ...}], "stdin_tests": [...]}
    """
    if problem.get("test_cases"):
        return problem  # already in LCB format

    io = problem.get("io")
    if io and isinstance(io, dict):
        inputs = io.get("inputs", [])
        outputs = io.get("outputs", [])
        tc = [
            {"input": inp, "output": out}
            for inp, out in zip(inputs, outputs)
        ]
        problem["test_cases"] = tc
        problem["stdin_tests"] = tc
    return problem


def load_problems(path: str, source_label: str) -> list[dict]:
    """Load JSONL problems and tag with source."""
    problems = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            p["source"] = p.get("source", source_label)
            p["difficulty"] = normalize_difficulty(p.get("difficulty", "medium"))
            p = _normalize_test_cases(p)
            problems.append(p)
    return problems


def build_prompt(problem: dict) -> str:
    """Build the training prompt for a problem."""
    question = problem.get("question", "")
    return f"{TRAINING_SYSTEM_PROMPT}\n\n{question}"


def sample_batch(
    problems_by_diff: dict[str, list[dict]],
    step: int,
    batch_size: int,
) -> list[dict]:
    """
    Sample a batch of problems using curriculum weights for the current step.
    Each problem is repeated GROUP_SIZE times (GRPO rollouts).
    """
    weights = get_curriculum_weights(step)

    sampled = []
    for _ in range(batch_size):
        # Weighted random difficulty selection
        diffs = list(weights.keys())
        probs = [weights[d] for d in diffs]

        # Filter to difficulties that have problems
        valid = [(d, p) for d, p in zip(diffs, probs) if problems_by_diff.get(d)]
        if not valid:
            continue
        diffs, probs = zip(*valid)
        total = sum(probs)
        probs = [p / total for p in probs]

        diff = random.choices(diffs, weights=probs, k=1)[0]
        problem = random.choice(problems_by_diff[diff])
        sampled.append(problem)

    return sampled


def problems_to_dataset(problems: list[dict]) -> Dataset:
    """
    Convert sampled problems to a HuggingFace Dataset for GRPOTrainer.
    Each problem becomes one row; GRPOTrainer generates GROUP_SIZE rollouts per row.
    """
    rows = []
    for p in problems:
        rows.append({
            "prompt": build_prompt(p),
            # Metadata passed through to reward_fn via kwargs
            "problem_json": json.dumps(p),
            "source": p.get("source", "apps"),
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward wrapper for GRPOTrainer
# ---------------------------------------------------------------------------

def make_reward_fn(all_problems_map: dict):
    """
    Create a reward function closure that GRPOTrainer can call.

    GRPOTrainer calls: reward_fn(completions, prompts, **kwargs)
    where completions and prompts are lists of strings.
    """
    def _reward(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        # Reconstruct problem dicts from the metadata
        problem_jsons = kwargs.get("problem_json", [])
        sources = kwargs.get("source", [])

        problems = []
        for pj in problem_jsons:
            try:
                problems.append(json.loads(pj))
            except (json.JSONDecodeError, TypeError):
                problems.append({})

        return reward_fn(
            completions=completions,
            prompts=prompts,
            problems=problems,
            source=sources,
        )

    return _reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick local smoke test with 1.5B model")
    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke_test:
        model_name = LOCAL_TRAINING_MODEL
        max_steps = 2
        batch_size = 2
        group_size = 4
        save_steps = 999
        vllm_mem = 0.3
        use_wandb = False
        logger.info("=== SMOKE TEST MODE ===")
    else:
        model_name = TRAINING_MODEL
        max_steps = MAX_TRAINING_STEPS
        batch_size = BATCH_SIZE
        group_size = GROUP_SIZE
        save_steps = SAVE_STEPS
        vllm_mem = VLLM_GPU_MEMORY_UTILIZATION
        use_wandb = True

    # Init wandb
    if use_wandb and wandb is not None:
        wandb.init(project=WANDB_PROJECT, config={
            "model": model_name,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "group_size": group_size,
            "lr": LEARNING_RATE,
            "kl_coeff": KL_COEFF,
            "lora_rank": LORA_RANK,
        })

    # Load data
    logger.info("Loading training data...")
    apps_problems = load_problems(APPS_CLEAN_PATH, "apps")
    lcb_problems = load_problems(LCB_SEEN_PATH, "lcb_seen")
    all_problems = apps_problems + lcb_problems

    logger.info(f"APPS: {len(apps_problems)} problems, LCB: {len(lcb_problems)} problems")

    # Group by difficulty
    problems_by_diff = {"easy": [], "medium": [], "hard": []}
    for p in all_problems:
        d = p["difficulty"]
        if d in problems_by_diff:
            problems_by_diff[d].append(p)

    for d, ps in problems_by_diff.items():
        logger.info(f"  {d}: {len(ps)} problems")

    # Build a full dataset for GRPOTrainer (it handles sampling internally)
    # We provide all problems; curriculum sampling happens via the dataset
    dataset = problems_to_dataset(all_problems)

    # If smoke test, trim dataset
    if args.smoke_test:
        dataset = dataset.select(range(min(20, len(dataset))))

    logger.info(f"Dataset size: {len(dataset)} problems")

    # Load tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    # GRPO training config
    training_args = GRPOConfig(
        output_dir=f"checkpoints/{model_name.split('/')[-1]}-grpo",
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS if not args.smoke_test else 1,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS if not args.smoke_test else 2,
        max_completion_length=MAX_NEW_TOKENS if not args.smoke_test else 512,
        num_generations=group_size,
        generation_batch_size=batch_size * group_size,
        temperature=ROLLOUT_TEMPERATURE,
        beta=KL_COEFF,
        logging_steps=LOGGING_STEPS,
        save_steps=save_steps,
        report_to="wandb" if use_wandb else "none",
        bf16=not args.smoke_test,
        use_cpu=args.smoke_test,
        model_init_kwargs={"torch_dtype": "float32"} if args.smoke_test else None,
        use_vllm=not args.smoke_test,
        vllm_gpu_memory_utilization=vllm_mem,
        # Disable pushing during training
        push_to_hub=False,
    )

    # Build reward function
    all_problems_map = {
        p.get("problem_id") or p.get("question_id", ""): p
        for p in all_problems
    }
    reward = make_reward_fn(all_problems_map)

    # Initialize trainer
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final checkpoint
    logger.info("Saving final model...")
    trainer.save_model()

    if use_wandb and wandb is not None:
        wandb.finish()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
