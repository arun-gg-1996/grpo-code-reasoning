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
import time
from datetime import datetime

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
    ATTN_IMPLEMENTATION,
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
    VLLM_MODE,
    TRAIN_SEED,
    GENERATION_BATCH_SIZE,
    VLLM_ENABLE_SLEEP_MODE,
    STRICT_TRAIN_CODE_TAGS,
    APPS_CLEAN_PATH,
    LCB_SEEN_PATH,
    SAVE_STEPS,
    LOGGING_STEPS,
    WANDB_PROJECT,
    PUSH_TO_HUB,
    HUB_MODEL_ID,
    get_curriculum_weights,
    normalize_difficulty,
)
from problem_format import is_function_style_problem, get_function_name
from reward.reward import reward_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None


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
    is_func_style = is_function_style_problem(problem)
    func_name = get_function_name(problem)

    if is_func_style:
        fn_line = f"- Implement the expected function `{func_name}` exactly.\n" if func_name else ""
        format_hint = (
            "Output format requirement (strict):\n"
            "- Output exactly one <think>...</think> block, then exactly one <code>...</code> block.\n"
            "- Do not output any text before <think> or after </code>.\n"
            "- In <think>, the first [STEP] must decide execution format and signature for this problem.\n"
            "- Use as many [STEP] blocks as needed; keep them concise and avoid filler/repetition.\n"
            "- This problem is function-style.\n"
            "- In <code>...</code>, provide only the function/class implementation expected by the prompt.\n"
            f"{fn_line}"
            "- Do NOT read from stdin and do NOT print to stdout unless explicitly required by the statement.\n"
            "- Never leave <code> empty.\n"
            "- Inside <code>, output raw Python only (no triple backticks)."
        )
    else:
        format_hint = (
            "Output format requirement (strict):\n"
            "- Output exactly one <think>...</think> block, then exactly one <code>...</code> block.\n"
            "- Do not output any text before <think> or after </code>.\n"
            "- In <think>, the first [STEP] must decide execution format and I/O handling for this problem.\n"
            "- Use as many [STEP] blocks as needed; keep them concise and avoid filler/repetition.\n"
            "- This problem is stdin/stdout style.\n"
            "- In <code>...</code>, provide a full program that reads input from stdin and prints output.\n"
            "- Never leave <code> empty.\n"
            "- Inside <code>, output raw Python only (no triple backticks)."
        )

    return f"{TRAINING_SYSTEM_PROMPT}\n\n{format_hint}\n\n{question}"


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

def make_reward_fn():
    """
    Create the reward function closure for GRPOTrainer.
    GRPOTrainer calls: reward_fn(completions, prompts, **kwargs)
    Problem metadata is passed through kwargs via the dataset columns.
    """
    def _reward(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
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
# Timed trainer wrapper (step timing + system metrics for wandb)
# ---------------------------------------------------------------------------

def _bytes_to_gb(n_bytes: float) -> float:
    return float(n_bytes) / (1024 ** 3)


class TimedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase_timing = {
            "timing/generation_s": 0.0,
            "timing/reward_calc_s": 0.0,
            "timing/loss_compute_s": 0.0,
            "timing/training_step_s": 0.0,
        }
        self._step_wall_start = time.perf_counter()
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._model_param_bytes = self._estimate_model_param_bytes()
        self._nvml_handle = None
        self._nvml_ready = False
        if torch.cuda.is_available() and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_ready = True
            except Exception:
                self._nvml_ready = False

    def _estimate_model_param_bytes(self) -> float:
        total = 0
        for p in self.model.parameters():
            total += p.numel() * p.element_size()
        return float(total)

    def _estimate_optimizer_state_bytes(self) -> float | None:
        if getattr(self, "optimizer", None) is None:
            return None
        total = 0
        for state in self.optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        return float(total)

    def _collect_system_metrics(self) -> dict:
        metrics = {}
        if self._process is not None:
            try:
                metrics["system/cpu_percent"] = float(self._process.cpu_percent(interval=None))
                rss = float(self._process.memory_info().rss)
                metrics["system/process_rss_gb"] = _bytes_to_gb(rss)
                vm = psutil.virtual_memory()
                metrics["system/ram_used_gb"] = _bytes_to_gb(float(vm.used))
                metrics["system/ram_percent"] = float(vm.percent)
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                allocated = float(torch.cuda.memory_allocated())
                reserved = float(torch.cuda.memory_reserved())
                metrics["gpu/torch_allocated_gb"] = _bytes_to_gb(allocated)
                metrics["gpu/torch_reserved_gb"] = _bytes_to_gb(reserved)
                metrics["gpu/torch_max_allocated_gb"] = _bytes_to_gb(float(torch.cuda.max_memory_allocated()))
                metrics["gpu/torch_max_reserved_gb"] = _bytes_to_gb(float(torch.cuda.max_memory_reserved()))
            except Exception:
                pass

            if self._nvml_ready and self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    metrics["gpu/utilization_percent"] = float(util.gpu)
                    metrics["gpu/nvml_mem_used_gb"] = _bytes_to_gb(float(mem.used))
                    metrics["gpu/nvml_mem_total_gb"] = _bytes_to_gb(float(mem.total))
                    # Approximate memory not tracked by torch (e.g., vLLM/runtime allocations)
                    if "gpu/torch_reserved_gb" in metrics:
                        metrics["gpu/non_torch_mem_gb_est"] = max(
                            0.0,
                            metrics["gpu/nvml_mem_used_gb"] - metrics["gpu/torch_reserved_gb"],
                        )
                except Exception:
                    pass

        return metrics

    def _flush_step_metrics(self):
        if wandb is None or wandb.run is None:
            self._phase_timing = {k: 0.0 for k in self._phase_timing}
            self._step_wall_start = time.perf_counter()
            return

        step_total_s = time.perf_counter() - self._step_wall_start
        log_dict = {
            "timing/step_total_s": float(step_total_s),
            **{k: float(v) for k, v in self._phase_timing.items()},
            "gpu/model_param_gb_est": _bytes_to_gb(self._model_param_bytes),
        }

        opt_bytes = self._estimate_optimizer_state_bytes()
        if opt_bytes is not None:
            log_dict["gpu/optimizer_state_gb_est"] = _bytes_to_gb(opt_bytes)

        log_dict.update(self._collect_system_metrics())
        wandb.log(log_dict)

        self._phase_timing = {k: 0.0 for k in self._phase_timing}
        self._step_wall_start = time.perf_counter()

    def _generate(self, prompts: list):
        t0 = time.perf_counter()
        out = super()._generate(prompts)
        self._phase_timing["timing/generation_s"] += time.perf_counter() - t0
        return out

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        t0 = time.perf_counter()
        out = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._phase_timing["timing/reward_calc_s"] += time.perf_counter() - t0
        return out

    def _compute_loss(self, model, inputs):
        t0 = time.perf_counter()
        out = super()._compute_loss(model, inputs)
        self._phase_timing["timing/loss_compute_s"] += time.perf_counter() - t0
        return out

    def training_step(self, model, inputs, num_items_in_batch):
        t0 = time.perf_counter()
        out = super().training_step(model, inputs, num_items_in_batch)
        self._phase_timing["timing/training_step_s"] += time.perf_counter() - t0
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._flush_step_metrics()
        return out

    def _save_checkpoint(self, model, trial):
        t0 = time.perf_counter()
        out = super()._save_checkpoint(model, trial)
        dt = time.perf_counter() - t0
        if wandb is not None and wandb.run is not None:
            wandb.log({
                "event/checkpointing": 1,
                "event/checkpoint_save_s": float(dt),
            })
        return out

    def _push_from_checkpoint(self, checkpoint_folder: str):
        t0 = time.perf_counter()
        out = super()._push_from_checkpoint(checkpoint_folder)
        dt = time.perf_counter() - t0
        if wandb is not None and wandb.run is not None:
            wandb.log({
                "event/hf_push_s": float(dt),
            })
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick local smoke test with 1.5B model")
    parser.add_argument(
        "--save-debug-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream per-completion training debug rows to timestamped JSONL",
    )
    parser.add_argument(
        "--train-artifacts-dir",
        type=str,
        default="results/train",
        help="Base directory for timestamped training artifacts",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=None,
        help="Override generation temperature for GRPO rollouts.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="Override vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max completion tokens used during training generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device train batch size.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume from, or 'latest' to auto-resume latest in output_dir.",
    )
    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke_test:
        model_name = LOCAL_TRAINING_MODEL
        max_steps = 2
        batch_size = 2
        group_size = 4
        save_steps = 999
        vllm_mem = 0.3
        rollout_temperature = ROLLOUT_TEMPERATURE
        max_new_tokens = 512
        use_wandb = False
        smoke_use_gpu = torch.cuda.is_available()
        smoke_use_vllm = smoke_use_gpu
        logger.info("=== SMOKE TEST MODE ===")
        logger.info(
            "Smoke test device: %s | vLLM: %s",
            "cuda" if smoke_use_gpu else "cpu",
            "enabled" if smoke_use_vllm else "disabled",
        )
    else:
        model_name = TRAINING_MODEL
        max_steps = MAX_TRAINING_STEPS
        batch_size = BATCH_SIZE
        group_size = GROUP_SIZE
        save_steps = SAVE_STEPS
        vllm_mem = VLLM_GPU_MEMORY_UTILIZATION
        rollout_temperature = ROLLOUT_TEMPERATURE
        max_new_tokens = MAX_NEW_TOKENS
        use_wandb = True

    # Optional runtime overrides for tuning sweeps.
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        batch_size = args.batch_size
    if args.rollout_temperature is not None:
        if args.rollout_temperature <= 0:
            raise ValueError("--rollout-temperature must be > 0")
        rollout_temperature = args.rollout_temperature
    if args.vllm_gpu_memory_utilization is not None:
        if not (0.05 <= args.vllm_gpu_memory_utilization <= 0.95):
            raise ValueError("--vllm-gpu-memory-utilization must be in [0.05, 0.95]")
        vllm_mem = args.vllm_gpu_memory_utilization
    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be > 0")
        max_new_tokens = args.max_new_tokens

    # Derived runtime knobs (must follow CLI overrides above).
    grad_accum_steps = GRADIENT_ACCUMULATION_STEPS if not args.smoke_test else 1
    effective_vllm_max_model_length = MAX_PROMPT_LENGTH + max_new_tokens

    # Reproducibility for curriculum prebuild order and sampling.
    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_SEED)

    # Timestamped train artifacts (similar to eval run folders)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_run_dir = os.path.join(args.train_artifacts_dir, run_ts)
    os.makedirs(train_run_dir, exist_ok=True)
    train_debug_path = os.path.join(train_run_dir, "train_details.jsonl")
    train_summary_path = os.path.join(train_run_dir, "summary.json")

    # Expose debug path to reward_fn via environment variables.
    if args.save_debug_details and (not args.smoke_test):
        os.environ["TRAIN_SAVE_DEBUG_DETAILS"] = "1"
        os.environ["TRAIN_DEBUG_DETAILS_PATH"] = train_debug_path
        logger.info(f"Train artifacts directory: {os.path.abspath(train_run_dir)}")
        logger.info(f"Streaming training debug details to {os.path.abspath(train_debug_path)}")
    else:
        os.environ["TRAIN_SAVE_DEBUG_DETAILS"] = "0"
        os.environ.pop("TRAIN_DEBUG_DETAILS_PATH", None)

    # Init wandb
    if use_wandb and wandb is not None:
        wandb.init(project=WANDB_PROJECT, config={
            "model": model_name,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "group_size": group_size,
            "generation_batch_size": min(GENERATION_BATCH_SIZE, batch_size * group_size),
            "lr": LEARNING_RATE,
            "kl_coeff": KL_COEFF,
            "rollout_temperature": rollout_temperature,
            "max_new_tokens": max_new_tokens,
            "vllm_gpu_memory_utilization": vllm_mem,
            "vllm_mode": VLLM_MODE,
            "vllm_max_model_length": effective_vllm_max_model_length,
            "vllm_enable_sleep_mode": VLLM_ENABLE_SLEEP_MODE,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "lora_rank": LORA_RANK,
            "train_seed": TRAIN_SEED,
            "strict_train_code_tags": STRICT_TRAIN_CODE_TAGS,
            "resume_from_checkpoint": args.resume_from_checkpoint,
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

    # Pre-build curriculum dataset in optimizer-step order.
    # One row group per optimizer step. With shuffle_dataset=False, the trainer
    # consumes rows in this exact order, so curriculum "step" matches global_step.
    logger.info("Building curriculum-weighted dataset...")
    all_sampled = []
    for step in range(max_steps):
        batch = sample_batch(problems_by_diff, step, batch_size)
        all_sampled.extend(batch)
    dataset = problems_to_dataset(all_sampled)

    # Log curriculum distribution that was baked in
    diff_counts: dict[str, int] = {}
    src_counts: dict[str, int] = {}
    for p in all_sampled:
        d = p["difficulty"]
        s = p.get("source", "unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1
        src_counts[s] = src_counts.get(s, 0) + 1
    logger.info(
        "Dataset: %d rows (%d optimizer steps × %d problems/step)",
        len(dataset), max_steps, batch_size
    )
    for d, c in sorted(diff_counts.items()):
        logger.info(f"  difficulty: {d}={c} ({c / len(all_sampled):.1%})")
    for s, c in sorted(src_counts.items()):
        logger.info(f"  source: {s}={c} ({c / len(all_sampled):.1%})")

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
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS if not args.smoke_test else 2,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=max_new_tokens,
        num_generations=group_size,
        generation_batch_size=min(GENERATION_BATCH_SIZE, batch_size * group_size),
        temperature=rollout_temperature,
        beta=KL_COEFF,
        logging_steps=LOGGING_STEPS,
        save_steps=save_steps,
        seed=TRAIN_SEED,
        report_to="wandb" if use_wandb else "none",
        gradient_checkpointing=not args.smoke_test,
        bf16=(not args.smoke_test) or (args.smoke_test and smoke_use_gpu),
        use_cpu=args.smoke_test and (not smoke_use_gpu),
        model_init_kwargs=(
            {"torch_dtype": "float32", "device_map": "cpu"}
            if (args.smoke_test and (not smoke_use_gpu))
            else (
                {"attn_implementation": ATTN_IMPLEMENTATION, "torch_dtype": torch.bfloat16}
                if not args.smoke_test
                else None
            )
        ),
        use_vllm=(not args.smoke_test) or (args.smoke_test and smoke_use_vllm),
        vllm_mode=VLLM_MODE,
        vllm_gpu_memory_utilization=vllm_mem,
        vllm_max_model_length=effective_vllm_max_model_length,
        vllm_enable_sleep_mode=VLLM_ENABLE_SLEEP_MODE,
        shuffle_dataset=False,
        push_to_hub=PUSH_TO_HUB and not args.smoke_test,
        hub_model_id=HUB_MODEL_ID if (PUSH_TO_HUB and not args.smoke_test) else None,
    )

    # Build reward function
    reward = make_reward_fn()

    # Initialize trainer
    logger.info("Initializing GRPOTrainer...")
    trainer = TimedGRPOTrainer(
        model=model_name,
        reward_funcs=reward,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting GRPO training...")
    run_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": os.path.abspath(train_run_dir),
        "status": "running",
        "model": model_name,
        "max_steps": int(max_steps),
        "batch_size": int(batch_size),
        "group_size": int(group_size),
        "generation_batch_size": int(min(GENERATION_BATCH_SIZE, batch_size * group_size)),
        "rollout_temperature": float(rollout_temperature),
        "max_new_tokens": int(max_new_tokens),
        "train_seed": int(TRAIN_SEED),
        "vllm_gpu_memory_utilization": float(vllm_mem),
        "vllm_mode": VLLM_MODE,
        "vllm_max_model_length": int(effective_vllm_max_model_length),
        "vllm_enable_sleep_mode": bool(VLLM_ENABLE_SLEEP_MODE),
        "attn_implementation": ATTN_IMPLEMENTATION,
        "strict_train_code_tags": bool(STRICT_TRAIN_CODE_TAGS),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "save_debug_details": bool(args.save_debug_details and (not args.smoke_test)),
        "train_debug_details_path": (
            os.path.abspath(train_debug_path)
            if (args.save_debug_details and (not args.smoke_test))
            else None
        ),
    }
    with open(train_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    try:
        resume_arg = args.resume_from_checkpoint
        if resume_arg is not None and resume_arg.strip().lower() == "latest":
            resume_arg = True
        if resume_arg is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=resume_arg)
        run_summary["status"] = "completed"
    except Exception as e:
        run_summary["status"] = "failed"
        run_summary["error"] = str(e)
        with open(train_summary_path, "w") as f:
            json.dump(run_summary, f, indent=2)
        raise

    # Save final checkpoint
    logger.info("Saving final model...")
    trainer.save_model()

    run_summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(train_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    if use_wandb and wandb is not None:
        wandb.finish()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
