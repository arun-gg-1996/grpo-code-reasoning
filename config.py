"""
config.py

Single source of truth for all configuration.
All prompts, weights, timeouts, and hyperparameters live here.
Everything else imports from this file — never hardcode values elsewhere.
"""

# ─────────────────────────────────────────
# Model config
# ─────────────────────────────────────────

TRAINING_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
LOCAL_TRAINING_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # for local smoke test
ATTN_IMPLEMENTATION = "sdpa"  # non-Flash attention backend for stable training without flash-attn build

# Judge — Gemini via API key auth (aiplatform.googleapis.com)
JUDGE_MODEL = "gemini-2.5-flash-lite"
# API key — loaded from .env file (key name: api_key)
import os
from dotenv import load_dotenv
load_dotenv()
# Primary key name in .env is gemini_api_key.
# Keep api_key as backward-compatible fallback.
GEMINI_API_KEY = os.environ.get("gemini_api_key", os.environ.get("api_key", ""))

# ─────────────────────────────────────────
# GRPO hyperparameters
# ─────────────────────────────────────────

G = 8  # rollouts per problem (GROUP_SIZE alias below)
GROUP_SIZE = G  # alias — used throughout reward.py and logging
BATCH_SIZE = 4  # problems per training step → 4 * 8 = 32 completions per step
ROLLOUT_TEMPERATURE = 0.9
EVAL_TEMPERATURE = 0.2
MAX_NEW_TOKENS = 1792
MAX_PROMPT_LENGTH = 1024

# ─────────────────────────────────────────
# Training hyperparameters
# Set from literature before cloud run — do not leave None
# ─────────────────────────────────────────

LEARNING_RATE = 2e-5  # increased to boost policy movement/clip engagement
KL_COEFF = 0.04  # KL penalty — controls drift from reference model
WARMUP_STEPS = 20  # shorter warmup for faster ramp to effective LR
MAX_TRAINING_STEPS = 2700  # extended run for better coverage with no-replacement sampling
GRADIENT_ACCUMULATION_STEPS = 8  # smoother effective updates without increasing VRAM much
VLLM_GPU_MEMORY_UTILIZATION = 0.25  # keep vLLM share stable with recent training runs
VLLM_MODE = "colocate"  # explicitly run colocated vLLM with trainer on single-GPU setup
TRAIN_SEED = 42  # deterministic curriculum pre-build sampling

# Generation/memory knobs for colocated vLLM.
# Keep per-step generations in smaller chunks to reduce peak memory.
GENERATION_BATCH_SIZE = 24
# vLLM KV cache cap: prompt + completion budget used by this project.
VLLM_MAX_MODEL_LENGTH = MAX_PROMPT_LENGTH + MAX_NEW_TOKENS
# In colocate mode, offload vLLM state during optimizer step to free VRAM headroom.
VLLM_ENABLE_SLEEP_MODE = True

# Format multipliers for training-time code extraction.
# strict: extracted from preferred ```python blocks
# fence:  extracted from legacy/secondary code block forms
FORMAT_PENALTY_STRICT = 1.00
FORMAT_PENALTY_FENCE = 1.00

# ─────────────────────────────────────────
# LoRA config
# ─────────────────────────────────────────

LORA_RANK = 64
LORA_ALPHA = 128  # keep alpha at 2x rank
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",  # attention layers
    "gate_proj", "up_proj", "down_proj"  # MLP layers
]
QUANTIZATION = None  # full precision LoRA — A100 80GB has headroom

# ─────────────────────────────────────────
# Reward weights
# ─────────────────────────────────────────

# Per-source reward weights
APPS_EXEC_WEIGHT = 0.75
APPS_REASONING_WEIGHT = 0.25
LCB_EXEC_WEIGHT = 0.60
LCB_REASONING_WEIGHT = 0.40

# Legacy tier-weight constants (kept for compatibility/tests).
# Current reward policy is Gemini-first for all tiers with presence fallback on judge failure.
# Easy/medium/hard weights below are not used in the active reasoning score path.
EASY_GEMINI_WEIGHT = 0.3
EASY_PRESENCE_WEIGHT = 0.7
MEDIUM_GEMINI_WEIGHT = 0.7
MEDIUM_PRESENCE_WEIGHT = 0.3
HARD_GEMINI_WEIGHT = 0.3
HARD_PRESENCE_WEIGHT = 0.7

REWARD_STD_WARNING_THRESHOLD = 0.05  # reward_std below this → diversity collapse warning

# Universal W&B reference-line thresholds (stable across experiments).
TRAIN_KL_REF_FLOOR = 0.001
TRAIN_KL_REF_CEILING = 0.10
TRAIN_CLIP_RATIO_REF_MIN = 0.005
GRPO_ALL_ZERO_REF_MAX = 0.50
GRPO_ALL_PERFECT_REF_MAX = 0.50
# Dynamic baseline window: post-warmup first N optimizer steps.
DYNAMIC_BASELINE_WINDOW_STEPS = 30

# ─────────────────────────────────────────
# Gemini judge
# ─────────────────────────────────────────

# Max concurrent Gemini calls in one reward batch.
# Keep modest to reduce 429 rate-limit bursts.
GEMINI_MAX_WORKERS = 8
# Timeout per call (seconds).
GEMINI_TIMEOUT = 30
# Judge response is short; cap output tokens.
GEMINI_MAX_TOKENS = 512
# Retry policy for transient API failures (429/5xx/timeouts).
GEMINI_MAX_RETRIES = 5
GEMINI_RETRY_BASE_DELAY_S = 1.5
GEMINI_RETRY_MAX_DELAY_S = 30.0

# ─────────────────────────────────────────
# Execution sandbox
# ─────────────────────────────────────────

SANDBOX_MAX_WORKERS = 16  # persistent pool size for parallel sandbox execution
SANDBOX_TIMEOUT = 5  # seconds per subprocess execution before SIGKILL
SANDBOX_PER_TEST_TIMEOUT_S = 4  # per-test timeout inside checker
SANDBOX_MAX_MEMORY_MB = 2048  # per-subprocess memory cap via RLIMIT
MAX_TEST_CASES = 10  # test cases per problem during training (cap for speed)

# Aliases used by reward/execution.py (DO NOT REMOVE)
EXEC_TIMEOUT = SANDBOX_TIMEOUT
EXEC_WORKERS = SANDBOX_MAX_WORKERS

# ─────────────────────────────────────────
# Checkpointing and logging
# ─────────────────────────────────────────

SAVE_STEPS = 200
PUSH_TO_HUB = True
HUB_MODEL_ID = "arun-ghontale/grpo-qwen-coder"
LOGGING_STEPS = 1  # log every step — only ~2000 steps total, want full resolution
WANDB_PROJECT = "grpo-code-gen"

# ─────────────────────────────────────────
# Mid-training evaluation
# ─────────────────────────────────────────
MID_EVAL_ENABLED = True
# Dense early evals for rapid signal, then every 200 steps.
MID_EVAL_STEPS = frozenset([
    100, 200, 300,
    500, 700, 900, 1100, 1300, 1500,
    1700, 1900, 2100, 2300, 2500, 2700,
])
MID_EVAL_N_GENERATIONS = 5
MID_EVAL_CHECKPOINT_ROOT = "checkpoints/mid_eval"
MID_EVAL_RESULTS_ROOT = "results/mid_eval"
MID_EVAL_SMOKE_N_PROBLEMS = 5
MID_EVAL_SMOKE_N_GENERATIONS = 2

# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

EVAL_BATCH_SIZE = 8
EVAL_N_GENERATIONS = 5  # generations per problem for pass@k
EVAL_K_VALUES = [1, 3]  # pass@1 is primary metric, pass@3 for completeness

# ─────────────────────────────────────────
# Curriculum schedule
# Keyed off global_step — stateless, correct on resume from checkpoint
# ─────────────────────────────────────────

CURRICULUM = [
    (0, {"difficulty": {"easy": 0.70, "medium": 0.30, "hard": 0.00}}),
    (80, {"difficulty": {"easy": 0.45, "medium": 0.50, "hard": 0.05}}),
    (250, {"difficulty": {"easy": 0.30, "medium": 0.50, "hard": 0.20}}),
    (700, {"difficulty": {"easy": 0.15, "medium": 0.40, "hard": 0.45}}),
]



# ─────────────────────────────────────────
# Difficulty normalization
# ─────────────────────────────────────────
DIFFICULTY_MAP = {
    "introductory": "easy",
    "interview": "medium",
    "competition": "hard",
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


def normalize_difficulty(raw: str) -> str:
    """Normalize APPS/LCB difficulty to easy/medium/hard."""
    return DIFFICULTY_MAP.get(raw, "medium")


def get_curriculum_weights(step: int) -> dict:
    """Return difficulty sampling weights for the current training step."""
    weights = CURRICULUM[0][1]
    for from_step, w in CURRICULUM:
        if step >= from_step:
            weights = w
    return weights["difficulty"]


def get_curriculum_phase(step: int) -> int:
    """Return 0-based curriculum phase index for a training step."""
    phase = 0
    for idx, (from_step, _w) in enumerate(CURRICULUM):
        if step >= from_step:
            phase = idx
    return phase


# ─────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────

APPS_CLEAN_PATH = "data/clean/apps_clean.jsonl"
LCB_SEEN_PATH = "data/clean/lcb_seen_clean.jsonl"
TACO_CLEAN_PATH = "data/clean/taco_verified_clean.jsonl"
LCB_EVAL_PATH = "data/clean/lcb_unseen_clean.jsonl"  # held out — eval only, never train
FAILED_DIR = "data/failed"

# ─────────────────────────────────────────
# Training prompt template
# Model is instructed to reason inside <think> tags,
# then write code inside a ```python fenced block.
# Code is extracted in reward/execution.py.
# ─────────────────────────────────────────

TRAINING_SYSTEM_PROMPT = """You are an expert competitive programmer.
Solve the following problem step by step.

Output format:
1) Think through your solution inside <think>...</think> tags. Reason step by step (algorithm choice, data structures, edge cases, implementation plan). Be concise and avoid repetition.
2) Then write your complete Python solution in a ```python code block.
3) Do not output any text before <think> or after the closing code fence.
4) Never leave the code block empty.

In your reasoning, first decide the execution format for this problem:
- function-style vs stdin/stdout
- required function name/signature (if function-style)
- input/output format and constraints.
"""

EVAL_SYSTEM_PROMPT_STDIO = """You are an expert competitive programmer.
Solve the following problem. Write your complete Python solution.
Read input from stdin and print output to stdout."""

EVAL_SYSTEM_PROMPT_LEETCODE = """You are an expert competitive programmer.
Complete the following function."""

# ─────────────────────────────────────────
# Gemini judge prompt
# Sent to gemini-2.5-flash with the reasoning section extracted from completion
# (text inside <think> tags, before fenced code block)
# ─────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of competitive programming reasoning.
You will be given a coding problem and a model's step-by-step reasoning trace.
Score reasoning quality from 0.0 to 1.0.

Evaluate based on:
- Are the reasoning steps logically correct?
- Does the reasoning lead toward a valid solution approach?
- Are there factually incorrect statements about algorithms, data structures, or Python?
- Is the reasoning coherent and progressive (not circular or confused)?
- Does it cover the critical decisions needed to implement a correct solution?
- Is it concise and non-repetitive? (do not reward verbosity)

Penalize explicitly:
- repeated restatement of the same idea across steps
- filler text with no new technical content
- long reasoning that does not improve correctness

Respond with ONLY JSON in this exact format:
{"overall": 0.0}
No markdown, no prose, no extra keys."""

# ─────────────────────────────────────────
# Aliases (must be at end of file — after definitions they reference)
# ─────────────────────────────────────────

# Used by reward/judge.py
REASONING_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT
JUDGE_TIMEOUT = GEMINI_TIMEOUT
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = GEMINI_MAX_TOKENS
