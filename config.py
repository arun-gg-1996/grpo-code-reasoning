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

# Judge — Gemini API
JUDGE_MODEL = "gemini-2.5-flash"

# ─────────────────────────────────────────
# GRPO hyperparameters
# ─────────────────────────────────────────

G = 8  # rollouts per problem (GROUP_SIZE alias below)
GROUP_SIZE = G  # alias — used throughout reward.py and logging
BATCH_SIZE = 4  # problems per training step → 4 * 8 = 32 completions per step
ROLLOUT_TEMPERATURE = 0.8
EVAL_TEMPERATURE = 0.2
MAX_NEW_TOKENS = 4096
MAX_PROMPT_LENGTH = 1024

# ─────────────────────────────────────────
# Training hyperparameters
# Set from literature before cloud run — do not leave None
# ─────────────────────────────────────────

LEARNING_RATE = 1e-6  # standard for GRPO on 7B (DeepSeek-R1 range: 1e-6 to 3e-6)
KL_COEFF = 0.04  # KL penalty — controls drift from reference model
WARMUP_STEPS = 50  # short warmup, standard for RL fine-tuning
MAX_TRAINING_STEPS = 2000  # ~enough for 4 curriculum phases + convergence
GRADIENT_ACCUMULATION_STEPS = 4
VLLM_GPU_MEMORY_UTILIZATION = 0.4  # tune based on smoke test; start conservative

# ─────────────────────────────────────────
# LoRA config
# ─────────────────────────────────────────

LORA_RANK = 8
LORA_ALPHA = 16  # 2x rank — standard
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",  # attention layers
    "gate_proj", "up_proj", "down_proj"  # MLP layers
]
QUANTIZATION = None  # full precision LoRA — A100 80GB has headroom

# ─────────────────────────────────────────
# Reward weights (flat across all difficulties and datasets)
# ─────────────────────────────────────────

EXEC_WEIGHT = 0.65  # execution reward weight
REASONING_WEIGHT = 0.35  # reasoning reward weight
# EXEC_WEIGHT + REASONING_WEIGHT must = 1.0

GEMINI_WEIGHT = 0.7  # within reasoning: Gemini score weight
PRESENCE_WEIGHT = 0.3  # within reasoning: presence heuristic weight
# GEMINI_WEIGHT + PRESENCE_WEIGHT must = 1.0

REWARD_STD_WARNING_THRESHOLD = 0.05  # reward_std below this → diversity collapse warning

# ─────────────────────────────────────────
# Presence heuristic step limits
# ─────────────────────────────────────────

MIN_STEPS = 3  # below this: partial credit (steps / MIN_STEPS)
MAX_PRESENCE_STEPS = 10  # above this: still 1.0 — never penalize verbose reasoning
# Note: named MAX_PRESENCE_STEPS to avoid conflict with MAX_TRAINING_STEPS

# ─────────────────────────────────────────
# Gemini judge
# ─────────────────────────────────────────

GEMINI_MAX_WORKERS = 32  # I/O-bound thread pool for parallel Gemini calls
GEMINI_TIMEOUT = 30  # seconds per Gemini API call before fallback
GEMINI_CORRELATION_INTERVAL = 50  # compute Gemini-presence correlation every N reward_fn calls
GEMINI_MAX_TOKENS = 512  # judge response is short, cap output tokens

# ─────────────────────────────────────────
# Execution sandbox
# ─────────────────────────────────────────

SANDBOX_MAX_WORKERS = 16  # CPU-bound thread pool for parallel sandbox calls
SANDBOX_TIMEOUT = 5  # seconds per subprocess execution before SIGKILL
SANDBOX_MEMORY_MB = 512  # memory cap per subprocess (resource.setrlimit)
MAX_TEST_CASES = 10  # test cases per problem during training (cap for speed)

# ─────────────────────────────────────────
# Checkpointing and logging
# ─────────────────────────────────────────

SAVE_STEPS = 200
PUSH_TO_HUB = True
HUB_MODEL_ID = "your-username/grpo-qwen-coder"  # update before training
LOGGING_STEPS = 1  # log every step — only ~2000 steps total, want full resolution
REPORT_TO = "wandb"
WANDB_PROJECT = "grpo-code-gen"

# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

EVAL_INTERVAL_STEPS = 250  # run Pass@1 on LCB every N training steps
EVAL_BATCH_SIZE = 8
EVAL_N_GENERATIONS = 5  # generations per problem for pass@k
EVAL_K_VALUES = [1, 3]  # pass@1 is primary metric, pass@3 for completeness

# ─────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────

EARLY_STOP_MIN_STEPS = 500  # don't stop before this regardless of metrics
EARLY_STOP_PATIENCE = 50  # steps without improvement before stopping
EARLY_STOP_THRESHOLD = 0.01  # minimum improvement to count as progress
VALIDATION_INTERVAL = 100  # steps between held-out validation eval
VALIDATION_PROBLEMS = 50  # problems in held-out subset

# ─────────────────────────────────────────
# Curriculum schedule
# Keyed off global_step — stateless, correct on resume from checkpoint
# ─────────────────────────────────────────

CURRICULUM = [
    # Phase 0: easy only — model learns format, [STEP] blocks, basic reward signal
    (0, {
        "difficulty": {"easy": 0.8, "medium": 0.2, "hard": 0.0}
    }),
    # Phase 1: introduce medium — easy anchors reward signal while medium challenges
    (300, {
        "difficulty": {"easy": 0.7, "medium": 0.3, "hard": 0.0}
    }),
    # Phase 2: introduce hard gradually
    (800, {
        "difficulty": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    }),
    # Phase 3: full distribution — bias toward harder problems
    (1500, {
        "difficulty": {"easy": 0.2, "medium": 0.4, "hard": 0.4}
    }),
]

ADAPTIVE_REWARD_THRESHOLD = 0.85  # if problem's running mean reward exceeds this,
ADAPTIVE_WINDOW = 50  # halve its sampling weight (over last N steps)


def get_curriculum_weights(step: int) -> dict:
    """Return difficulty sampling weights for the current training step."""
    weights = CURRICULUM[0][1]
    for from_step, w in CURRICULUM:
        if step >= from_step:
            weights = w
    return weights["difficulty"]


# ─────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────

APPS_CLEAN_PATH = "data/clean/apps_clean.jsonl"
LCB_SEEN_PATH = "data/clean/lcb_seen_clean.jsonl"
LCB_EVAL_PATH = "data/clean/lcb_unseen_clean.jsonl"  # held out — eval only, never train
FAILED_DIR = "data/failed"

# ─────────────────────────────────────────
# Training prompt template
# Model is instructed to reason inside <think> tags using [STEP] blocks,
# then write solution inside <code> tags.
# Both tags are parsed in reward.py for scoring.
# ─────────────────────────────────────────

TRAINING_SYSTEM_PROMPT = """You are an expert competitive programmer.
Solve the following problem step by step.

First, think through your approach inside <think> tags using exactly these steps:
[STEP] Problem understanding: what is being asked, input/output format, constraints
[STEP] Algorithm choice: what algorithm/approach and why
[STEP] Data structures: what data structures are needed and why
[STEP] Time and space complexity: expected complexity
[STEP] Edge cases: what edge cases need to be handled
[STEP] Implementation plan: how to translate the approach to code

Then write your complete Python solution inside <code> tags.
Read input from stdin and print output to stdout.

Use as many [STEP] blocks as you need — there is no limit."""

EVAL_SYSTEM_PROMPT_STDIO = """You are an expert competitive programmer.
Solve the following problem. Write your complete Python solution.
Read input from stdin and print output to stdout."""

EVAL_SYSTEM_PROMPT_LEETCODE = """You are an expert competitive programmer.
Complete the following function."""

# ─────────────────────────────────────────
# Gemini judge prompt
# Sent to gemini-2.5-flash with the reasoning section extracted from completion
# (text inside <think> tags, before <code> block)
# ─────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of competitive programming reasoning.
You will be given a coding problem and a model's step-by-step reasoning trace.
Score the reasoning quality from 0.0 to 1.0.

Evaluate based on:
- Are the reasoning steps logically correct?
- Does the reasoning lead toward a valid solution approach?
- Are there factually incorrect statements about algorithms, data structures, or Python?
- Is the reasoning coherent and progressive (not circular or confused)?

Respond with ONLY a float between 0.0 and 1.0. Nothing else. No explanation."""

