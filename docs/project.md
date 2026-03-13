# GRPO Fine-tuning for Code Generation — Full Project Documentation

> **Purpose of this document:** Complete reference for Cursor. Every architectural decision,
> every config value, every data contract, every known risk, and every file's responsibility
> is documented here. When something is ambiguous in code, check here first.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Config Reference](#3-config-reference)
4. [Data Pipeline](#4-data-pipeline)
5. [Sandbox](#5-sandbox)
6. [Reward System](#6-reward-system)
7. [Training Loop](#7-training-loop)
8. [Evaluation](#8-evaluation)
9. [WandB Tracking — Complete Metrics Spec](#9-wandb-tracking--complete-metrics-spec)
10. [Known Risks and Failure Modes](#10-known-risks-and-failure-modes)
11. [Pre-training Checklist](#11-pre-training-checklist)
12. [Decisions Log](#12-decisions-log)

---

## 1. Project Overview

**Goal:** Fine-tune Qwen2.5-Coder-7B-Instruct using GRPO with a multi-signal reward
(execution + reasoning quality), and show measurable Pass@1 improvement on LiveCodeBench.
Portfolio project demonstrating post-training RL skills for MLE roles.

**Model:** `Qwen/Qwen2.5-Coder-7B-Instruct`

**Training data:**
- APPS dataset (`codeparrot/apps`), introductory difficulty only, ~2739 clean problems
- LiveCodeBench seen split (`livecodebench/code_generation_lite`, release_v4), ~325 problems

**Eval data:** LiveCodeBench (713 problems), Pass@1 metric. Completely held out — no contamination.

**Compute:** Vast.ai single A100 80GB

**Training framework:** TRL `GRPOTrainer` + PEFT LoRA + vLLM colocate mode

---

## 2. Repository Structure

```
GRPO/
├── sandbox/
│   ├── __init__.py
│   └── testing_util.py          # Core execution engine — APPS + LCB formats
│
├── reward/
│   ├── __init__.py
│   ├── execution.py             # Subprocess sandbox, resource limits, partial credit
│   ├── judge.py                 # Gemini judge + presence heuristic
│   └── reward.py                # Main reward_fn — combines all signals + all logging
│
├── scripts/
│   ├── validate_apps.py         # Validates sandbox against APPS ground truth
│   ├── build_lcb_data.py        # Builds lcb_seen_clean.jsonl from raw LCB
│   ├── lcb_sandbox_compat.py    # Confirms LCB test case format works in sandbox
│   ├── generate_reasoning.py    # Claude Haiku → [STEP] traces for APPS problems
│   └── generate_lcb_solutions.py # Claude Sonnet → verified solutions for LCB
│
├── data/
│   └── clean/
│       ├── apps_clean.jsonl     # ~2739 APPS intro problems, sandbox-validated
│       └── lcb_seen_clean.jsonl # ~325 LCB seen problems
│
├── config.py                    # Single source of truth for all hyperparameters
├── train.py                     # GRPOTrainer setup, curriculum sampler, main loop
├── eval.py                      # Pass@1 evaluation on LCB, called from train.py
└── requirements.txt
```

---

## 3. Config Reference

All values live in `config.py`. Nothing is hardcoded in other files.

```python
# --- Model ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_NEW_TOKENS = 4096
MAX_PROMPT_LENGTH = 1024

# --- LoRA ---
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QUANTIZATION = None  # Full precision LoRA, no quantization

# --- GRPO ---
GROUP_SIZE = 4          # G — completions per problem per step
BATCH_SIZE = 8          # problems per step → 8 * 4 = 32 total completions
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 4
KL_COEFF = 0.04         # beta in GRPO loss — controls KL penalty

# --- Reward weights ---
EXEC_WEIGHT = 0.65               # execution reward weight
REASONING_WEIGHT = 0.35          # reasoning reward weight
GEMINI_WEIGHT = 0.7              # within reasoning: Gemini score weight
PRESENCE_WEIGHT = 0.3            # within reasoning: presence heuristic weight
# NOTE: GEMINI_WEIGHT + PRESENCE_WEIGHT must = 1.0
# NOTE: EXEC_WEIGHT + REASONING_WEIGHT must = 1.0

# --- Presence heuristic ---
MIN_STEPS = 3       # below this: partial credit (steps / MIN_STEPS)
MAX_STEPS = 10      # above this: still 1.0 — never penalize more steps
# Presence score = steps/MIN_STEPS if steps < MIN_STEPS, else 1.0

# --- Gemini ---
GEMINI_API_KEY = ""   # set via env var GEMINI_API_KEY, not hardcoded
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_MAX_WORKERS = 32          # I/O-bound thread pool size
GEMINI_CORRELATION_INTERVAL = 50 # compute Gemini-presence correlation every N reward_fn calls

# --- Sandbox ---
SANDBOX_MAX_WORKERS = 16         # CPU-bound thread pool size
SANDBOX_TIMEOUT = 5              # seconds per test case execution
SANDBOX_MEMORY_MB = 512          # memory cap per subprocess

# --- Curriculum (4 phases, keyed off global_step) ---
CURRICULUM = [
    {"step": 0,    "easy": 1.0, "medium": 0.0, "hard": 0.0},
    {"step": 300,  "easy": 0.5, "medium": 0.5, "hard": 0.0},
    {"step": 800,  "easy": 0.3, "medium": 0.5, "hard": 0.2},
    {"step": 1500, "easy": 0.2, "medium": 0.4, "hard": 0.4},
]

# --- Checkpointing ---
SAVE_STEPS = 200
PUSH_TO_HUB = True
HUB_MODEL_ID = "your-username/grpo-qwen-coder"   # update before training

# --- Evaluation ---
EVAL_INTERVAL_STEPS = 250        # run Pass@1 eval every N steps
EVAL_BATCH_SIZE = 8

# --- vLLM ---
VLLM_MODE = "colocate"
VLLM_GPU_MEMORY_UTILIZATION = 0.4   # tune based on A100 80GB headroom
# Supported versions: 0.10.2, 0.11.0, 0.11.1, 0.11.2, 0.12.0

# --- Logging ---
LOGGING_STEPS = 1
REPORT_TO = "wandb"
WANDB_PROJECT = "grpo-code-gen"
```

---

## 4. Data Pipeline

### 4.1 APPS Dataset

**Source:** `codeparrot/apps`, introductory split only.

**Format after cleaning (`apps_clean.jsonl`):**
```json
{
  "problem_id": "apps_1234",
  "difficulty": "easy",
  "description": "Given an array...",
  "test_cases": {
    "inputs": ["3\n1 2 3", "5\n1 2 3 4 5"],
    "outputs": ["6", "15"]
  }
}
```

**Test case format:** stdin/stdout style. Each input is a raw string piped to stdin.
Each output is the expected stdout string. NOT function-call style.

**Cleaning criteria (`validate_apps.py`):**
- Run ground truth solutions through sandbox
- Keep only problems where ground truth passes ≥ 95% of test cases
- ~2739 problems survived cleaning

### 4.2 LiveCodeBench Seen Split

**Source:** `livecodebench/code_generation_lite`, release_v4, "seen" split only.
The "seen" split is problems the base model may have encountered — used for training.
The full 713-problem set is held out for evaluation only.

**Format after cleaning (`lcb_seen_clean.jsonl`):**
```json
{
  "problem_id": "lcb_456",
  "difficulty": "medium",
  "description": "Given a string...",
  "test_cases": {
    "inputs": ["hello", "world"],
    "outputs": ["olleh", "dlrow"]
  }
}
```

**Difficulty distribution in LCB seen split:** not uniform — expect more medium/hard
than APPS which is mostly easy. Curriculum handles this via weighted sampling.

### 4.3 Data Flow into Training

Each sample loaded by the curriculum sampler passes these fields into the training batch,
which flow through to `reward_fn` via `**kwargs`:

```python
kwargs = {
    "difficulty": ["easy", "easy", "easy", "easy", "medium", ...],  # length = batch*G
    "test_cases": [tc1, tc1, tc1, tc1, tc2, tc2, tc2, tc2, ...],    # repeated G times
    "problem_ids": ["apps_1", "apps_1", ..., "lcb_5", ...],
    "problem_descriptions": ["desc1", "desc1", ..., "desc5", ...],
}
```

**CRITICAL ordering contract:** Problem i occupies indices `[i*G : (i+1)*G]` in all lists.
GRPO advantage computation groups completions this way. Never shuffle inside `reward_fn`.

### 4.4 Reference Generation Scripts (Optional Pre-processing)

These scripts generate reference data that was considered but ultimately NOT required
for training — the reward is purely execution-based + Gemini live scoring.
Scripts exist if needed for future SFT warmup or analysis.

- `scripts/generate_reasoning.py` — Claude Haiku → [STEP] reasoning traces for APPS
- `scripts/generate_lcb_solutions.py` — Claude Sonnet → verified solutions for LCB seen

---

## 5. Sandbox

### 5.1 File: `sandbox/testing_util.py`

Core execution engine. Handles both APPS (stdin/stdout) and LCB formats.

**Primary function:**
```python
def run_test(code: str, test_cases: dict) -> dict:
    """
    Execute code against test cases in isolated subprocess.

    Args:
        code: Python source code string
        test_cases: {"inputs": List[str], "outputs": List[str]}

    Returns:
        {"passed": int, "total": int, "results": List[bool]}

    Resource limits (enforced in child process):
        - Memory: SANDBOX_MEMORY_MB (512MB)
        - Timeout: SANDBOX_TIMEOUT (5s) per test case
        - No network access
    """
```

**Execution model:**
- Spawns a subprocess for each test case
- Pipes input string to stdin
- Captures stdout, compares to expected output (stripped)
- Kills subprocess on timeout via SIGKILL
- Returns partial credit: `passed / total`

**Known edge cases:**
- Some APPS problems require specific floating point formatting — sandbox uses exact string match
- LCB problems occasionally have trailing newlines in expected output — strip before comparison
- Recursive solutions can hit stack limits before memory limit — acceptable failure mode

### 5.2 File: `reward/execution.py`

Thin wrapper around `testing_util.run_test` that adds:
- Error handling and logging
- Returns 0.0 on any exception (never crashes training)
- Extracts code block from completion before passing to sandbox

**Interface:**
```python
def score_execution(completion: str, test_cases: dict) -> float:
    """Returns float in [0.0, 1.0]. Never raises."""
```

---

## 6. Reward System

### 6.1 Reward Formula

```
final_reward = EXEC_WEIGHT * execution_score
             + REASONING_WEIGHT * reasoning_score

reasoning_score = GEMINI_WEIGHT * gemini_score
                + PRESENCE_WEIGHT * presence_score
              (OR presence_score only if Gemini fails)
```

With locked values:
```
final_reward = 0.65 * execution_score + 0.35 * reasoning_score
reasoning_score = 0.7 * gemini_score + 0.3 * presence_score
```

**Flat weighting across all difficulties** — no tiering by easy/medium/hard.
Tiering was considered and rejected (see Decisions Log §12.3).

### 6.2 Execution Score

- Range: [0.0, 1.0]
- = `passed_tests / total_tests` (continuous partial credit)
- 0.0 if no valid ```python block found in completion
- 0.0 on any sandbox error

### 6.3 Presence Score

- Range: [0.0, 1.0]
- Counts `[STEP N]` blocks in completion (case-insensitive regex)
- `steps < MIN_STEPS (3)` → `steps / 3` (partial credit)
- `MIN_STEPS ≤ steps ≤ MAX_STEPS (10)` → `1.0`
- `steps > MAX_STEPS` → `1.0` (never penalize verbose reasoning)
- Provides non-zero reward floor even before Gemini signal is reliable

### 6.4 Gemini Score

- Range: [0.0, 1.0] or None on error
- Scores reasoning quality: correctness of steps, logical progression, absence of factual errors
- Uses `gemini-1.5-flash` (not Pro — cost vs quality tradeoff is acceptable)
- On `None` (any API error): fall back to presence score, log error
- Prompt sends only the reasoning section (before ```python block), not the full completion
- Gemini sees the problem description for grounding

### 6.5 Signal Coverage Tiers

Every completion falls into exactly one tier:
```
exec_only       — no [STEP N] blocks found in completion
exec_presence   — has steps, but Gemini API errored → presence only
exec_full       — has steps + Gemini score returned successfully
```

These are tracked as fractions per batch in WandB. Watch `exec_only_fraction` decrease
over training — this is evidence the model is learning to produce structured reasoning.

### 6.6 File: `reward/reward.py`

Main reward function called by `GRPOTrainer`.

**Signature:**
```python
def reward_fn(
    completions: list[str],
    prompts: list[str],
    tokenizer=None,
    max_new_tokens: int = 4096,
    **kwargs,
) -> list[float]:
```

**Parallelism architecture:**
- Module-level `_sandbox_pool` (ThreadPoolExecutor, 16 workers) — persistent, not recreated per call
- Module-level `_gemini_pool` (ThreadPoolExecutor, 32 workers) — persistent, not recreated per call
- Sandbox futures submitted first, Gemini futures submitted immediately after without waiting
- Both pools execute simultaneously — minimizes per-step latency
- Results collected via `as_completed()` — order preserved by index tracking

**Module-level state (survives across reward_fn calls):**
```python
_sandbox_pool: ThreadPoolExecutor        # persistent thread pool
_gemini_pool: ThreadPoolExecutor         # persistent thread pool
_seen_problem_ids: set[str]              # all problem IDs seen so far (for coverage metric)
_gemini_score_buffer: list[float]        # rolling buffer for correlation computation
_presence_score_buffer: list[float]      # rolling buffer for correlation computation
_reward_fn_call_count: int               # step counter for periodic logging
```

**CRITICAL: Do not shuffle `completions` or any kwargs list inside `reward_fn`.**
The ordering contract (problem i at indices `[i*G:(i+1)*G]`) is required for GRPO
advantage computation. Violation produces incorrect advantages and broken training.

### 6.7 File: `reward/judge.py`

Contains:
- `_call_gemini(completion, problem_description) -> float | None`
- `_presence_score(completion) -> tuple[float, int]`
- `_extract_reasoning(completion) -> str`
- `_extract_code(completion) -> str | None`

These are called from `reward.py`. Judge logic is separated for testability.

---

## 7. Training Loop

### 7.1 File: `train.py`

**Responsibilities:**
- Load model + tokenizer
- Set up LoRA via PEFT
- Initialize curriculum sampler
- Configure GRPOTrainer with vLLM
- Wire reward_fn with tokenizer + kwargs
- Set up checkpointing to HuggingFace Hub
- Call eval.py every EVAL_INTERVAL_STEPS steps

**GRPOConfig key settings:**
```python
GRPOConfig(
    output_dir="./checkpoints",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_generations=GROUP_SIZE,           # G=4
    max_completion_length=MAX_NEW_TOKENS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    kl_coeff=KL_COEFF,
    use_vllm=True,
    vllm_mode=VLLM_MODE,                  # "colocate"
    vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    save_steps=SAVE_STEPS,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HUB_MODEL_ID,
    logging_steps=LOGGING_STEPS,
    report_to=REPORT_TO,
    num_train_epochs=NUM_TRAIN_EPOCHS,
)
```

**vLLM notes:**
- `vllm_mode="colocate"` — vLLM and trainer share the same A100
- vLLM holds its own internal weight representation (separate from PyTorch training repr)
- LoRA adapter deltas synced from PyTorch → vLLM after each training step (base weights not copied)
- KV cache cleared between steps (stale after weight update)
- Pin vLLM version: `0.10.2`, `0.11.0`, `0.11.1`, `0.11.2`, or `0.12.0` — no others supported by TRL

### 7.2 Curriculum Sampler

Samples problems from `apps_clean.jsonl` + `lcb_seen_clean.jsonl` with difficulty weights
determined by current `global_step`.

```python
def get_curriculum_weights(global_step: int) -> dict:
    """Returns {"easy": float, "medium": float, "hard": float} based on current step."""
    for phase in reversed(CURRICULUM):
        if global_step >= phase["step"]:
            return {"easy": phase["easy"], "medium": phase["medium"], "hard": phase["hard"]}
    return CURRICULUM[0]
```

**Phase schedule:**
```
Step 0:    easy=1.0, medium=0.0, hard=0.0
Step 300:  easy=0.5, medium=0.5, hard=0.0
Step 800:  easy=0.3, medium=0.5, hard=0.2
Step 1500: easy=0.2, medium=0.4, hard=0.4
```

**Resume behavior:** Phase is computed from `global_step` on each sample call.
Stateless — correct behavior automatically on resume from checkpoint. No special handling needed.

**Re-seeing problems:** Fine. GRPO generates fresh rollouts each step so re-seeing
the same problem just produces new completions with current policy weights.

### 7.3 Reward Function Wiring

TRL's `GRPOTrainer` calls `reward_fn(completions, prompts, **kwargs)`.
The `**kwargs` must be passed through the dataset — each dataset row should contain
`difficulty`, `test_cases`, `problem_id`, `description` as fields,
and TRL passes them through automatically if the dataset columns match kwargs keys.

**Tokenizer injection:** Pass tokenizer to reward_fn via a wrapper:
```python
from functools import partial
reward_fn_with_tokenizer = partial(reward_fn, tokenizer=tokenizer, max_new_tokens=MAX_NEW_TOKENS)
trainer = GRPOTrainer(reward_funcs=[reward_fn_with_tokenizer], ...)
```

---

## 8. Evaluation

### 8.1 File: `eval.py`

Runs Pass@1 on the full LiveCodeBench eval set (713 problems).
Called from `train.py` every `EVAL_INTERVAL_STEPS` steps.

**Pass@1 definition:** For each problem, generate 1 completion, execute against test cases.
Pass if all test cases pass. Pass@1 = fraction of problems passed.

**Evaluation uses the same sandbox** as training — consistent scoring.

**Results logged to WandB:**
```
eval/pass@1_lcb
eval/pass@1_easy
eval/pass@1_medium
eval/pass@1_hard
```

**Important:** Eval uses greedy decoding (`temperature=0`, `do_sample=False`).
Training uses sampling for diversity. This is standard practice.

---

## 9. WandB Tracking — Complete Metrics Spec

All metrics logged every step unless noted otherwise.

### 9.1 Training Dynamics (TRL auto-logged)
```
train/loss
train/learning_rate
train/grad_norm
train/global_step
train/epoch
grpo/kl_divergence       # ← MOST IMPORTANT. Spike = stop training immediately
grpo/clip_ratio          # fraction of updates where ratio was clipped
grpo/advantage_mean
grpo/advantage_std
grpo/entropy             # TRL auto-logs this — verify after 10 steps, add custom if missing
```

### 9.2 Reward Signals (custom, logged in reward_fn)
```
reward/mean
reward/std
reward/non_zero_fraction
reward/execution_mean
reward/reasoning_mean
reward/execution_easy
reward/execution_medium
reward/execution_hard
reward/reasoning_easy
reward/reasoning_medium
reward/reasoning_hard
```

### 9.3 Signal Coverage Tiers (custom)
```
reward/exec_only_fraction       # no structured reasoning found
reward/exec_presence_fraction   # Gemini failed, presence only
reward/exec_full_fraction       # full signal: exec + presence + Gemini
# These three sum to 1.0
```

### 9.4 GRPO Degenerate Group Detection (custom)
```
grpo/all_zero_fraction       # fraction of G-groups where all rewards = 0
grpo/all_correct_fraction    # fraction of G-groups where all rewards ≥ 0.99
grpo/degenerate_fraction     # all_zero OR all_correct (no learning signal)
grpo/reward_std_mean         # mean std across all G-groups (near 0 = degenerate)
```

### 9.5 Generation Quality (custom)
```
gen/valid_code_fraction              # completions with a ```python block
gen/valid_code_easy
gen/valid_code_medium
gen/valid_code_hard
gen/truncated_fraction               # completions that hit MAX_NEW_TOKENS
gen/empty_completion_fraction        # completions with no content at all
gen/steps_count_mean                 # mean [STEP N] blocks per completion
gen/steps_count_easy
gen/steps_count_medium
gen/steps_count_hard
gen/reasoning_tokens_mean            # tokens in reasoning section (before ```python)
gen/reasoning_tokens_easy
gen/reasoning_tokens_medium
gen/reasoning_tokens_hard
gen/code_tokens_mean                 # tokens in ```python block only
gen/code_tokens_easy
gen/code_tokens_medium
gen/code_tokens_hard
```

### 9.6 Gemini Judge Reliability (custom)
```
judge/gemini_error_rate              # fraction of Gemini calls that returned None
judge/gemini_latency_mean            # mean API call latency in ms
judge/fallback_fraction_easy         # error rate broken out by difficulty
judge/fallback_fraction_medium
judge/fallback_fraction_hard
judge/gemini_presence_correlation    # logged every GEMINI_CORRELATION_INTERVAL steps
                                     # near 1.0 = Gemini just counting steps (no value)
                                     # lower = Gemini adding semantic signal
```

### 9.7 Dataset Coverage (custom)
```
data/unique_problems_seen            # running count of distinct problem IDs encountered
data/hard_problems_attempted_this_batch  # hard problems in current batch
curriculum/current_phase             # 0, 1, 2, or 3
curriculum/easy_fraction
curriculum/medium_fraction
curriculum/hard_fraction
```

### 9.8 Evaluation (every EVAL_INTERVAL_STEPS steps)
```
eval/pass@1_lcb
eval/pass@1_easy
eval/pass@1_medium
eval/pass@1_hard
```

### 9.9 Alert Conditions

| Metric | Threshold | Action |
|--------|-----------|--------|
| `grpo/kl_divergence` | Sudden spike (>2x baseline) | Stop training immediately, reduce LR or KL coeff |
| `gen/valid_code_fraction` | < 0.5 | Model forgetting to write code — check prompt format |
| `grpo/degenerate_fraction` | > 0.8 sustained | Learning stalled — check curriculum difficulty |
| `grpo/reward_std_mean` | < 0.05 sustained | Advantages near zero — same as above |
| `judge/gemini_error_rate` | > 0.2 | Gemini unreliable — check API key, rate limits |
| `gen/truncated_fraction` | > 0.3 | Completions being cut off — reasoning too long |
| `judge/gemini_presence_correlation` | > 0.95 | Gemini adding no semantic value over presence |

---

## 10. Known Risks and Failure Modes

### 10.1 vLLM Memory Contention (HIGH RISK)

**Problem:** vLLM and PyTorch trainer both hold model representations on the same A100.
On 80GB this should be fine, but if `VLLM_GPU_MEMORY_UTILIZATION` is set too high,
OOM can occur mid-training hours in.

**Mitigation:**
- Start with `VLLM_GPU_MEMORY_UTILIZATION = 0.4`
- Run 50-step smoke test and monitor GPU memory via `nvidia-smi`
- If OOM: reduce `VLLM_GPU_MEMORY_UTILIZATION` or reduce `BATCH_SIZE`

**Architecture note:** vLLM holds a separate internal representation of model weights
optimized for generation (not the same tensor object as PyTorch training repr).
Only LoRA adapter deltas are synced between them after each step — base weights are not
copied. KV cache is cleared between steps because it's stale after weight update.

### 10.2 Sparse Reward Early in Training (MEDIUM RISK)

**Problem:** Early in training, Qwen generates poor-quality traces (malformed steps,
no code block). This can cause very sparse reward signal, slowing initial learning.

**Mitigation already in design:**
- Presence heuristic provides non-zero floor reward for any structured output with ≥ 3 steps
- Curriculum starts with easy-only problems (higher chance of non-zero execution reward)
- `grpo/all_zero_fraction` tracks this — watch it decrease over first 200 steps

### 10.3 Gemini API Errors During Training (MEDIUM RISK)

**Problem:** Gemini API errors mid-training fallback to presence score silently.
If error rate is high, reasoning signal is effectively just a presence heuristic.

**Mitigation:**
- `judge/gemini_error_rate` tracks this per step
- Fallback is graceful — training continues, never crashes
- If error rate > 20% sustained, investigate API key / rate limits before continuing
- Rate limit: Tier 1 (billing enabled) = 150-300 RPM. At batch 32 = 32 Gemini calls/step,
  well within limits unless training steps are < 1s each.

### 10.4 Subprocess Security on Vast.ai (LOW RISK)

**Problem:** LLM-generated code runs in subprocesses on a rented machine.
Resource limits (memory cap, timeout) are enforced but not perfect isolation.

**Mitigation:** Resource limits via `resource.setrlimit` in child process.
Acceptable risk for a single-machine training run.

### 10.5 Checkpoint Loss on Instance Preemption (LOW RISK IF CONFIGURED)

**Problem:** Vast.ai instances can be preempted. Without checkpointing, all progress lost.

**Mitigation:** `save_steps=200`, `push_to_hub=True` to HuggingFace Hub.
Never save only to instance local disk. Hub provides persistent storage + version history.

### 10.6 Gemini Structural Limitation

**Problem:** The Gemini judge was evaluated on Gemini-generated traces (not real Qwen outputs).
This means we cannot detect false negatives — cases where Qwen produces subtly correct
reasoning that Gemini might undervalue.

**Why we proceed anyway:** Execution reward dominates (0.65 weight). Even if Gemini
gives a false negative on reasoning, correct code still gets high reward. The false
negative risk on the reasoning component (0.35 weight) is dampened to acceptable levels.

---

## 11. Pre-training Checklist

Run these before committing to a full training run:

- [ ] **Validate sandbox on Vast.ai instance** — run `python scripts/validate_apps.py`
      on the actual machine. OS-level subprocess behavior can differ from local dev.
- [ ] **50-step smoke test** — set `num_train_epochs` to hit ~50 steps, verify:
  - `reward/non_zero_fraction` > 0 (reward signal exists)
  - GPU memory stable (no OOM creep)
  - Checkpoints saving to Hub
  - WandB dashboard populated
- [ ] **Verify TRL auto-logs entropy** — after 10 steps, check WandB for `grpo/entropy`.
      If missing, add custom callback. Don't add it preemptively.
- [ ] **Set `HUB_MODEL_ID`** in config.py to your actual HuggingFace username/repo
- [ ] **Set `GEMINI_API_KEY`** via environment variable (never hardcode)
- [ ] **Pin vLLM version** in requirements.txt to one of the supported versions
- [ ] **Confirm `run_test` return format** — `reward.py` expects `{"passed": int, "total": int}`.
      Check `sandbox/testing_util.py` and update if format differs.

---

## 12. Decisions Log

Record of key decisions made during design, with rationale. When Cursor finds something
that looks odd, check here before changing it.

### 12.1 Model: 7B not 1.5B

Started with Qwen2.5-Coder-1.5B-Instruct. Upgraded to 7B because:
- A100 80GB has headroom
- 7B baseline Pass@1 is more interesting to improve
- Portfolio signal: RL on 7B is more impressive than 1.5B

### 12.2 No Quantization

QLoRA (4-bit) was the original plan. Switched to full precision LoRA because:
- 80GB A100 has enough VRAM
- 4-bit quantization adds complexity and can hurt training stability
- Full precision LoRA gives cleaner gradient signal

### 12.3 Flat Gemini Weighting (Not Tiered)

**Considered:** Tiered weighting by difficulty:
- Easy → presence only (Gemini unreliable, no spread)
- Medium → 70% Gemini / 30% presence
- Hard → 30% Gemini / 70% presence

**Rejected because:**
- Tiering was based on evaluation of Gemini-generated traces, not real Qwen outputs
- The "easy has no spread" finding was an artifact of Gemini generating its own traces
- Adds a moving part with no strong empirical justification
- Makes reward surface unnecessarily dynamic during training

**Decision:** Flat 70/30 across all difficulties. Simpler, cleaner gradient.

### 12.4 No Reference Solutions Required

**Considered:** Pre-generating reference solutions with Claude to use as reward signal.

**Rejected because:**
- Execution reward (passed_tests / total_tests) is already strong signal
- Gemini judge scores reasoning quality without needing reference
- Reference generation adds cost and complexity with no clear benefit
- APPS test cases already provide ground truth validation

### 12.5 No Problem-Level Tracking

**Considered:** Tracking which specific problems have been seen and how often,
to implement adaptive difficulty sampling.

**Rejected because:**
- Memory overhead of storing per-problem stats
- GRPO re-generates fresh rollouts per step — re-seeing a problem is fine
- Curriculum + random sampling provides sufficient coverage
- Unique problem count (set of IDs) is tracked as a lighter proxy

### 12.6 Presence Heuristic Step Limits

- Floor: 3 steps (below → partial credit)
- Ceiling: 10 steps (above → still 1.0, never penalize)
- MAX_NEW_TOKENS (4096) is the real hard limit on runaway generation
- Step ceiling is soft — encourages structure without suppressing complex reasoning
- 6-step hard limit was considered and rejected (was artifact of evaluation prompt)

### 12.7 ThreadPoolExecutor Over asyncio

**Decision:** Use `ThreadPoolExecutor` for both sandbox and Gemini calls.

**Rationale:**
- Sandbox: CPU-bound subprocess calls benefit from threads (GIL released in subprocess)
- Gemini: I/O-bound, threads are appropriate and simpler than asyncio
- Module-level pools avoid creation/destruction overhead per reward_fn call
- asyncio would require rewriting the entire reward_fn as async, which conflicts
  with TRL's synchronous reward function contract

### 12.8 Curriculum vs Randomization

Papers show randomization performs comparably to curriculum on final performance.
Curriculum's main benefit is training stability in early steps.

Decision: Keep curriculum. Rationale:
- Easy-only start reduces sparse reward problem early in training
- Stability matters more than marginal performance on a budget run
- Clean story for portfolio: explicit curriculum phases are explainable

---

*Document version: March 2026. Update when architectural decisions change.*