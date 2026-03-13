# GRPO Code Generation Project Plan

## Project Goal

Fine-tune Qwen2.5-Coder-7B-Instruct using GRPO with a 2-component reward signal (execution + reasoning) to show measurable Pass@1 and Pass@3 improvement on LiveCodeBench release_v4. Resume project targeting MLE roles at AI labs.

---

## Models

| Role | Model |
|---|---|
| Training model | Qwen2.5-Coder-7B-Instruct |
| Judge model | Qwen2.5-Coder-7B-Instruct (dedicated GPU) |
| Reference trace generation (offline, one-time) | Claude Haiku (`claude-haiku-4-5-20251001`) |
| LCB solution generation (offline, one-time) | Claude API (Sonnet) |
| Local pipeline testing (Mac) | Qwen2.5-Coder-1.5B-Instruct (training) + Qwen2.5-Coder-0.5B-Instruct (judge) |

---

## Compute

| Resource | Spec |
|---|---|
| Training GPU | A100 80GB (Vast.ai) |
| Judge GPU | A100 40GB (Vast.ai, same machine, NVLink preferred) |
| Budget | ~$100 total |
| Estimated cost | $3-5 offline preprocessing + $5-8 smoke test + $3-5 baseline eval + $50-70 training + $3-5 final eval |

---

## Datasets

### Training Data — Final Validated Counts

| Source | Problems | Weighted | Notes |
|---|---|---|---|
| APPS introductory | 2392 | 2392 | weight 1.0 |
| APPS competition | 347 | 694 | weight 2.0 |
| LCB seen | 325 | 650 | weight 2.0 |
| **TOTAL** | **3064** | **3736** | |

### APPS Dataset

**Validation method**: Official Hendrycks `testing_util.run_test` harness (`sandbox/testing_util.py`).
Each solution runs in an isolated subprocess to prevent `reliability_guard()` from poisoning the parent process.
A problem is clean if at least one ground truth solution scores ≥ 0.9 (90% test cases pass).

**APPS introductory** (`codeparrot/apps`, split=train, difficulty=introductory):
- Total: 2639 problems
- Clean: 2392 (90.6%)
- stdin/stdout: 130, function-call: 2262
- Dropped (no io): 241, dropped (score): 6
- Avg test cases: varies, capped at 10 during training
- 95% of problems are function-call style (fn_name present in input_output JSON)

**APPS competition** (`codeparrot/apps`, split=train, difficulty=competition):
- Total: 361 problems
- Clean: 347 (96.1%)
- stdin/stdout: 347, function-call: 0
- Dropped (no io): 1, dropped (score): 13
- Avg test cases: 9.6

**APPS interview**: Excluded. 53/57 failed sandbox (Python 2 contamination), only 4 clean after 3+ test case filter.

**Sandbox**: `testing_util.run_test` handles both stdin/stdout and function-call natively.
- stdin/stdout: wraps code in `code()` function, patches sys.stdin
- function-call: uses `RuntimeModule.from_string` + `Solution()` pattern
- Each call spawns a fresh subprocess via `multiprocessing.fork`

### LCB Seen Dataset

**Source**: `livecodebench/code_generation_lite`, split=test, version_tag=release_v2
**Filter**: contest_date < 2024-01-01 (seen split)

| Metric | Value |
|---|---|
| Total seen problems | 325 |
| LeetCode (functional) | 150 |
| AtCoder + Codeforces (stdin) | 175 |
| Avg test cases | 2.6 |
| Min / Max | 1 / 6 |

**Test case breakdown**:
- Public test cases: parsed from `public_test_cases` field (JSON list)
- Private test cases: decompressed from `private_test_cases` (base64 + gzip)
- Total stdin tests: 483, functional tests: 363

**No ground truth solutions** — LCB is an eval benchmark, solutions intentionally withheld.
Reference solutions will be generated via `generate_lcb_solutions.py` (Claude API enrichment step, offline, ~$5).
Claude-generated solutions accepted only if they pass the problem's test cases via sandbox execution.

**Saved to**: `data/clean/lcb_seen_clean.jsonl`
Each problem includes: `question`, `platform`, `difficulty`, `contest_date`, `starter_code`, `func_name`, `test_cases`, `stdin_tests`, `functional_tests`, `reference_solution` (None until enrichment), `source`, `sampling_weight`

### Evaluation Data

**Source**: `livecodebench/code_generation_lite`, split=test, version_tag=release_v4
**Total**: 713 problems (1 dropped — "Sort", abc350_c, zero public test cases)
**Usable**: 712 problems

| Platform | Count | Style |
|---|---|---|
| AtCoder | 384 usable | stdin/stdout |
| LeetCode | 319 | function-call |
| Codeforces | 9 | stdin/stdout |

| Difficulty | Count |
|---|---|
| Easy | 238 (33%) |
| Medium | 279 (39%) |
| Hard | 196 (28%) |

**Validation confirmed**:
- All 319 LeetCode problems: `func_name` present, `starter_code` present — exhaustively verified
- All 394 AtCoder/CF problems: stdin/stdout format — exhaustively verified
- Private test cases: 0 across all 713 problems
- Public test cases: avg 2.6, min 0, max 6

---

## Reward Function

### Per-Source Reward Weights

| Source | Execution | Reasoning |
|---|---|---|
| APPS | 0.75 | 0.25 |
| LCB (any) | 0.60 | 0.40 |

**Rationale**: APPS has 9.6 avg test cases → execution signal reliable → higher execution weight.
LCB has 2.6 avg test cases → execution signal noisy → upweight reasoning to compensate.
Similarity scoring was dropped — it provides SFT-like pattern-matching signal that conflicts
with the RL objective of teaching reasoning behavior.

### Component 1 — Execution Score

```python
execution_score = passed_test_cases / total_test_cases  # 0.0 to 1.0
```

- APPS problems: `testing_util.run_test` in isolated subprocess
- LCB stdin problems: `testing_util.run_test` in isolated subprocess
- LCB functional problems: `LeetCodeSandbox` in isolated subprocess
- Partial credit — not binary
- Capped at 10 test cases per problem during training
- Timeout: 15s wall-clock subprocess guard
  (note: `testing_util` uses a 4s `SIGALRM` per test case internally)
- `sys.set_int_max_str_digits(100000)` set in each sandbox subprocess

### Component 2 — Reasoning Score

Three cases depending on whether the model produced a `<think>` block and whether execution passed:

```python
# Case 1: no <think> block in response
reasoning_score = 0.0

# Case 2: <think> block present AND execution_score == 1.0
# Presence heuristic only — no judge call.
# A fully correct solution should not be penalized for reasoning style differences.
steps = think_block.split("[STEP]")[1:]
valid_steps = sum(1 for s in steps[:6] if len(s.strip()) >= 20)
reasoning_score = valid_steps / 6.0

# Case 3: <think> block present AND execution_score < 1.0
# Judge comparison against reference reasoning steps.
reasoning_score = judge_compare(problem, reference_steps, model_steps)
# Falls back to heuristic if no reference_reasoning stored for the problem.
```

**Reference think blocks**: Generated offline once before training using Claude Haiku
(`generate_reasoning.py`). Stored in `apps_clean.jsonl` as `reference_reasoning`.

### Judge Inference
- Reasoning scoring: Qwen2.5-Coder-7B-Instruct on dedicated A100 40GB
- Served via vLLM, inference only — no gradients, no LoRA
- All judge calls sent concurrently via `asyncio` + `AsyncOpenAI` client.
  vLLM receives all requests simultaneously and batches them internally.

---

## Sandbox Architecture

### Execution Flow
```
GRPO training step:
  sample problem from dataset
  generate G=16 rollouts
  for each rollout (parallel pool):
    spawn subprocess → sandbox execution → execution score 0.0-1.0
  send all judge calls concurrently (asyncio) → reasoning scores
  combine: reward = w_exec * exec_score + w_reason * reason_score
  compute advantages from 16 reward scores
  update weights
```

### Subprocess Isolation
All sandbox execution runs in isolated subprocesses via `multiprocessing.fork`.
Required because `testing_util.reliability_guard()` permanently nullifies `os.kill`,
`subprocess.Popen` etc in the calling process. Subprocess isolation contains this damage.

### Sandbox Routing
```python
if problem["source"] == "apps" or not problem["is_leetcode"]:
    sandbox = testing_util.run_test      # handles both stdin and function-call
elif problem["is_leetcode"]:
    sandbox = LeetCodeSandbox            # for LCB functional problems
```

### Parallel Execution
At training time: `multiprocessing.Pool` with 16 workers on judge GPU CPUs.
All 16 rollout solutions execute in parallel — takes as long as slowest one.
Sequential is acceptable for offline validation scripts only.

---

## Training Method

### QLoRA Configuration
```
quantization:     None (full LoRA — A100 80GB has headroom, avoids NF4 dequant overhead)
lora_rank:        8
lora_alpha:       16
target_modules:   q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
lora_dropout:     0.05
trainable_params: ~1.5-2M (out of 7B total)
```

### GRPO Configuration
```
group_size G:     16
framework:        TRL GRPOTrainer
rollout engine:   vLLM (dedicated to training GPU)
temperature:      0.8 during training rollouts
                  0.2 during eval inference
```

### Memory Layout
```
Training GPU (A100 80GB):
  vLLM Qwen-7B rollout generation:   ~28GB (35% cap)
  QLoRA trainer:                      ~10GB
  Activations + overhead:             ~10GB
  Remaining headroom:                 ~32GB

Judge GPU (A100 40GB):
  vLLM Qwen-7B judge inference:       ~14GB
  Sandbox execution workers:          ~4GB
  Reward aggregation:                 ~2GB
  Remaining headroom:                 ~20GB
```

---

## Training Prompt Format

```
You are an expert competitive programmer.
Solve the following problem step by step.

First, think through your approach using exactly these steps inside <think> tags:
[STEP] Problem understanding: what is being asked, input/output format, constraints
[STEP] Algorithm choice: what algorithm/approach and why
[STEP] Data structures: what data structures are needed and why
[STEP] Time and space complexity: expected complexity
[STEP] Edge cases: what edge cases need to be handled
[STEP] Implementation plan: how to translate the approach to code

Then write your complete Python solution inside <code> tags.
Read input from stdin and print output to stdout.

Problem:
{problem_description}
```

### Output Format Enforcement
- No `<think>` block → `reasoning_score = 0.0` (Case 1)
- Fewer than 6 valid `[STEP]` blocks → lower heuristic score (Case 2) or lower judge score (Case 3)
- No `<code>` block → `execution_score = 0.0`

---

## Evaluation Protocol

### Metrics
- **Pass@1** and **Pass@3** (computed from n=10 generations per problem)

```python
from math import comb

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

### Reporting Split

| Slice | Description |
|---|---|
| Overall | All 712 problems |
| stdin/stdout subset | 393 AtCoder + Codeforces |
| LeetCode subset | 319 LeetCode (never trained on this format) |
| Easy | 238 problems |
| Medium | 279 problems |
| Hard | 196 problems |

**Key signal**: If LeetCode subset improves post-training (model never trained on function-call format), that is genuine reasoning generalization.

### Eval Prompt Format

stdin/stdout problems:
```
You are an expert competitive programmer.
Solve the following problem. Write your complete Python solution.
Read input from stdin and print output to stdout.

Problem:
{problem_description}
```

LeetCode problems:
```
You are an expert competitive programmer.
Complete the following function:

{starter_code}

Problem:
{problem_description}
```

---

## Early Stopping

```
minimum_steps:          500
patience_window:        50 steps
improvement_threshold:  0.01 mean reward improvement
validation_interval:    every 100 steps on 50-problem held-out subset
```

---

## Development Workflow

### Phase 1 — Local Mac (zero cloud cost) ← CURRENT
- ✅ Dataset validation (APPS + LCB)
- ✅ Sandbox build (testing_util + LeetCodeSandbox)
- ✅ Data pipeline (apps_clean.jsonl + lcb_seen_clean.jsonl)
- ✅ Claude Haiku reference think block generation — generate_reasoning.py
- ✅ Claude Sonnet LCB solution enrichment — generate_lcb_solutions.py
- ✅ Reward function (execution + reasoning, async judge)
- ✅ Config (config.py)

### Phase 2 — Local Mac with small models
- ⬜ Full pipeline smoke test: Qwen2.5-Coder-1.5B + Qwen2.5-Coder-0.5B judge
- ⬜ 10-20 training steps — confirm loop runs
- ⬜ Output parsing validation
- ⬜ Reward function sanity check

### Phase 3 — Cloud (A100 80GB + A100 40GB)
- ⬜ Timing benchmark: one training step end to end
- ⬜ Baseline eval: Pass@1 and Pass@3 pre-training
- ⬜ Full training run
- ⬜ Final eval: Pass@1 and Pass@3 post-training

---

## File Structure

```
GRPO/
  sandbox/
    __init__.py
    testing_util.py          ← official APPS/LCB harness (Hendrycks)
    leetcode_sandbox.py      ← LCB functional problems
    stdio_sandbox.py         ← legacy, may deprecate
  scripts/
    validate_apps.py         ← APPS validation (done)
    build_lcb_dataset.py     ← LCB parsing (done)
    generate_lcb_solutions.py ← Claude API enrichment (TODO)
    test_sandbox.py          ← sandbox unit tests
  data/
    clean/
      apps_clean.jsonl       ← 2739 problems
      lcb_seen_clean.jsonl   ← 325 problems
    failed/
      apps/                  ← failed sample inspection
      lcb/
  requirements.txt
```

---

## Hyperparameters (decide before training)

- Learning rate
- Batch size
- Number of training steps
- KL penalty coefficient
- Warmup steps
- vLLM memory fraction (empirical from smoke test)

---

## Key Decisions Log

| Decision | Choice | Rationale |
|---|---|---|
| APPS difficulties | introductory + competition | interview excluded (Python 2, 1.5 avg test cases) |
| Execution harness | testing_util.run_test | official harness handles both stdin and function-call |
| Subprocess isolation | multiprocessing.fork per call | reliability_guard() poisons calling process |
| LCB in training | yes, seen split (325 problems) | higher quality problems, weight 2x |
| Similarity scoring | dropped entirely | SFT-like pattern-matching signal conflicts with RL objective |
| LCB reward weights | 0.60 exec / 0.40 reasoning | 2.6 avg tests is noisy, upweight reasoning |
| LCB reference solutions | Claude Sonnet API, verified by execution | no ground truth shipped, generate + verify |
| Reasoning on perfect execution | heuristic (no judge call) | correct solution should not be judged against reference style |
| Adaptive sampling | by difficulty, not adaptive rollouts | avoids reasoning contamination |
| G | fixed at 16 | consistent advantage estimation |