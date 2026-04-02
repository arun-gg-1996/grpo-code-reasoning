# GRPO Training Runbook

This repo fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` with GRPO on APPS + LCB.

The biggest pain point was unclear run steps. This README is now operator-first: what to run before every training run, how to start, what to watch, and when to stop.

## Source Of Truth

If docs and code disagree, trust code:
- `config.py` for all hyperparameters, paths, prompts, and model IDs
- `train.py` for training runtime behavior and CLI
- `eval.py` for evaluation runtime behavior and CLI
- `METRICS.md` for metric definitions

## 0) Before Every Training Run (Required)

Run this exact sequence from repo root:

```bash
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/pretrain_checks.sh
```

That command does all required preflight checks:
1. GPU + torch visibility
2. data file presence (`APPS`, `LCB train`, `LCB eval`)
3. config/credential sanity (Gemini key presence + HF hub model id)
4. `python smoke_test.py`
5. live Gemini API check (fails if all Gemini calls fallback)
6. optional trainer smoke test (off by default)

Optional toggles:

```bash
# Skip live Gemini call (quota/network situations)
SKIP_GEMINI=1 bash scripts/pretrain_checks.sh

# Also run train.py smoke test
RUN_TRAIN_SMOKE=1 bash scripts/pretrain_checks.sh

# Fail hard if gemini_api_key is missing
PRETRAIN_STRICT_ENV=1 bash scripts/pretrain_checks.sh

# Fail hard if torch import check fails
PRETRAIN_STRICT_TORCH=1 bash scripts/pretrain_checks.sh
```

Note:
- In some constrained environments, `torch` import can fail with OpenMP shared-memory errors.
- `scripts/pretrain_checks.sh` will warn and continue by default in that case.
- Set `PRETRAIN_STRICT_TORCH=1` if you want that to be a hard failure.

## 1) One-Time Setup (Per Machine)

```bash
git clone <repo>
cd GRPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in repo root:

```env
gemini_api_key=YOUR_GEMINI_API_KEY
```

Login once:

```bash
wandb login
huggingface-cli login
```

## 2) Start Training

Default run:

```bash
python train.py --save-debug-details
```

Recommended stabilization run (current tuning direction):

```bash
python train.py \
  --save-debug-details \
  --batch-size 2 \
  --rollout-temperature 0.7 \
  --vllm-gpu-memory-utilization 0.25 \
  --max-new-tokens 2048
```

Artifacts are written to timestamped folders:
- `results/train/<YYYYMMDD_HHMMSS>/summary.json`
- `results/train/<YYYYMMDD_HHMMSS>/train_details.jsonl` (if `--save-debug-details`)
- checkpoints in `checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/`

Quickly open latest train summary:

```bash
latest=$(ls -1dt results/train/*/ | head -n 1)
cat "${latest}summary.json"
```

## 3) What To Watch In W&B (First 30-60 Minutes)

Project: `grpo-code-gen`

Pin these first:
- `reward/execution_mean`
- `exec/zero_fraction`
- `exec/infra_zero_fraction`
- `exec/model_zero_fraction`
- `gen/valid_code_fraction`
- `judge/fallback_fraction`
- `judge/retry_count`
- `judge/rate_limit_count`
- `grpo/reward_std_mean`

Interpretation:
- High `exec/infra_zero_fraction`: formatting/infrastructure failures dominate (`error`/`timeout`/`empty`).
- High `exec/model_zero_fraction`: code ran, but logic failed tests.
- High `judge/fallback_fraction`: Gemini reliability issue.
- Near-zero `grpo/reward_std_mean`: weak GRPO learning signal.

## 4) Stop / Continue Heuristic

Continue when most are true:
- `gen/valid_code_fraction >= 0.75`
- `exec/zero_fraction <= 0.70`
- `exec/timeout_fraction <= 0.10`
- `grpo/reward_std_mean >= 0.05`
- `reward/execution_mean` is flat-to-up, not collapsing

Restart/tune if any persists for ~100+ steps:
- `exec/zero_fraction > 0.85`
- `exec/infra_zero_fraction` remains high (prompt/format/runtime issue)
- `judge/fallback_fraction` remains high with rising `judge/rate_limit_count`
- `gen/likely_truncated_fraction > 0.05` (consider `--max-new-tokens 3072`)

## 5) Evaluation (Before And After Training)

Baseline (before training):

```bash
python eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir results/eval \
  --save-debug-details
```

Evaluate trained checkpoint:

```bash
python eval.py \
  --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-2000 \
  --output-dir results/eval \
  --save-debug-details
```

Each eval run writes:
- `results/eval/<YYYYMMDD_HHMMSS>/summary.json`
- `results/eval/<YYYYMMDD_HHMMSS>/details.jsonl` (if enabled)

Show latest eval summary:

```bash
latest=$(ls -1dt results/eval/*/ | head -n 1)
cat "${latest}summary.json"
```

## 6) Minimal Daily Workflow

1. `source venv/bin/activate`
2. `bash scripts/pretrain_checks.sh`
3. `python train.py ...` (use the tuned command above)
4. monitor W&B for first 30-60 minutes
5. if healthy, let it run; if unhealthy, stop and retune
6. run `eval.py` on baseline/checkpoints/final model

## 7) Useful Commands

Smoke tests only:

```bash
python smoke_test.py
python train.py --smoke-test
```

Execution stress test:

```bash
python scripts/stress_execution.py --rounds-clean 10 --rounds-mixed 5 --batch-size 32 --workers 16 --timeout 5
```

Pull latest artifacts (if you use the provided helper scripts):

```bash
bash scripts/pull_latest_train.sh
bash scripts/pull_latest_eval.sh
```

## 8) Notes On Recent Stabilization Fixes

Important behavior change already merged:
- prompt format and execution scoring now share the same function-style detector (`problem_format.py`)
- this fixed the APPS mismatch where many problems were prompted as stdin/stdout but scored as function-style
- new execution diagnostics split zeros into infra-driven vs model-driven causes
