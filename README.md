# GRPO A100 Runbook (Operator-First)

This repo trains `Qwen/Qwen2.5-Coder-7B-Instruct` with GRPO on APPS + LCB.

This runbook is for day-to-day operation on a single A100 instance: setup, preflight, train, resume, evaluate, and troubleshoot.

## Source Of Truth

If docs and code disagree, trust:
- `config.py` for hyperparameters, prompts, paths, curriculum, judge/sandbox knobs
- `train.py` for runtime behavior and CLI
- `eval.py` for evaluation behavior and CLI
- `METRICS.md` for metric definitions

## 1) One-Time Machine Setup

```bash
git clone <repo-url>
cd GRPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
gemini_api_key=YOUR_GEMINI_API_KEY
```

Login once:

```bash
wandb login
huggingface-cli login
```

## 2) Before Every Training Run (Required)

From repo root:

```bash
source venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
bash scripts/pretrain_checks.sh
```

Pass condition: script exits `0` and prints `All pre-train checks passed.`

Useful toggles:

```bash
# Skip live Gemini API call (quota/network situations)
SKIP_GEMINI=1 bash scripts/pretrain_checks.sh

# Also run train.py smoke test
RUN_TRAIN_SMOKE=1 bash scripts/pretrain_checks.sh

# Fail hard if gemini_api_key is missing
PRETRAIN_STRICT_ENV=1 bash scripts/pretrain_checks.sh

# Fail hard if torch probe fails
PRETRAIN_STRICT_TORCH=1 bash scripts/pretrain_checks.sh
```

## 3) Start Training (Current Recommended Path)

Use config defaults (no override flags needed):

```bash
python train.py
```

Current defaults in `config.py` include:
- `BATCH_SIZE=4`
- `G=8`
- `ROLLOUT_TEMPERATURE=0.7`
- `MAX_NEW_TOKENS=2048`
- `GRADIENT_ACCUMULATION_STEPS=8`
- `VLLM_MODE="colocate"`
- `VLLM_GPU_MEMORY_UTILIZATION=0.25`
- `VLLM_ENABLE_SLEEP_MODE=True`
- `VLLM_MAX_MODEL_LENGTH=MAX_PROMPT_LENGTH+MAX_NEW_TOKENS`
- `ATTN_IMPLEMENTATION="flash_attention_2"`
- `GEMINI_MAX_WORKERS=2`

Training artifacts:
- `results/train/<timestamp>/summary.json`
- `results/train/<timestamp>/train_details.jsonl` (enabled by default)
- `checkpoints/<model>-grpo/...`

## 4) Resume Training

Resume latest checkpoint in output dir:

```bash
python train.py --resume-from-checkpoint latest
```

Resume from a specific checkpoint path:

```bash
python train.py --resume-from-checkpoint checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-800
```

## 5) What To Watch In W&B

Project: `grpo-code-gen`

Pin first:
- `reward/execution_mean`
- `exec/zero_fraction`
- `exec/infra_zero_fraction`
- `exec/model_zero_fraction`
- `gen/valid_code_fraction`
- `gen/has_reasoning_fraction`
- `grpo/reward_std_mean`
- `judge/fallback_fraction`
- `judge/json_fraction`
- `judge/retry_count`
- `judge/rate_limit_count`
- `judge/consecutive_rate_limit_steps`
- `timing/reward_judge_s`
- `observer/critical_flag`
- `observer/attention_count`

Quick interpretation:
- High `exec/infra_zero_fraction`: formatting/runtime path issue (often missing `<code>` block).
- High `exec/model_zero_fraction`: code executes but fails tests (capability issue).
- Low `grpo/reward_std_mean`: weak contrast signal for GRPO.
- Rising retries/rate-limits + high judge time: Gemini pressure.

## 6) Continue / Stop Decision

Use rolling windows (not single-step spikes). You can run:

```bash
python scripts/check_continue_stop.py \
  --entity <wandb-entity> \
  --project grpo-code-gen \
  --run <run_id> \
  --window 50 \
  --baseline-start 700 \
  --baseline-end 750
```

Get run id quickly:

```bash
python - <<'PY'
import wandb
api = wandb.Api()
run = api.run("/<entity>/grpo-code-gen/runs/<run_id>")
print("run_id:", run.id)
print("name:", run.name)
print("url:", run.url)
PY
```

## 7) Evaluation

Base model baseline:

```bash
python eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir results/eval_base \
  --save-debug-details
```

Evaluate a merged local checkpoint (recommended):

```bash
python eval.py \
  --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/merged-checkpoint-800 \
  --output-dir results/eval \
  --save-debug-details
```

### If eval fails with “Invalid repository ID or local directory”

That means the path passed to `--model` is not a full model directory (missing `config.json`).

If your checkpoint is LoRA adapter-only, merge first:

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_path = "checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-800"
out_path = "checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/merged-checkpoint-800"

base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto", device_map="auto", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(out_path)
tok.save_pretrained(out_path)
print("Saved merged model to:", out_path)
PY
```

Then run `eval.py --model <merged-path>`.

## 8) Common Issues

### OOM during training
- Keep `vllm_mode="colocate"` and sleep mode on (already default).
- Reduce `BATCH_SIZE` in `config.py` from 4 to 3 if needed.
- Keep `VLLM_GPU_MEMORY_UTILIZATION=0.25` unless you have clear headroom.

### Gemini fallback spikes
- Watch `judge/fallback_fraction`, `judge/retry_count`, `judge/rate_limit_count`.
- Current default `GEMINI_MAX_WORKERS=2` is already conservative.

### High infra zeros
- Check `gen/valid_code_fraction`, `exec/empty_fraction`.
- Training now uses strict code-tag extraction mode (`STRICT_TRAIN_CODE_TAGS=True`), so completions missing `<code>...</code>` are intentionally penalized.

## 9) Daily Workflow (Short Version)

1. `source venv/bin/activate`
2. `export PYTORCH_ALLOC_CONF=expandable_segments:True`
3. `bash scripts/pretrain_checks.sh`
4. `python train.py`
5. monitor W&B for first 30-60 minutes
6. continue/stop using rolling-window metrics and `check_continue_stop.py`
7. run eval on base + merged checkpoint

