# Pre-Train Checklist (A100 Operator Quick Reference)

Use this before every run.

## One Command Path

```bash
source venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
bash scripts/pretrain_checks.sh
```

Pass condition: exit code `0` and message `All pre-train checks passed.`

## Toggle Modes

```bash
# Skip live Gemini API probe
SKIP_GEMINI=1 bash scripts/pretrain_checks.sh

# Include trainer smoke run
RUN_TRAIN_SMOKE=1 bash scripts/pretrain_checks.sh

# Strict env checks
PRETRAIN_STRICT_ENV=1 bash scripts/pretrain_checks.sh
PRETRAIN_STRICT_TORCH=1 bash scripts/pretrain_checks.sh
```

## Manual Equivalent (Step-by-Step)

### 1) GPU and torch

```bash
python --version
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY
nvidia-smi
```

### 2) Data files present

```bash
python - <<'PY'
import os
from config import APPS_CLEAN_PATH, LCB_SEEN_PATH, LCB_EVAL_PATH
for label, path in [
    ("APPS", APPS_CLEAN_PATH),
    ("LCB train", LCB_SEEN_PATH),
    ("LCB eval", LCB_EVAL_PATH),
]:
    ok = os.path.exists(path)
    print(f"{label}: {path} -> {'OK' if ok else 'MISSING'}")
    assert ok, f"Missing: {path}"
print("data paths: PASS")
PY
```

### 3) Core smoke

```bash
python smoke_test.py
```

### 4) Live Gemini probe

```bash
python - <<'PY'
from reward.judge import score_batch, get_last_batch_stats
problems = ["Print hello world", "Given integer n, print n squared"]
thinks = [
    "[STEP] Print a constant string to stdout.",
    "[STEP] Read n, compute n*n, print result.",
]
difficulties = ["medium", "medium"]
scores = score_batch(problems, thinks, difficulties)
stats = get_last_batch_stats()
print("scores:", scores)
print("stats:", stats)
assert len(scores) == len(problems)
assert all(0.0 <= s <= 1.0 for s in scores)
assert stats.get("fallback_fraction", 1.0) < 1.0
print("gemini probe: PASS")
PY
```

## Train / Resume

Start new run:

```bash
python train.py
```

Resume latest checkpoint:

```bash
python train.py --resume-from-checkpoint latest
```

Resume specific checkpoint:

```bash
python train.py --resume-from-checkpoint checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-800
```

## Eval Commands

Base model:

```bash
python eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir results/eval_base \
  --save-debug-details
```

Merged checkpoint:

```bash
python eval.py \
  --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/merged-checkpoint-800 \
  --output-dir results/eval \
  --save-debug-details
```

## Useful Debug Helpers

Continue/stop decision:

```bash
python scripts/check_continue_stop.py \
  --entity <wandb-entity> \
  --project grpo-code-gen \
  --run <run_id> \
  --window 50 \
  --baseline-start 700 \
  --baseline-end 750
```

Execution stress:

```bash
python scripts/stress_execution.py \
  --rounds-clean 10 \
  --rounds-mixed 5 \
  --batch-size 32 \
  --workers 16 \
  --timeout 5
```

