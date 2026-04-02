# Pre-Train Checklist (Operator Reference)

This file is the command-focused companion to `README.md`.

If you want one command, use:

```bash
bash scripts/pretrain_checks.sh
```

---

## Required Before Every Training Run

```bash
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/pretrain_checks.sh
```

Pass condition: script exits 0 and prints `All pre-train checks passed.`

---

## Manual Equivalent (If You Prefer Step-By-Step)

### 1) Environment + GPU

```bash
python --version
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
```

### 2) Data Presence

```bash
python - <<'PY'
import os
from config import APPS_CLEAN_PATH, LCB_SEEN_PATH, LCB_EVAL_PATH
for label, path in [
    ('APPS', APPS_CLEAN_PATH),
    ('LCB train', LCB_SEEN_PATH),
    ('LCB eval', LCB_EVAL_PATH),
]:
    ok = os.path.exists(path)
    print(f'{label}: {path} -> {"OK" if ok else "MISSING"}')
    assert ok, f'Missing {path}'
print('data_paths: PASS')
PY
```

### 3) Core Smoke Test

```bash
python smoke_test.py
```

### 4) Live Gemini Check

```bash
python - <<'PY'
from reward.judge import score_batch, get_last_batch_stats
problems = ['Print hello world', 'Given integer n, print n squared']
thinks = [
    '[STEP] Print a constant string to stdout.',
    '[STEP] Read n, compute n*n, print result.',
]
difficulties = ['medium', 'medium']
scores = score_batch(problems, thinks, difficulties)
stats = get_last_batch_stats()
print('gemini_score:', scores)
print('gemini_stats:', stats)
assert len(scores) == len(problems)
assert all(0.0 <= s <= 1.0 for s in scores)
assert stats.get('fallback_fraction', 1.0) < 1.0
print('gemini_check: PASS')
PY
```

### 5) Optional Trainer Smoke

```bash
python train.py --smoke-test
```

---

## Start Training

Default:

```bash
python train.py --save-debug-details
```

Recommended stabilization config:

```bash
python train.py \
  --save-debug-details \
  --batch-size 2 \
  --rollout-temperature 0.7 \
  --vllm-gpu-memory-utilization 0.25 \
  --max-new-tokens 2048
```

---

## Evaluate Baseline / Checkpoints

Baseline:

```bash
python eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir results/eval \
  --save-debug-details
```

Checkpoint:

```bash
python eval.py \
  --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-2000 \
  --output-dir results/eval \
  --save-debug-details
```

---

## Optional Stress / Diagnostics

Execution stress:

```bash
python scripts/stress_execution.py --rounds-clean 10 --rounds-mixed 5 --batch-size 32 --workers 16 --timeout 5
```

Fence sanitization regression:

```bash
python scripts/test_fence_sanitization.py
```

Check continue/stop decision from W&B run:

```bash
python scripts/check_continue_stop.py \
  --entity <wandb-entity> \
  --project grpo-code-gen \
  --run <run_id>
```

---

## Pull Latest Artifacts

```bash
bash scripts/pull_latest_train.sh
bash scripts/pull_latest_eval.sh
```
