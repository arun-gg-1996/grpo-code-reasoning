# Pre-Training Validation Checklist

Use this file as a quick execution checklist only.

Canonical configuration lives in:

- `config.py` (parameters/prompts/paths)
- `README.md` (workflow and monitoring)

If this checklist and code ever disagree, trust code.

---

## 1. Environment

```bash
source venv/bin/activate
python --version
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Core Smoke Test (required)

```bash
python smoke_test.py
```

Pass condition: output contains `ALL SMOKE TESTS PASSED`.

## 3. Data Presence Check

```bash
python - <<'PY'
import os
from config import APPS_CLEAN_PATH, LCB_SEEN_PATH, LCB_EVAL_PATH

for label, path in [
    ("APPS", APPS_CLEAN_PATH),
    ("LCB seen", LCB_SEEN_PATH),
    ("LCB eval", LCB_EVAL_PATH),
]:
    ok = os.path.exists(path)
    print(f"{label}: {path} -> {'OK' if ok else 'MISSING'}")
    assert ok, f"Missing {path}"

print("Data paths: PASS")
PY
```

## 4. Gemini API Check

```bash
python -c "
from reward.judge import score_batch
scores = score_batch(['Print hello world'], ['[STEP] Use print statement'], ['medium'])
print('Gemini score:', scores)
assert len(scores) == 1 and 0.0 <= scores[0] <= 1.0
print('Gemini: PASS')
"
```

## 5. Optional End-to-End Trainer Smoke

```bash
python train.py --smoke-test
```

This validates GRPOTrainer wiring with a tiny local run.

## 6. Sandbox Stress Test (recommended before long eval/train)

Quick:

```bash
python scripts/stress_execution.py \
  --rounds-clean 10 \
  --rounds-mixed 5 \
  --batch-size 32 \
  --workers 16 \
  --timeout 5
```

Heavier:

```bash
python scripts/stress_execution.py \
  --rounds-clean 30 \
  --rounds-mixed 10 \
  --batch-size 32 \
  --workers 16 \
  --timeout 5
```

## 7. Baseline Eval Before Full Training

```bash
python eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --save-debug-details
```

Notes:

- Eval artifacts are auto-saved to a timestamped folder:
`results/eval/<YYYYMMDD_HHMMSS>/`
- Files:
  - `summary.json`
  - `details.jsonl` (raw per-completion rows)

Live tail while eval runs:

```bash
latest=$(ls -1dt results/eval/*/ | head -n 1)
tail -f "${latest}details.jsonl"
```

Verify new execution aggregates exist after eval:

```bash
python - <<'PY'
import json,glob
p=sorted(glob.glob("results/eval/*/summary.json"))[-1]
d=json.load(open(p))
print("summary:", p)
print("has execution_metrics:", "execution_metrics" in d)
for k in sorted(d.get("execution_metrics",{}).keys()):
    print(k, "=", d["execution_metrics"][k])
PY
```

Pull latest eval folder from server to local machine:

```bash
bash scripts/pull_latest_eval.sh
```

## 8. Start Full Training

```bash
python train.py
```

Monitor in W&B using the project configured in `config.py`.

## 9. Verify Checkpointing + HF Push (during real training)

Run these checks after training starts:

```bash
python -c "
from config import PUSH_TO_HUB, HUB_MODEL_ID, SAVE_STEPS
print('PUSH_TO_HUB:', PUSH_TO_HUB)
print('HUB_MODEL_ID:', HUB_MODEL_ID)
print('SAVE_STEPS:', SAVE_STEPS)
"
```

```bash
python -c "
from huggingface_hub import HfApi
print('HF user:', HfApi().whoami()['name'])
"
```

After first save step (default every 200 steps), verify local checkpoint:

```bash
ls -la checkpoints/Qwen2.5-Coder-7B-Instruct-grpo | head
ls -la checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-200
```

Verify Hub repo updated:

```bash
python -c "
from config import HUB_MODEL_ID
from huggingface_hub import list_repo_files
files = list_repo_files(HUB_MODEL_ID)
print('repo file count:', len(files))
print('sample files:', files[:20])
"
```

In W&B, confirm event metrics appear:

- `event/checkpointing`
- `event/checkpoint_save_s`
- `event/hf_push_s`

