# A100 Commands (Run This File)

This is the command sheet for daily use.

## 0) Activate Env

```bash
source venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

## 1) Pre-Train Checks (Single Entry Point)

Before pre-train checks, make sure data files exist on server:

```bash
ls -lh data/clean/apps_clean.jsonl data/clean/lcb_seen_clean.jsonl data/clean/lcb_unseen_clean.jsonl
```

If missing, run this from your local machine:

```bash
rsync -av --progress -e "ssh -i ~/.ssh/primeintellect_ed25519" \
  "/Users/arun-ghontale/UB/Personal Project/GRPO/data/clean/" \
  ubuntu@216.81.248.8:/home/ubuntu/grpo-code-reasoning/data/clean/
```

Run this and proceed only if it passes:

```bash
bash scripts/pretrain_checks.sh
```

Optional flags:

```bash
SKIP_GEMINI=1 bash scripts/pretrain_checks.sh
RUN_TRAIN_SMOKE=1 bash scripts/pretrain_checks.sh
PRETRAIN_STRICT_ENV=1 bash scripts/pretrain_checks.sh
PRETRAIN_STRICT_TORCH=1 bash scripts/pretrain_checks.sh
```

## 2) Train

```bash
python train.py
```

## 3) Resume

Resume latest:

```bash
python train.py --resume-from-checkpoint latest
```

Resume specific:

```bash
python train.py --resume-from-checkpoint checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-800
```

## 4) Eval (Base Model)

```bash
python eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output-dir results/eval_base \
  --save-debug-details
```

## 5) Eval (Trained Checkpoint)

Use merged checkpoint path (must contain `config.json`):

```bash
python eval.py \
  --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/merged-checkpoint-800 \
  --output-dir results/eval \
  --save-debug-details
```

If you only have adapter checkpoint (`checkpoint-800`), merge first:

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

## 6) W&B Continue/Stop Helper

```bash
python scripts/check_continue_stop.py \
  --entity <wandb-entity> \
  --project grpo-code-gen \
  --run <run_id> \
  --window 50 \
  --baseline-start 700 \
  --baseline-end 750
```

## 7) Helper Scripts Index

Pull latest train artifacts from remote (always prompts for host IP):

```bash
bash scripts/pull_latest_train.sh
```

Pull latest eval artifacts from remote (always prompts for host IP):

```bash
bash scripts/pull_latest_eval.sh
```

Analyze one W&B run locally:

```bash
python scripts/analyze_wandb_run.py --help
```

Analyze training mistakes with Gemini:

```bash
python scripts/analyze_train_mistakes_gemini.py --help
```

Sandbox and execution stress checks:

```bash
python scripts/stress_execution.py --help
python scripts/test_sandbox.py
python scripts/test_fence_sanitization.py
```

Eval audit helper:

```bash
bash scripts/audit_latest_eval.sh
```
