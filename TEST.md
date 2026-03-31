# Server Pre-Training Test Checklist

Run these in order on the cloud server before starting `python train.py`.
Each section must pass fully before moving to the next.

---

## Setup (First Time Only)

```bash
# Clone the repo and enter the project directory
git clone <repo-url>
cd GRPO

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Transfer data from local machine (run this on your LOCAL machine, not the server)
rsync -avz data/ user@server:/path/to/GRPO/data/

# Set up Gemini API key
echo "api_key=YOUR_GEMINI_API_KEY" > .env

# Log in to Weights & Biases
wandb login
# paste your API key from wandb.ai → User Settings → API Keys

# Log in to Hugging Face
huggingface-cli login
# paste a Write token from huggingface.co → Settings → Access Tokens

# Create logs directory
mkdir -p logs
```

---

## 0. Environment Setup (Prerequisites)

```bash
# Activate venv (every time you SSH in)
source venv/bin/activate

# Verify Python version (must be 3.10+)
python --version

# Verify GPU is visible
nvidia-smi

# Verify CUDA is available to PyTorch
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

**Expected:**
- Python 3.10.x or 3.11.x
- `nvidia-smi` shows A100 80GB
- `CUDA: True | Device: NVIDIA A100 80GB`

---

## 1. Logic and Sandbox Test (No GPU, No API calls)

```bash
python smoke_test.py
```

**Expected output:**
```
ALL SMOKE TESTS PASSED
```

**What it tests:**
- All module imports (config, sandbox, reward.execution, reward.reward, reward.judge)
- Config values and curriculum weight logic
- Code and `<think>` block extraction
- Judge response parsing
- Tier and source reward weight tables
- Sandbox code execution (correct/wrong/empty solutions)
- `reward_fn` end-to-end with easy problems (no Gemini call)
- Data loading: APPS (2739 problems), LCB seen (325 problems)
- APPS io normalization to test_cases format
- Execution test on a real APPS problem

**If it fails:** Do not proceed. Fix the error before continuing.

---

## 2. Data File Verification

```bash
python -c "
import json

# Check APPS
apps = [json.loads(l) for l in open('data/clean/apps_clean.jsonl') if l.strip()]
print(f'APPS: {len(apps)} problems')
assert len(apps) >= 2700, 'APPS too small'

# Check LCB seen (training)
lcb = [json.loads(l) for l in open('data/clean/lcb_seen_clean.jsonl') if l.strip()]
print(f'LCB seen: {len(lcb)} problems')
assert len(lcb) >= 300, 'LCB seen too small'

# Check LCB unseen (eval)
lcb_eval = [json.loads(l) for l in open('data/clean/lcb_unseen_clean.jsonl') if l.strip()]
print(f'LCB unseen: {len(lcb_eval)} problems')
assert len(lcb_eval) >= 380, 'LCB eval too small'

# Check platform field for eval split
platforms = [p.get('platform','').lower() for p in lcb_eval]
lc = platforms.count('leetcode')
ac = platforms.count('atcoder')
print(f'  LCB unseen platforms: leetcode={lc}, atcoder={ac}')
assert lc > 100 and ac > 100, 'Platform counts unexpected'

print('Data files OK')
"
```

**Expected:**
```
APPS: 2739 problems
LCB seen: 325 problems
LCB unseen: 387 problems
  LCB unseen platforms: leetcode=169, atcoder=218
Data files OK
```

---

## 3. Gemini API Verification (Live Call)

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from reward.judge import score_batch
scores = score_batch(
    ['Print hello world'],
    ['[STEP] Use print statement to output hello world\n[STEP] No input handling needed'],
    ['medium']
)
print('Gemini OK, score:', scores)
assert scores is not None and len(scores) == 1
assert 0.0 <= scores[0] <= 1.0
print('Gemini API: PASS')
"
```

**Expected:** Prints a float score like `[0.6]`. Any valid float confirms the API is live.

**If it fails:**
- Check `.env` file exists with `api_key=YOUR_KEY`
- Check GCP project `grpo-reasoning-2` has the Gemini API enabled
- Check API key is not expired

---

## 4. Weights & Biases Verification

```bash
python -c "
import wandb
run = wandb.init(project='grpo-code-gen', name='connectivity-test', mode='online')
wandb.log({'test_metric': 1.0})
wandb.finish()
print('wandb: PASS')
"
```

**Expected:** Run appears at `wandb.ai/arun-gv-ghontale/grpo-code-gen`

**If it fails:** Run `wandb login` and paste your API key from wandb.ai → User Settings → API Keys.

---

## 5. Hugging Face Hub Verification

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f'HF logged in as: {user[\"name\"]}')
assert user['name'] == 'arun-gv-ghontale', f'Wrong user: {user[\"name\"]}'
print('HF Hub: PASS')
"
```

**Expected:** `HF logged in as: arun-gv-ghontale`

**If it fails:** Run `huggingface-cli login` and paste a Write-access token from huggingface.co → Settings → Access Tokens.

---

## 6. GPU Memory and vLLM Check

```bash
python -c "
import torch
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU total memory: {total:.1f} GB')
assert total >= 75, f'Expected >=75GB, got {total:.1f}GB'

# Quick vLLM smoke test
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-Coder-1.5B-Instruct', gpu_memory_utilization=0.3)
params = SamplingParams(temperature=0.7, max_tokens=32)
out = llm.generate(['def hello():\n'], params)
print('vLLM output:', out[0].outputs[0].text[:80])
print('vLLM: PASS')
del llm
import gc; gc.collect(); torch.cuda.empty_cache()
"
```

**Expected:**
- `GPU total memory: 80.0 GB`
- vLLM generates some Python code
- No OOM errors

---

## 7. Full Pipeline Test (GRPOTrainer, 2 Steps)

```bash
python train.py --smoke-test
```

**Expected output (last few lines):**
```
Starting GRPO training...
100%|██████████| 2/2 [...]
Saving final model...
Training complete.
```

**What it tests:**
- Full GRPOTrainer loop with vLLM generation (on GPU this time)
- Reward function called inside the training loop
- LoRA config applied correctly
- Model saves without error

**Note:** On the A100 with GPU + vLLM this should complete in ~2 minutes (vs ~20 min on CPU locally).

**If it fails:** Check the traceback carefully — most likely causes are OOM (reduce `batch_size` or `group_size` in config) or a CUDA/vLLM version mismatch.

---

## 8. Reward Function Sanity Check (With GPU)

After step 7 completes, verify the reward metrics looked reasonable in the console output.

Look for this line in the step 1 output:
```
'rewards/_reward/mean': '<some_float>', 'reward_std': '<some_float>'
```

- `reward/mean` can be low (0.0–0.3 is normal for an untrained model)
- `reward_std` should be **> 0** — if it's exactly 0.0 on both steps, something is wrong with reward variation

---

## 9. Final Pre-Training Check

```bash
python -c "
from config import (
    TRAINING_MODEL, MAX_TRAINING_STEPS, BATCH_SIZE, GROUP_SIZE,
    MAX_NEW_TOKENS, LEARNING_RATE, KL_COEFF, LORA_RANK,
    SAVE_STEPS, PUSH_TO_HUB, HUB_MODEL_ID, WANDB_PROJECT,
    APPS_CLEAN_PATH, LCB_SEEN_PATH, LCB_EVAL_PATH,
)
import os

print('=== Training Config ===')
print(f'Model:          {TRAINING_MODEL}')
print(f'Steps:          {MAX_TRAINING_STEPS}')
print(f'Batch size:     {BATCH_SIZE}')
print(f'Group size:     {GROUP_SIZE}')
print(f'Max new tokens: {MAX_NEW_TOKENS}')
print(f'Learning rate:  {LEARNING_RATE}')
print(f'KL coeff:       {KL_COEFF}')
print(f'LoRA rank:      {LORA_RANK}')
print(f'Save every:     {SAVE_STEPS} steps')
print(f'Push to Hub:    {PUSH_TO_HUB} → {HUB_MODEL_ID}')
print(f'wandb project:  {WANDB_PROJECT}')
print()
print('=== Data Paths ===')
for label, path in [('APPS', APPS_CLEAN_PATH), ('LCB seen', LCB_SEEN_PATH), ('LCB eval', LCB_EVAL_PATH)]:
    exists = os.path.exists(path)
    print(f'{label}: {path} — {\"OK\" if exists else \"MISSING\"}')
    assert exists, f'Missing: {path}'
print()
print('All checks passed. Ready to train.')
"
```

**Expected:**
```
=== Training Config ===
Model:          Qwen/Qwen2.5-Coder-7B-Instruct
Steps:          2000
Batch size:     4
Group size:     8
Max new tokens: 8192
Learning rate:  1e-06
KL coeff:       0.04
LoRA rank:      8
Save every:     200 steps
Push to Hub:    True → arun-gv-ghontale/grpo-qwen-coder
wandb project:  grpo-code-gen

=== Data Paths ===
APPS: data/clean/apps_clean.jsonl — OK
LCB seen: data/clean/lcb_seen_clean.jsonl — OK
LCB eval: data/clean/lcb_unseen_clean.jsonl — OK

All checks passed. Ready to train.
```

---

## All Passed? Run Baseline Eval First

Before training, capture the untrained baseline:

```bash
python eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output results/baseline_results.json
```

Takes ~30–45 min. Record the pass@1/all and pass@3/all numbers — you'll compare against these after training.

---

## Then Start Training

```bash
screen -S train
python train.py
```

- Detach (keep running): `Ctrl+A` then `D`
- Re-attach later: `screen -r train`
- List screens: `screen -ls`

Monitor at: [wandb.ai/arun-gv-ghontale/grpo-code-gen](https://wandb.ai/arun-gv-ghontale/grpo-code-gen)

---

## Quick Reference: Pass / Fail Summary

| Test | Command | Pass Condition |
|------|---------|----------------|
| 0. GPU visible | `nvidia-smi` | A100 80GB shown |
| 1. Logic + sandbox | `python smoke_test.py` | `ALL SMOKE TESTS PASSED` |
| 2. Data files | inline python | 2739 / 325 / 387 problems |
| 3. Gemini API | inline python | float score returned |
| 4. wandb | inline python | run appears on wandb.ai |
| 5. HF Hub | inline python | `logged in as: arun-gv-ghontale` |
| 6. vLLM + GPU | inline python | 80GB, vLLM generates text |
| 7. GRPOTrainer loop | `python train.py --smoke-test` | `Training complete.` in ~2 min |
| 8. Reward sanity | check step 7 output | `reward_std > 0` |
| 9. Config check | inline python | all paths OK, correct values |
