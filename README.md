# GRPO — Qwen2.5-Coder-7B on Competitive Programming

Fine-tunes **Qwen2.5-Coder-7B-Instruct** with GRPO on APPS + LCB problems.
Reward signal: execution correctness + Gemini-judged reasoning quality.
Eval: Pass@1 on LiveCodeBench v4 (held-out, never seen during training).

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd GRPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Transfer data (not in git — large files)

```bash
# From your local machine, copy data to the server
rsync -avz data/ user@server:/path/to/GRPO/data/
```

### 3. Credentials

**Gemini judge (Google Cloud):**

Create `.env` in the project root:
```
api_key=YOUR_GEMINI_API_KEY
```
Get the key from: GCP Console → API Keys (project: `grpo-reasoning-2`)

**Weights & Biases:**

1. Create a free account at [wandb.ai](https://wandb.ai)
2. On the server, run:
```bash
wandb login
```
It will print a URL — open it, copy your API key, paste it back. Training logs automatically to project `grpo-code-gen`.

**Hugging Face (for pushing checkpoints to Hub):**

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens → create a token with **write** permission
3. On the server:
```bash
export HF_TOKEN=your_hf_token
```
Note: checkpoints are also saved locally to `checkpoints/` regardless. HF Hub is optional but recommended for backup.

### 4. Config

Edit `config.py` before training:
```python
HUB_MODEL_ID = "your-hf-username/grpo-qwen-coder"  # where checkpoints are pushed to HF Hub
```

---

## Running

### Step 1 — Logic and sandbox smoke test (no GPU, no API calls)
```bash
python smoke_test.py
```
Tests imports, config logic, reward computation, and sandbox execution. Fast (~1 min). Run this first.

### Step 2 — Full pipeline smoke test (no GPU needed)
```bash
python train.py --smoke-test
```
Runs 2 training steps with 1.5B model on CPU. Tests the full GRPOTrainer pipeline end-to-end.
On CPU: ~20 min. On GPU: ~2 min.

### Step 3 — Verify Gemini API (on the cloud, before full training)
```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from reward.judge import score_batch
scores = score_batch(['Test problem'], ['[STEP] thinking about this'], ['medium'])
print('Gemini OK:', scores)
"
```
Neither smoke test makes a live Gemini call — run this once on the server to confirm the API key works.

### Step 4 — Baseline eval (optional, before training)
```bash
python eval.py --model Qwen/Qwen2.5-Coder-7B-Instruct --output baseline_results.json
```

### Step 5 — Full training (A100 80GB)
```bash
python train.py
```
~2000 steps, ~24–36 hours. Checkpoints saved every 200 steps to `checkpoints/` and pushed to HF Hub.
wandb and vLLM are only exercised here — the first few steps will confirm they work.

### Eval after training
```bash
# From local checkpoint
python eval.py --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo --output final_results.json

# From HF Hub
python eval.py --model your-hf-username/grpo-qwen-coder --output final_results.json
```

### What each step covers

| | `smoke_test.py` | `train.py --smoke-test` | Full training |
|---|---|---|---|
| Imports + config | ✓ | ✓ | ✓ |
| Sandbox execution | ✓ | ✓ | ✓ |
| reward_fn logic | ✓ | ✓ | ✓ |
| Data loading | ✓ | ✓ | ✓ |
| GRPOTrainer loop | — | ✓ | ✓ |
| GPU | — | — | ✓ |
| vLLM | — | — | ✓ |
| Live Gemini API | — | — | ✓ (step 3 above) |
| wandb logging | — | — | ✓ |
| HF Hub push | — | — | ✓ |

---

## Monitoring

Training logs to wandb project `grpo-code-gen`. Check the dashboard for:

| Metric | Healthy range | Warning |
|--------|--------------|---------|
| `grpo/reward_std_mean` | > 0.05 | GRPO signal collapsing if near 0 |
| `grpo/all_zero_fraction` | < 0.3 | GRPO collapse if high — advantage undefined |
| `grpo/all_perfect_fraction` | < 0.5 | Problems too easy — no learning contrast |
| `reward/non_zero_fraction` | > 0.3 | Model not learning output format |
| `reward/execution_mean` | trending up | Should improve over training |
| `reward/reasoning_mean` | trending up | Reasoning signal contributing |
| `exec/zero_fraction` | < 0.5 | High = formatting issue or sandbox failure |
| `exec/nonzero_mean` | > 0.15 | Low = model capability issue (code runs but fails tests) |
| `exec/timeout_fraction` | < 0.3 | High = increase `EXEC_TIMEOUT` in config |
| `presence/mean` | trending up | Model structuring its thinking |
| `gen/valid_code_fraction` | > 0.7 | Model stopped producing `<code>` tags |
| `gen/has_reasoning_fraction` | > 0.5 | Model stopped generating `<think>` blocks |
| `gen/empty_completion_fraction` | < 0.1 | Model stopped generating output entirely |
| `judge/gemini_mean` | 0.3–0.8 | Stuck near 0.5 = silent API failure |
| `kl` | 0.01–0.5 | Diverging if > 1.0 |

Warnings fire automatically on console **and** in wandb when thresholds are breached:
- Binary `warn/*` flags logged per step so you can see exactly when each warning fired on the dashboard
- `wandb.alert()` sends a push notification (email/Slack) for each triggered warning

Key diagnostic distinctions:
- `exec/zero_fraction` high + `gen/valid_code_fraction` low → **formatting breakdown** (no `<code>` tags)
- `exec/zero_fraction` high + code blocks present → **sandbox/test case issue**
- `exec/nonzero_mean` low (< 0.15) → **model capability** issue, not formatting
- `gen/empty_completion_fraction` high → **generation stopped entirely** (most severe)

---

## File Structure

```
GRPO/
├── config.py                  # all hyperparameters — edit before training
├── train.py                   # GRPO training loop
├── eval.py                    # LiveCodeBench v4 evaluation
├── smoke_test.py              # pipeline sanity check (no GPU needed)
├── .env                       # GEMINI_API_KEY (not in git)
├── data/
│   └── clean/
│       ├── apps_clean.jsonl       # 2739 APPS problems (not in git)
│       ├── lcb_seen_clean.jsonl   # 325 LCB training problems (not in git)
│       └── lcb_unseen_clean.jsonl # 387 LCB eval problems (not in git)
├── reward/
│   ├── execution.py           # subprocess sandbox, test case scoring
│   ├── judge.py               # Gemini judge via GCP Vertex AI
│   └── reward.py              # combines execution + reasoning
└── sandbox/
    └── testing_util.py        # Hendrycks APPS harness (do not modify)
```

---

## Key Config Values

| Parameter | Value | Note |
|-----------|-------|------|
| `GROUP_SIZE` | 8 | rollouts per problem per step |
| `BATCH_SIZE` | 4 | problems per step |
| `MAX_TRAINING_STEPS` | 2000 | total training steps |
| `LEARNING_RATE` | 1e-6 | |
| `LORA_RANK` | 8 | |
| `JUDGE_MODEL` | `gemini-2.5-flash-lite` | GCP project: `grpo-reasoning-2` |
