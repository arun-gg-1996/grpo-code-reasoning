# GRPO — Qwen2.5-Coder-7B on Competitive Programming

Fine-tunes **Qwen2.5-Coder-7B-Instruct** with GRPO on APPS + LCB problems.
Reward signal: execution correctness + Gemini-judged reasoning quality.
Eval: Pass@1 and Pass@3 on LiveCodeBench v4 (387 held-out problems, never seen during training).

## Source of Truth

To avoid doc drift, keep these as the only canonical references:
- `config.py` — all tunable hyperparameters, prompts, weights, and paths
- `train.py` / `eval.py` — actual runtime behavior
- `README.md` — operator runbook
- `METRICS.md` — plain-English glossary of custom W&B metrics

`docs/project.md` and `project_desc.md` were intentionally removed because they had stale/duplicate parameter values.

---

## 1. Setup

### 1.1 Clone and install

```bash
git clone <repo>
cd GRPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Transfer data (not in git — large files)

Run this from your **local machine**:
```bash
rsync -avz data/ user@server:/path/to/GRPO/data/
```

### 1.3 Credentials

**Gemini API key:**

Create `.env` in the project root:
```
gemini_api_key=YOUR_GEMINI_API_KEY
```
Get the key from: GCP Console → API Keys → project `grpo-reasoning-2`

**Weights & Biases:**

1. Go to [wandb.ai](https://wandb.ai) → **User Settings → API Keys** → copy your key
2. On the server:
```bash
wandb login
# paste your API key when prompted
```
Training auto-creates the project `grpo-code-gen` on first run.
View at: `wandb.ai/arun-gv-ghontale/grpo-code-gen`

**Hugging Face:**

1. Go to [huggingface.co](https://huggingface.co) → **Settings → Access Tokens → New token** → role: **Write**
2. On the server:
```bash
huggingface-cli login
# paste your token when prompted
```
Checkpoints push to `arun-gv-ghontale/grpo-qwen-coder` automatically every 200 steps.

### 1.4 Verify config

`config.py` is already set correctly. No changes needed unless you want to tune hyperparameters.

```python
HUB_MODEL_ID = "arun-gv-ghontale/grpo-qwen-coder"  # ← already set
```

---

## 2. Smoke Tests

Run both before anything else. Catch environment issues early.

### 2.1 Logic and sandbox test (fast, no GPU, no API calls)

```bash
python smoke_test.py
```

Tests imports, config, reward computation, and sandbox execution. Takes ~1 min.
Expected output: `ALL SMOKE TESTS PASSED`

### 2.2 Full pipeline test (no GPU needed)

```bash
python train.py --smoke-test
```

Runs 2 training steps with 1.5B model on CPU. Tests the full GRPOTrainer loop end-to-end.
Expected: completes without errors in ~2 min on GPU, ~20 min on CPU.

### 2.3 Verify Gemini API (live call — run on the server)

Neither smoke test above makes a real Gemini call. Verify it works before training:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from reward.judge import score_batch
scores = score_batch(['Print hello world'], ['[STEP] Use print statement'], ['medium'])
print('Gemini OK, score:', scores)
"
```

Expected: prints a float score like `[0.6]`. Any non-0.5 value confirms the API is live.

### What each test covers

| | `smoke_test.py` | `train.py --smoke-test` | Full training |
|---|---|---|---|
| Imports + config | ✓ | ✓ | ✓ |
| Sandbox execution | ✓ | ✓ | ✓ |
| reward_fn logic | ✓ | ✓ | ✓ |
| Data loading | ✓ | ✓ | ✓ |
| GRPOTrainer loop | — | ✓ | ✓ |
| GPU | — | — | ✓ |
| vLLM | — | — | ✓ |
| Live Gemini API | — | ✓ (step 2.3) | ✓ |
| wandb logging | — | — | ✓ |
| HF Hub push | — | — | ✓ |

---

## 3. Baseline Eval (before training)

Evaluate the untrained base model on the 387 held-out LCB problems. This is your baseline to compare against after training.

```bash
python eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output results/baseline_results.json
```

Takes ~30–45 min on A100. Generates 5 solutions per problem, scores with the same sandbox used in training.

Results are printed to console as a table and saved to `results/baseline_results.json`.

**Console output looks like:**
```
================================================================
  EVALUATION RESULTS — Qwen/Qwen2.5-Coder-7B-Instruct
================================================================
Split        Problems    pass@1    pass@3
----------------------------------------------------------------
Overall           387    0.XXXX    0.XXXX
Easy              124    0.XXXX    0.XXXX
Medium            143    0.XXXX    0.XXXX
Hard              120    0.XXXX    0.XXXX
LeetCode          169    0.XXXX    0.XXXX
AtCoder           219    0.XXXX    0.XXXX
================================================================
```

**Saved JSON format (`results/baseline_results.json`):**
```json
{
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "timestamp": "2025-XX-XX XX:XX:XX",
  "n_problems": 387,
  "n_generations": 5,
  "problem_counts": { "all": 387, "easy": 124, "medium": 143, "hard": 120, "leetcode": 169, "atcoder": 219 },
  "metrics": {
    "pass@1/all":      0.XXXX,
    "pass@3/all":      0.XXXX,
    "pass@1/easy":     0.XXXX,
    "pass@3/easy":     0.XXXX,
    "pass@1/medium":   0.XXXX,
    "pass@3/medium":   0.XXXX,
    "pass@1/hard":     0.XXXX,
    "pass@3/hard":     0.XXXX,
    "pass@1/leetcode": 0.XXXX,
    "pass@3/leetcode": 0.XXXX,
    "pass@1/atcoder":  0.XXXX,
    "pass@3/atcoder":  0.XXXX
  }
}
```

**Record before training:**

| Split | pass@1 | pass@3 |
|-------|--------|--------|
| Overall (387) | — | — |
| Easy (124) | — | — |
| Medium (143) | — | — |
| Hard (120) | — | — |
| LeetCode (169) | — | — |
| AtCoder (219) | — | — |

Fill in from the console output and keep as your baseline reference.

---

## 4. Training

```bash
python train.py
```

~2000 steps, ~24–36 hours on A100 80GB.

**What happens during training:**
- Curriculum sampling: easy-heavy early (steps 0–300), hard problems introduced at step 800, full distribution by step 1500
- Checkpoints saved locally to `checkpoints/` every 200 steps
- Checkpoints pushed to HF Hub (`arun-gv-ghontale/grpo-qwen-coder`) every 200 steps
- Metrics logged to wandb every step

### 4.1 Monitoring on Weights & Biases

Go to [wandb.ai/arun-gv-ghontale/grpo-code-gen](https://wandb.ai/arun-gv-ghontale/grpo-code-gen) after training starts.

**The most important charts to watch:**

**Is GRPO actually learning?**

| Metric | Healthy | Problem |
|--------|---------|---------|
| `grpo/reward_std_mean` | > 0.05 | Near 0 = all rollouts getting same reward, no learning signal |
| `grpo/all_zero_fraction` | < 0.3 | > 0.5 = model can't solve any variants, GRPO has nothing to learn from |
| `grpo/all_perfect_fraction` | < 0.5 | > 0.5 = problems too easy, no contrast between rollouts |
| `kl` | 0.01–0.5 | > 1.0 = model drifting too far from base, reduce learning rate |

**Is the reward improving?**

| Metric | Healthy | Problem |
|--------|---------|---------|
| `reward/execution_mean` | trending up | Flat = model not improving at solving problems |
| `reward/reasoning_mean` | trending up | Flat = reasoning quality not improving |
| `judge/fallback_fraction` | near 0 | High = Gemini calls failing/parsing fallback |
| `judge/step_json_fraction` | near 1 | Low = Gemini not returning per-step JSON reliably |
| `reward/mean_easy` | should rise first | — |
| `reward/mean_medium` | rises after ~step 500 | — |
| `reward/mean_hard` | rises after ~step 1000 | — |
| `reward/apps_mean` / `reward/lcb_mean` | trending up | Large gap = one source dominating |

**Is the model generating correctly?**

| Metric | Healthy | Problem |
|--------|---------|---------|
| `reward/non_zero_fraction` | > 0.3 | Low = model not learning `<code>` / `<think>` format |
| `gen/valid_code_fraction` | > 0.7 | Low = model stopped producing `<code>` tags |
| `gen/has_reasoning_fraction` | > 0.5 | Low = model stopped producing `<think>` blocks |
| `gen/empty_completion_fraction` | < 0.1 | High = model generating nothing (severe) |
| `exec/zero_fraction` | < 0.5 | See diagnostics below |
| `exec/timeout_fraction` | < 0.3 | High = increase `EXEC_TIMEOUT` in config |

**Is the model generating at the right length?**

| Metric | Healthy | Problem |
|--------|---------|---------|
| `gen/mean_think_chars` | trending up | Down = model reasoning less (collapsing) |
| `gen/mean_code_chars` | slight upward trend | — |
| `gen/likely_truncated_fraction` | near 0 | > 5% = hitting 8192 token cap, increase `MAX_NEW_TOKENS` |

**Is curriculum working?**

| Metric | What to check |
|--------|--------------|
| `data/easy_seen` | Grows fastest in steps 0–300 |
| `data/medium_seen` | Accelerates from step 300 |
| `data/hard_seen` | Accelerates from step 800 |

**Timing, system, and events (new):**

| Metric | What to check |
|--------|----------------|
| `timing/step_total_s` | overall iteration speed |
| `timing/generation_s` | rollout generation time |
| `timing/reward_execution_s` | sandbox scoring time inside reward |
| `timing/reward_judge_s` | Gemini scoring time inside reward |
| `gpu/utilization_percent` | GPU compute saturation |
| `gpu/nvml_mem_used_gb` | total GPU memory in use |
| `gpu/non_torch_mem_gb_est` | rough non-PyTorch memory (vLLM/runtime) |
| `system/ram_percent` | host RAM pressure |
| `event/checkpointing` | checkpoint pulse marker |
| `event/checkpoint_save_s` | checkpoint save duration |
| `event/hf_push_s` | Hugging Face push duration |

**Early warnings** fire automatically as console logs and wandb alerts when thresholds are breached. Binary `warn/*` flags are also plotted per step so you can see exactly when issues started.

**Diagnostic guide:**
- `exec/zero_fraction` high + `gen/valid_code_fraction` low → model stopped producing `<code>` tags (formatting)
- `exec/zero_fraction` high + code blocks present → sandbox or test case issue
- `exec/nonzero_mean` < 0.15 → code runs but fails tests (model capability, not formatting)
- `gen/empty_completion_fraction` high → most severe — model generating nothing at all

### 4.2 Where models are saved

**Locally on the server** — after every 200 steps:
```
checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/
├── checkpoint-200/
├── checkpoint-400/
├── ...
└── checkpoint-2000/   ← final
```

**On Hugging Face Hub** — pushed automatically every 200 steps:
`https://huggingface.co/arun-gv-ghontale/grpo-qwen-coder`

Each push overwrites the previous checkpoint on the Hub (only the latest is kept). Local checkpoints are all retained.

---

## 5. Eval After Training

Run the same eval on the trained model and compare against baseline.

```bash
# From local final checkpoint
python eval.py \
  --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-2000 \
  --output results/trained_results.json

# Or from HF Hub
python eval.py \
  --model arun-gv-ghontale/grpo-qwen-coder \
  --output results/trained_results.json
```

**To eval intermediate checkpoints** (full learning curve):
```bash
mkdir -p results
for step in 200 400 600 800 1000 1200 1400 1600 1800 2000; do
  python eval.py \
    --model checkpoints/Qwen2.5-Coder-7B-Instruct-grpo/checkpoint-$step \
    --output results/checkpoint_${step}_results.json
done
```

### 5.1 Comparing results

Results are in `results/trained_results.json`. Compare against `results/baseline_results.json`:

Same table format as baseline eval — printed to console and saved to `results/trained_results.json`.

**Fill in the comparison after training:**

| Split | Baseline pass@1 | Trained pass@1 | Δ | Baseline pass@3 | Trained pass@3 | Δ |
|-------|----------------|----------------|---|----------------|----------------|---|
| Overall (387) | — | — | — | — | — | — |
| Easy (124) | — | — | — | — | — | — |
| Medium (143) | — | — | — | — | — | — |
| Hard (120) | — | — | — | — | — | — |
| LeetCode (169) | — | — | — | — | — | — |
| AtCoder (219) | — | — | — | — | — | — |

A positive Δ on pass@1/all of ~3–5% is a meaningful result for a 7B model with this dataset size.

---

## File Structure

```
GRPO/
├── config.py                  # all hyperparameters — single source of truth
├── train.py                   # GRPO training loop
├── eval.py                    # LiveCodeBench v4 evaluation (pass@1, pass@3)
├── smoke_test.py              # pipeline sanity check (no GPU needed)
├── .env                       # Gemini API key (not in git)
├── results/                   # eval output files (not in git)
│   ├── baseline_results.json
│   └── trained_results.json
├── checkpoints/               # local model checkpoints (not in git)
├── data/
│   └── clean/
│       ├── apps_clean.jsonl        # 2739 APPS problems (not in git)
│       ├── lcb_seen_clean.jsonl    # 325 LCB training problems (not in git)
│       └── lcb_unseen_clean.jsonl  # 387 LCB eval problems (not in git)
├── reward/
│   ├── execution.py           # subprocess sandbox, parallel test case scoring
│   ├── judge.py               # Gemini judge via GCP API key auth
│   └── reward.py              # combines execution + reasoning, wandb logging
└── sandbox/
    └── testing_util.py        # Hendrycks APPS harness (do not modify)
```

---

## Key Config Values

| Parameter | Value | Note |
|-----------|-------|------|
| `GROUP_SIZE` | 8 | rollouts per problem per step |
| `BATCH_SIZE` | 4 | problems per step → 32 completions per step |
| `MAX_TRAINING_STEPS` | 2000 | ~24–36 hrs on A100 |
| `MAX_NEW_TOKENS` | 8192 | max generation per rollout (think + code) |
| `LEARNING_RATE` | 1e-6 | standard for GRPO on 7B |
| `KL_COEFF` | 0.04 | KL penalty against reference model |
| `LORA_RANK` | 8 | LoRA rank |
| `EXEC_TIMEOUT` | 5s | sandbox timeout per test case execution |
| `SAVE_STEPS` | 200 | checkpoint every N steps |
| `JUDGE_MODEL` | `gemini-2.5-flash-lite` | GCP project: `grpo-reasoning-2` |
