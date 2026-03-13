# GRPO Code Reasoning

Fine-tune **Qwen2.5-Coder-7B-Instruct** with GRPO to improve competitive programming
performance, measured by Pass@1 and Pass@3 on **LiveCodeBench v4**.

The goal is to teach the model to reason through problems step-by-step before coding,
not just pattern-match to solutions. The reward signal is RL-only — no imitation learning.

---

## What I'm Building

A GRPO training loop where for each problem the model generates 16 candidate solutions.
Each solution is scored on two axes:

| Component | How | Weight (APPS) | Weight (LCB) |
|---|---|---|---|
| Execution | fraction of test cases passed | 0.75 | 0.60 |
| Reasoning | judge scores the `<think>` block vs. reference steps | 0.25 | 0.40 |

The model learns to write better code *and* reason more carefully about the approach.

LCB (LeetCode subset) is never seen in training, so improvement there is a clean signal
of genuine reasoning transfer.

---

## The Hypothesis

Supervised fine-tuning on code teaches syntax and patterns.
GRPO with an execution reward teaches correctness under test cases.
Adding a *reasoning* reward — scoring the step-by-step think trace against a reference —
should push the model toward solution strategies that generalize to unseen problems.

---

## Training Data

| Source | Problems | Notes |
|---|---|---|
| APPS introductory | 2392 | mix of stdin and function-call |
| APPS competition | 347 | stdin only, harder |
| LCB seen (pre-2024) | 325 | pulled from LiveCodeBench, 2× weight |

APPS interview excluded — 53/57 failed sandbox (Python 2 contamination).

---

## Evaluation

**LiveCodeBench v4** — 712 problems (AtCoder, LeetCode, Codeforces).
Pass@1 and Pass@3 computed from n=10 generations per problem using the unbiased estimator.

Key slice: **LeetCode (319 problems)** — function-call format never seen in training.
If this improves, it's reasoning generalization, not format memorization.

---

## Architecture

```
Training GPU (A100 80GB)          Judge GPU (A100 40GB)
─────────────────────────         ──────────────────────────
vLLM rollout server               vLLM judge server
  generate G=16 completions         score reasoning traces
QLoRA trainer                     Sandbox workers (16x pool)
  update LoRA weights               run code, score test cases
```

- **Execution scoring**: multiprocessing pool (spawn→fork), 16 workers, parallel per step
- **Reasoning scoring**: all judge calls sent concurrently via `asyncio` + `AsyncOpenAI`
- **LoRA**: rank=8, alpha=16, all attention + MLP layers, no quantization

---
Local smoke test uses Qwen2.5-Coder-1.5B (training) + Qwen2.5-Coder-0.5B (judge).

---

## Progress

```
✅ Phase 1 — Local Mac
   ✅ APPS + LCB dataset validation and cleaning
   ✅ Sandbox (testing_util + subprocess isolation)
   ✅ Claude Haiku reference reasoning traces (APPS)
   ✅ Claude Sonnet reference solutions (LCB)
   ✅ reward/execution.py
   ✅ reward/judge.py (async)
   ✅ reward/reward.py (2-component, DAPO metrics)
   ✅ config.py

⬜ Phase 2 — Smoke test (small models, Mac)
   ⬜ End-to-end loop: 10–20 steps
   ⬜ Output parsing + reward sanity check

⬜ Phase 3 — Cloud
   ⬜ Baseline eval (pre-training)
   ⬜ Full training run
   ⬜ Final eval + Pass@1 / Pass@3 comparison
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` for the offline reference generation scripts.
Set `HF_TOKEN` if datasets require authentication.

---

## File Structure

```
GRPO/
  sandbox/
    testing_util.py          # Hendrycks APPS harness (do not modify)
  reward/
    execution.py             # subprocess sandbox scoring
    judge.py                 # async vLLM judge client
    reward.py                # combines execution + reasoning
  scripts/
    validate_apps.py         # APPS dataset validation
    build_lcb_data.py        # LCB dataset parsing
    generate_reasoning.py    # Claude Haiku reference think traces
    generate_lcb_solutions.py# Claude Sonnet LCB solutions
  data/
    clean/
      apps_clean.jsonl       # 2739 validated problems
      lcb_seen_clean.jsonl   # 325 problems
  config.py                  # all hyperparameters and paths
  train.py                   # GRPOTrainer loop (TODO)
  eval.py                    # LiveCodeBench evaluation (TODO)
```
