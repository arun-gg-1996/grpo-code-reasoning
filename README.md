# GRPO (Simple Runbook)

This file is intentionally short.

Use this for:

- one-time machine setup
- where to run pre-train checks
- where to find train/eval commands

For day-to-day run commands, use [TEST.md](/Users/arun-ghontale/UB/Personal%20Project/GRPO/TEST.md).

## 1) One-Time Setup (Fresh A100 Ubuntu Machine)

If this is a fresh server, use the setup script:

```bash
bash scripts/setup_persistent_server.sh
```

What it does:

- installs system deps
- creates venv + installs `requirements.txt`
- creates `.env`
- optionally sets `wandb`, `hf`, and `gemini_api_key`

After setup:

```bash
source venv/bin/activate
```

## 2) Existing Machine (Already Setup)

From repo root:

```bash
source venv/bin/activate
```

If you still need API key setup manually, edit `.env`:

```env
gemini_api_key=YOUR_GEMINI_API_KEY
```

And login once:

```bash
wandb login
huggingface-cli login
```

## 3) Before Every Train

Make sure dataset files exist on server:
- `data/clean/apps_clean.jsonl`
- `data/clean/lcb_seen_clean.jsonl`
- `data/clean/lcb_unseen_clean.jsonl`

If missing, sync from your local machine to server:

```bash
rsync -av --progress -e "ssh -i ~/.ssh/primeintellect_ed25519" \
  "/Users/arun-ghontale/UB/Personal Project/GRPO/data/clean/" \
  ubuntu@216.81.248.8:/home/ubuntu/grpo-code-reasoning/data/clean/
```

Then verify on server:

```bash
ls -lh data/clean/apps_clean.jsonl data/clean/lcb_seen_clean.jsonl data/clean/lcb_unseen_clean.jsonl
```

Use exactly one preflight script:

```bash
bash scripts/pretrain_checks.sh
```

That is the only required pre-train check entrypoint.

## 4) Train / Resume / Eval Commands

All canonical commands are in:

- [TEST.md](/Users/arun-ghontale/UB/Personal%20Project/GRPO/TEST.md)

## 5) Source Of Truth

If docs and code differ, trust code:

- [config.py](/Users/arun-ghontale/UB/Personal%20Project/GRPO/config.py)
- [train.py](/Users/arun-ghontale/UB/Personal%20Project/GRPO/train.py)
- [eval.py](/Users/arun-ghontale/UB/Personal%20Project/GRPO/eval.py)
- [METRICS.md](/Users/arun-ghontale/UB/Personal%20Project/GRPO/METRICS.md)
