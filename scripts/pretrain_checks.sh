#!/usr/bin/env bash
set -euo pipefail

# Optional toggles:
#   SKIP_GEMINI=1         -> skip live Gemini API check
#   RUN_TRAIN_SMOKE=1     -> run `python train.py --smoke-test`
#   PRETRAIN_STRICT_ENV=1 -> fail if gemini_api_key is missing
#   PRETRAIN_STRICT_TORCH=1 -> fail if torch import probe fails

SKIP_GEMINI="${SKIP_GEMINI:-0}"
RUN_TRAIN_SMOKE="${RUN_TRAIN_SMOKE:-0}"
PRETRAIN_STRICT_ENV="${PRETRAIN_STRICT_ENV:-0}"
PRETRAIN_STRICT_TORCH="${PRETRAIN_STRICT_TORCH:-0}"

echo "== GRPO pre-train checks =="
echo "cwd: $(pwd)"

if command -v nvidia-smi >/dev/null 2>&1; then
  printf "\n[1/6] GPU status\n"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
else
  printf "\n[1/6] GPU status\n"
  echo "nvidia-smi not found (continuing)"
fi

printf "\n[2/6] Python + torch\n"
if python - <<'PY'
import sys
print('python:', sys.version.split()[0])
try:
    import torch
    print('torch:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('cuda_device:', torch.cuda.get_device_name(0))
except Exception as e:
    print('torch_check_error:', e)
    raise
PY
then
  true
else
  echo "WARNING: torch import check failed."
  if [[ "$PRETRAIN_STRICT_TORCH" == "1" || "$PRETRAIN_STRICT_ENV" == "1" ]]; then
    echo "PRETRAIN_STRICT_TORCH=1 or PRETRAIN_STRICT_ENV=1 set; failing."
    exit 1
  fi
fi

printf "\n[3/6] Config, credentials, and data files\n"
python - <<'PY'
import os
from config import APPS_CLEAN_PATH, LCB_SEEN_PATH, LCB_EVAL_PATH, GEMINI_API_KEY, HUB_MODEL_ID

for label, path in [
    ('APPS', APPS_CLEAN_PATH),
    ('LCB train', LCB_SEEN_PATH),
    ('LCB eval', LCB_EVAL_PATH),
]:
    ok = os.path.exists(path)
    print(f'{label}: {path} -> {"OK" if ok else "MISSING"}')
    if not ok:
        raise FileNotFoundError(path)

print('HF hub model id:', HUB_MODEL_ID)
print('gemini_api_key_present:', bool(GEMINI_API_KEY))
PY

if [[ "$PRETRAIN_STRICT_ENV" == "1" ]]; then
  printf "\n[3b/6] Strict env check\n"
  python - <<'PY'
from config import GEMINI_API_KEY
assert GEMINI_API_KEY, 'gemini_api_key is missing but PRETRAIN_STRICT_ENV=1'
print('strict env: PASS')
PY
fi

printf "\n[4/6] Core smoke test\n"
python smoke_test.py

if [[ "$SKIP_GEMINI" != "1" ]]; then
  printf "\n[5/6] Live Gemini API check\n"
  python - <<'PY'
from reward.judge import score_batch, get_last_batch_stats

problems = [
    "Print hello world",
    "Given an integer n, return n squared",
]
think_blocks = [
    "[STEP] Output a constant string using print to stdout.",
    "[STEP] Read input n, compute n*n, then print the result.",
]
difficulties = ["medium", "medium"]

scores = score_batch(problems, think_blocks, difficulties)
stats = get_last_batch_stats()

print('gemini_score:', scores)
print('gemini_stats:', stats)
assert len(scores) == len(problems)
assert all(0.0 <= s <= 1.0 for s in scores)
assert stats.get("fallback_fraction", 1.0) < 1.0, "all Gemini calls fell back"
print('gemini_check: PASS')
PY
else
  printf "\n[5/6] Live Gemini API check skipped (SKIP_GEMINI=1)\n"
fi

if [[ "$RUN_TRAIN_SMOKE" == "1" ]]; then
  printf "\n[6/6] Trainer smoke test\n"
  python train.py --smoke-test
else
  printf "\n[6/6] Trainer smoke test skipped (set RUN_TRAIN_SMOKE=1 to enable)\n"
fi

printf "\nAll pre-train checks passed.\n"
