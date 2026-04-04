#!/usr/bin/env bash
set -euo pipefail

# Pull latest training artifacts folder from remote server into local results/train.
#
# Defaults can be overridden:
#   REMOTE_USER, REMOTE_PORT, SSH_KEY, REMOTE_TRAIN_DIR, LOCAL_TRAIN_DIR
#
# Example:
#   ./scripts/pull_latest_train.sh
#   SSH_KEY=~/.ssh/my_key ./scripts/pull_latest_train.sh

REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_PORT="${REMOTE_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/primeintellect_ed25519}"
REMOTE_TRAIN_DIR="${REMOTE_TRAIN_DIR:-/home/${REMOTE_USER}/grpo-code-reasoning/results/train}"
LOCAL_TRAIN_DIR="${LOCAL_TRAIN_DIR:-results/train}"

if [[ ! -t 0 ]]; then
  echo "ERROR: This script requires interactive input for remote host/IP." >&2
  exit 1
fi
read -r -p "Enter remote host/IP: " REMOTE_HOST
if [[ -z "$REMOTE_HOST" ]]; then
  echo "ERROR: remote host/IP is required." >&2
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $0

Pull latest train artifacts from remote:
- latest timestamp run folder under REMOTE_TRAIN_DIR
- pulls summary.json and train_details.jsonl (if present)

Config via env vars:
  REMOTE_USER      (default: $REMOTE_USER)
  REMOTE_PORT      (default: $REMOTE_PORT)
  SSH_KEY          (default: $SSH_KEY)
  REMOTE_TRAIN_DIR (default: $REMOTE_TRAIN_DIR)
  LOCAL_TRAIN_DIR  (default: $LOCAL_TRAIN_DIR)
EOF
  exit 0
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: SSH key not found at: $SSH_KEY" >&2
  exit 1
fi

mkdir -p "$LOCAL_TRAIN_DIR"

SSH_BASE=(ssh -i "$SSH_KEY" -p "$REMOTE_PORT")
SCP_BASE=(scp -i "$SSH_KEY" -P "$REMOTE_PORT")
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

latest_run_dir="$("${SSH_BASE[@]}" "$REMOTE" "ls -1dt '${REMOTE_TRAIN_DIR}'/*/ 2>/dev/null | head -n 1" || true)"

if [[ -z "$latest_run_dir" ]]; then
  echo "No run directory found in ${REMOTE}:${REMOTE_TRAIN_DIR}"
  exit 1
fi

latest_run_dir="${latest_run_dir%/}"
run_base="$(basename "$latest_run_dir")"
remote_summary="${latest_run_dir}/summary.json"
remote_details="${latest_run_dir}/train_details.jsonl"
local_run_dir="${LOCAL_TRAIN_DIR}/${run_base}"
mkdir -p "$local_run_dir"

echo "Latest run directory: $run_base"
pulled_any=0

if "${SSH_BASE[@]}" "$REMOTE" "test -f '$remote_summary'"; then
  echo "Pulling summary -> $local_run_dir/"
  "${SCP_BASE[@]}" "$REMOTE:$remote_summary" "$local_run_dir/"
  pulled_any=1
else
  echo "summary.json not found yet."
fi

if "${SSH_BASE[@]}" "$REMOTE" "test -f '$remote_details'"; then
  echo "Pulling train details -> $local_run_dir/"
  "${SCP_BASE[@]}" "$REMOTE:$remote_details" "$local_run_dir/"
  pulled_any=1
else
  echo "train_details.jsonl not found yet (or --save-debug-details was not used)."
fi

if [[ "$pulled_any" -eq 0 ]]; then
  echo "No files available yet in latest run folder. Try again in a minute."
  exit 0
fi

echo "Done. Local files:"
ls -lh "${local_run_dir}/summary.json" "${local_run_dir}/train_details.jsonl" 2>/dev/null || \
  ls -lh "${local_run_dir}/summary.json" "${local_run_dir}/train_details.jsonl" 2>/dev/null || true
