#!/usr/bin/env bash
set -euo pipefail

# Pull latest eval artifacts folder from remote server into local results/eval.
#
# Defaults are set for current project setup but can be overridden:
#   REMOTE_HOST, REMOTE_USER, REMOTE_PORT, SSH_KEY, REMOTE_EVAL_DIR, LOCAL_EVAL_DIR
#
# Example:
#   ./scripts/pull_latest_eval.sh
#   REMOTE_HOST=1.2.3.4 SSH_KEY=~/.ssh/my_key ./scripts/pull_latest_eval.sh

REMOTE_HOST="${REMOTE_HOST:-95.216.229.231}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PORT="${REMOTE_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/primeintellect_ed25519}"
REMOTE_EVAL_DIR="${REMOTE_EVAL_DIR:-/root/grpo-code-reasoning/results/eval}"
LOCAL_EVAL_DIR="${LOCAL_EVAL_DIR:-results/eval}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $0

Pull latest eval artifacts from remote:
- latest timestamp run folder under REMOTE_EVAL_DIR
- pulls summary.json and details.jsonl (if present)

Config via env vars:
  REMOTE_HOST     (default: $REMOTE_HOST)
  REMOTE_USER     (default: $REMOTE_USER)
  REMOTE_PORT     (default: $REMOTE_PORT)
  SSH_KEY         (default: $SSH_KEY)
  REMOTE_EVAL_DIR (default: $REMOTE_EVAL_DIR)
  LOCAL_EVAL_DIR  (default: $LOCAL_EVAL_DIR)
EOF
  exit 0
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: SSH key not found at: $SSH_KEY" >&2
  exit 1
fi

mkdir -p "$LOCAL_EVAL_DIR"

SSH_BASE=(ssh -i "$SSH_KEY" -p "$REMOTE_PORT")
SCP_BASE=(scp -i "$SSH_KEY" -P "$REMOTE_PORT")
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

latest_run_dir="$("${SSH_BASE[@]}" "$REMOTE" "ls -1dt '${REMOTE_EVAL_DIR}'/*/ 2>/dev/null | head -n 1" || true)"

if [[ -z "$latest_run_dir" ]]; then
  echo "No run directory found in ${REMOTE}:${REMOTE_EVAL_DIR}"
  exit 1
fi

latest_run_dir="${latest_run_dir%/}"
run_base="$(basename "$latest_run_dir")"
remote_summary="${latest_run_dir}/summary.json"
remote_details="${latest_run_dir}/details.jsonl"
local_run_dir="${LOCAL_EVAL_DIR}/${run_base}"
mkdir -p "$local_run_dir"

echo "Latest run directory: $run_base"
echo "Pulling summary -> $local_run_dir/"
"${SCP_BASE[@]}" "$REMOTE:$remote_summary" "$local_run_dir/"

if "${SSH_BASE[@]}" "$REMOTE" "test -f '$remote_details'"; then
  echo "Pulling details -> $local_run_dir/"
  "${SCP_BASE[@]}" "$REMOTE:$remote_details" "$local_run_dir/"
else
  echo "No matching details file found for latest run (this is normal if --save-debug-details was not used)."
fi

echo "Done. Local files:"
ls -lh "${local_run_dir}/summary.json" "${local_run_dir}/details.jsonl" 2>/dev/null || \
  ls -lh "${local_run_dir}/summary.json"
