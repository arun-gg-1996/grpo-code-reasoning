#!/usr/bin/env bash
set -euo pipefail

# Pull latest eval artifacts (summary + details) from remote server
# into local results/eval folder.
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
- latest *_summary.json
- matching *_details.jsonl (if present)

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

latest_summary="$("${SSH_BASE[@]}" "$REMOTE" "ls -1t '${REMOTE_EVAL_DIR}'/*_summary.json 2>/dev/null | head -n 1" || true)"

if [[ -z "$latest_summary" ]]; then
  echo "No summary file found in ${REMOTE}:${REMOTE_EVAL_DIR}"
  exit 1
fi

latest_details="${latest_summary%_summary.json}_details.jsonl"
run_base="$(basename "${latest_summary%_summary.json}")"

echo "Latest run: $run_base"
echo "Pulling summary -> $LOCAL_EVAL_DIR/"
"${SCP_BASE[@]}" "$REMOTE:$latest_summary" "$LOCAL_EVAL_DIR/"

if "${SSH_BASE[@]}" "$REMOTE" "test -f '$latest_details'"; then
  echo "Pulling details -> $LOCAL_EVAL_DIR/"
  "${SCP_BASE[@]}" "$REMOTE:$latest_details" "$LOCAL_EVAL_DIR/"
else
  echo "No matching details file found for latest run (this is normal if --save-debug-details was not used)."
fi

echo "Done. Local files:"
ls -lh "$LOCAL_EVAL_DIR"/"${run_base}"_summary.json "$LOCAL_EVAL_DIR"/"${run_base}"_details.jsonl 2>/dev/null || \
  ls -lh "$LOCAL_EVAL_DIR"/"${run_base}"_summary.json
