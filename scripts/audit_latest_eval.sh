#!/usr/bin/env bash
set -euo pipefail

# Audit likely false negatives for the latest local eval run.
#
# Usage:
#   bash scripts/audit_latest_eval.sh
#   MAX_CASES=60 RELAXED_TIMEOUT=30 bash scripts/audit_latest_eval.sh

EVAL_ROOT="${EVAL_ROOT:-results/eval}"
MAX_CASES="${MAX_CASES:-50}"
RELAXED_TIMEOUT="${RELAXED_TIMEOUT:-20}"

latest_details="$(find "$EVAL_ROOT" -type f -name "details.jsonl" | sort | tail -n 1)"
if [[ -z "${latest_details:-}" ]]; then
  echo "No details.jsonl found under $EVAL_ROOT"
  exit 1
fi

run_dir="$(dirname "$latest_details")"
out_json="${run_dir}/false_negative_audit.json"

echo "Auditing: $latest_details"
python scripts/audit_false_negatives.py \
  --details "$latest_details" \
  --max-cases "$MAX_CASES" \
  --relaxed-timeout "$RELAXED_TIMEOUT" \
  --out "$out_json"

echo
echo "Saved report: $out_json"
