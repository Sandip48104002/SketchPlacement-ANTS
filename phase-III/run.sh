#!/usr/bin/env bash
# Phase III — CMS Simulation
#
# Usage:
#   # Single strategy, no baseline
#   ./run.sh <paths.jsonl.gz> <assignments.json>
#
#   # All three strategies + baseline comparison
#   ./run.sh <paths.jsonl.gz> \
#       <balance_size/assignments.json> \
#       <balance_count/assignments.json> \
#       <first_match/assignments.json> \
#       --baseline
#
# Pass --baseline as the last argument to include the full-deployment baseline.

set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PATHS="${1:-}"
if [[ -z "$PATHS" ]]; then
  echo "Usage: $0 <paths.jsonl.gz> <assignments.json> [more ...] [--baseline]" >&2
  exit 1
fi
shift

# Separate --baseline flag from assignment files.
ASSIGNMENTS=()
BASELINE_FLAG=""
for arg in "$@"; do
  if [[ "$arg" == "--baseline" ]]; then
    BASELINE_FLAG="--baseline"
  else
    ASSIGNMENTS+=("$arg")
  fi
done

if [[ ${#ASSIGNMENTS[@]} -eq 0 ]]; then
  echo "Error: at least one assignments.json required." >&2
  exit 1
fi

# Auto-detect output dir token from first assignments path.
first_asgn="${ASSIGNMENTS[0]}"
token=$(basename "$(dirname "$(dirname "$first_asgn")")")
OUT_DIR="${OUT_DIR:-out/${token}}"

cmd=(python simulate.py --paths "$PATHS" --out-dir "$OUT_DIR" --assignments "${ASSIGNMENTS[@]}")
if [[ -n "$BASELINE_FLAG" ]]; then
  cmd+=("$BASELINE_FLAG")
fi

echo "[run] ${cmd[*]}"
"${cmd[@]}"