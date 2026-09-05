#!/usr/bin/env bash
# Phase II — Flow Assignment
#
# Usage:
#   ./run.sh <paths.jsonl[.gz]> <solution.json> [token] [strategy]
#
# Examples:
#   ./run.sh ../dataset/out/abilene/paths.jsonl.gz ../phase-I/out/abilene/greedy/solution.json
#   ./run.sh ../dataset/out/abilene/paths.jsonl.gz ../phase-I/out/abilene/cover/solution.json abilene_ilp balance_size
#
# strategy: first_match | balance_count | balance_size  (default: balance_size)

set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PATHS="${1:-}"
SOLUTION="${2:-}"
TOKEN="${3:-}"
STRATEGY="${4:-balance_size}"

if [[ -z "$PATHS" || -z "$SOLUTION" ]]; then
  echo "Usage: $0 <paths.jsonl[.gz]> <solution.json> [token] [strategy]" >&2
  exit 1
fi

# Auto-detect token from solution path if not given.
# e.g. ../phase-I/out/abilene/greedy/solution.json  →  token = abilene_greedy
if [[ -z "$TOKEN" ]]; then
  sol_dir=$(dirname "$SOLUTION")
  method=$(basename "$sol_dir")            # greedy / cover
  dataset=$(basename "$(dirname "$sol_dir")")  # abilene / cogent / geant
  if [[ -n "$dataset" && "$dataset" != "." ]]; then
    TOKEN="${dataset}_${method}"
  else
    TOKEN="run"
  fi
fi

OUT_DIR="${OUT_DIR:-out/${TOKEN}/${STRATEGY}}"

cmd=(python -m main --paths "$PATHS" --solution "$SOLUTION" --strategy "$STRATEGY" --out-dir "$OUT_DIR")

echo "[run] ${cmd[*]}"
"${cmd[@]}"