#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="cover"
if [[ $# -gt 0 && ( "$1" == "greedy" || "$1" == "cover" ) ]]; then
  MODEL="$1"
  shift
fi

INPUT="${1:-}"
TOKEN="${2:-}"
if [[ -z "$INPUT" ]]; then
  echo "Usage: $0 [cover|greedy] <paths.jsonl[.gz]> [token]" >&2
  exit 1
fi

# To run the program "./run.sh greedy ../dataset/out/cogent/paths.jsonl.gz cogent"
# Name outputs by dataset token if present in the path (abilene/cogent/geant).
if [[ -z "$TOKEN" ]]; then
  base=$(basename "$INPUT")
  if [[ "$base" == *abilene* ]]; then TOKEN="abilene"; fi
  if [[ "$base" == *cogent* ]]; then TOKEN="cogent"; fi
  if [[ "$base" == *geant* ]]; then TOKEN="geant"; fi
fi
if [[ -z "$TOKEN" ]]; then
  TOKEN="run"
fi

if [[ "$MODEL" == "greedy" ]]; then
  basename=$(basename "$INPUT")
  OUT_DIR="${OUT_DIR:-out/${TOKEN}/greedy}"
  cmd=(python -m greedy.main --input "$INPUT" --out-dir "$OUT_DIR")
elif [[ "$MODEL" == "cover" ]]; then
  OUT_DIR="${OUT_DIR:-out/${TOKEN}/cover}"
  cmd=(python -m main --input "$INPUT" --out-dir "$OUT_DIR")
fi

echo "[run] ${cmd[*]}"
"${cmd[@]}"