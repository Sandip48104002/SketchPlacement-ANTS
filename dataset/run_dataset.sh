#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-configs/run.yaml}"

python3 -m src.run -c "$CONFIG_PATH"


