#!/usr/bin/env bash
# Stage 2: compact finalized JSONL output into partitioned Parquet.
#
#   ./scripts/02_compact.sh                  # all sources
#   ./scripts/02_compact.sh --source europeana
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
LAD=".venv/bin/lad"

if [ "${1:-}" = "--source" ]; then
  $LAD compact --source "$2"
else
  $LAD compact --all
fi
