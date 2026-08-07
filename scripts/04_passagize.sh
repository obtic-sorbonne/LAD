#!/usr/bin/env bash
# Stage 4: chunk harvested records into retrieval-unit passages
# (data/passages/<source>.jsonl) for the RAG index to consume.
#
#   ./scripts/04_passagize.sh                  # all sources
#   ./scripts/04_passagize.sh --source europeana
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
LAD=".venv/bin/lad"

if [ "${1:-}" = "--source" ]; then
  $LAD passagize --source "$2"
else
  $LAD passagize --all
fi
