#!/usr/bin/env bash
# Stage 8: run one term through the full pipeline (lexical enrichment ->
# retrieval -> synthesis) and print the structured result.
#
#   ./scripts/08_query.sh "manuscript" en
#   ./scripts/08_query.sh --no-synthesis "gilding" en   # retrieval only, no Claude call
#
# Needs ANTHROPIC_API_KEY set for synthesis (get one at
# console.anthropic.com) -- same pattern as EUROPEANA_API_KEY, not set by
# this script. Export it yourself or put it in .env first.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh

ARGS=()
TERM=""
LANG=""
for arg in "$@"; do
  case "$arg" in
    --no-synthesis) ARGS+=("--no-synthesis") ;;
    *) if [ -z "$TERM" ]; then TERM="$arg"; else LANG="$arg"; fi ;;
  esac
done

if [ -z "$TERM" ] || [ -z "$LANG" ]; then
  echo "Usage: $0 [--no-synthesis] <term> <lang: ar|en|fr>" >&2
  exit 1
fi

.venv/bin/lad query "$TERM" --lang "$LANG" "${ARGS[@]}"
