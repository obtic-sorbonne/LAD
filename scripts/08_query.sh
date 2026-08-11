#!/usr/bin/env bash
# Stage 8: run one term through the full pipeline (lexical enrichment ->
# retrieval -> synthesis) and print the structured result.
#
#   ./scripts/08_query.sh "manuscript" en
#   ./scripts/08_query.sh "gilding" en --no-synthesis   # retrieval only, no Claude call
#   ./scripts/08_query.sh "gilding" fr --lexical --rerank --generator claude
#
# Flags forward straight through to `lad query`: --no-synthesis, --rerank,
# --lexical (bare), --generator/--top-k (take a value). Term and lang can
# come in any order relative to the flags, but must be the only two bare
# positional arguments.
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
skip_next=false
for arg in "$@"; do
  if $skip_next; then
    ARGS+=("$arg")
    skip_next=false
    continue
  fi
  case "$arg" in
    --no-synthesis|--rerank|--lexical) ARGS+=("$arg") ;;
    --generator|--top-k) ARGS+=("$arg"); skip_next=true ;;
    --*) echo "Unknown flag: $arg" >&2; exit 1 ;;
    *)
      if [ -z "$TERM" ]; then
        TERM="$arg"
      elif [ -z "$LANG" ]; then
        LANG="$arg"
      else
        echo "Unexpected extra argument: $arg" >&2
        exit 1
      fi
      ;;
  esac
done

if [ -z "$TERM" ] || [ -z "$LANG" ]; then
  echo "Usage: $0 [flags] <term> <lang: ar|en|fr>" >&2
  echo "  flags: --no-synthesis --rerank --lexical --generator claude|jais2 --top-k N" >&2
  exit 1
fi

.venv/bin/lad query "$TERM" --lang "$LANG" "${ARGS[@]}"
