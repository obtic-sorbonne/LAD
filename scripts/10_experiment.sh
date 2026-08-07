#!/usr/bin/env bash
# Stage 10: interactive live-experiment REPL. Loads LaBSE + the FAISS
# index ONCE, then lets you query as many terms as you want -- unlike
# 08_query.sh, which reloads the model (~15s) on every single call.
#
#   ./scripts/10_experiment.sh
#   ./scripts/10_experiment.sh --no-synthesis   # retrieval only, no API key needed
#
# Requires the index to already exist (./scripts/07_build_index.sh).
# Synthesis needs ANTHROPIC_API_KEY -- paste yours below, or export it /
# put it in .env yourself. Without one, use --no-synthesis.
set -euo pipefail
cd "$(dirname "$0")/.."

# ============================================================
# API KEY -- paste yours between the quotes below.
# ============================================================
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"      # https://console.anthropic.com/settings/keys
export ANTHROPIC_API_KEY

[ -d .venv ] || ./scripts/00_setup.sh

if [ -z "$ANTHROPIC_API_KEY" ] && [[ "$*" != *"--no-synthesis"* ]]; then
  echo "No ANTHROPIC_API_KEY set -- running with --no-synthesis (retrieval only)." >&2
  echo "Paste a key into this script, or export ANTHROPIC_API_KEY, to get live synthesis too." >&2
  set -- "$@" --no-synthesis
fi

.venv/bin/lad repl "$@"
