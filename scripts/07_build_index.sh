#!/usr/bin/env bash
# Stage 7: build the FAISS passage index (one per language: ar/en/fr) from
# already-passagized data. Defaults to the museum-specific subset (Getty
# AAT + UNESCO Thesaurus + Europeana + WDL) -- see rag/index.py.
#
#   ./scripts/07_build_index.sh
#   ./scripts/07_build_index.sh --sources getty_aat,unesco_thesaurus
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
.venv/bin/lad build-index "$@"
