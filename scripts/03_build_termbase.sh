#!/usr/bin/env bash
# Stage 3: build the interim LAD Termbase substitute from already-harvested
# UNESCO Thesaurus + Getty AAT data (no new fetching). See README: Termbase.
#
#   ./scripts/03_build_termbase.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
.venv/bin/lad build-termbase
