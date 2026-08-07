#!/usr/bin/env bash
# Stage 3b: parse the REAL Louvre Abu Dhabi Termbase (data/Export from
# Kalcium.xlsx, institutional data -- not fetched by any connector, must
# already be present) into data/termbase/real_termbase.jsonl. See
# pipeline/build_termbase_from_kalcium.py and README: Termbase.
#
# Run after 03_build_termbase.sh -- lexical_enrichment.py merges both
# termbases, real_termbase.jsonl first.
#
#   ./scripts/03b_build_real_termbase.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
.venv/bin/lad build-real-termbase
