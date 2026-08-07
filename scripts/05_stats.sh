#!/usr/bin/env bash
# Stage 5: regenerate stats.txt from whatever is currently in
# data/processed/, data/termbase/, data/passages/.
#
#   ./scripts/05_stats.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
.venv/bin/lad stats
