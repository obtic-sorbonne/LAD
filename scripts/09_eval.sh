#!/usr/bin/env bash
# Stage 9: run the interim gold-standard set through the pipeline and
# print Part C metrics (retrieval P@5/R@10/MRR, synthesis attestation
# accuracy / equivalence correctness). Builds the gold set first if it
# doesn't exist yet. Not comparable to the LAD paper's reported numbers --
# see rag/eval/metrics.py.
#
#   ./scripts/09_eval.sh
#   ./scripts/09_eval.sh --no-synthesis   # retrieval metrics only, no Claude calls needed
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
.venv/bin/lad eval "$@"
