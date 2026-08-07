#!/usr/bin/env bash
# Stage 12: build the LAD Publications gold-standard set -- real termbase
# entries independently confirmed attested in the actual LAD Publications
# text (see rag/eval/lad_publications_gold_set.py). Run after Stage 11
# (ingest + passagize lad_publications) and Stage 3b (real termbase).
#
#   ./scripts/12_build_lad_publications_gold_set.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || ./scripts/00_setup.sh
.venv/bin/python -c "
from lad.rag.eval.lad_publications_gold_set import build_lad_publications_gold_set
path, stats = build_lad_publications_gold_set()
print(f'[done] wrote {stats[\"n_total\"]} entries -> {path}')
print(f'       ({stats[\"n_triple_attested_ar_en_fr\"]} of those are attested in all three languages, AR+EN+FR)')
"
