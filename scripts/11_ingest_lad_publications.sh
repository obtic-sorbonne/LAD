#!/usr/bin/env bash
# Stage 11: ingest the real LAD Publications PDFs (provided directly, not
# fetched by any connector -- must already be present at SOURCE_DIR) and
# passagize them through the same pipeline every other source uses. This
# is the first REAL institutional documentation this project indexes --
# see pipeline/ingest_lad_publications.py and PROJECT_STATUS.md.
#
#   ./scripts/11_ingest_lad_publications.sh
#   ./scripts/11_ingest_lad_publications.sh --source-dir "/path/to/pdfs"
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE_DIR="data/raw_pdfs/lad_publications"

while [ $# -gt 0 ]; do
  case "$1" in
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -d .venv ] || ./scripts/00_setup.sh
LAD=".venv/bin/lad"

echo ""
echo "=== Ingesting LAD Publications from: $SOURCE_DIR ==="
$LAD ingest-lad-publications --source-dir "$SOURCE_DIR"

echo ""
echo "=== Passagizing lad_publications ==="
$LAD passagize --source lad_publications

echo ""
echo "Next: ./scripts/07_build_index.sh --sources getty_aat,unesco_thesaurus,europeana,world_digital_library,lad_publications"
echo "      (add lad_publications to the museum subset to test corpus expansion)"
