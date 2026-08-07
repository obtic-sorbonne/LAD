#!/usr/bin/env bash
# Stage 1: harvest one or all sources.
#
#   ./scripts/01_harvest.sh                          # all enabled sources, PAGE_LIMIT pages each
#   ./scripts/01_harvest.sh --source europeana        # one source only
#   ./scripts/01_harvest.sh --source europeana --clean --refresh  # wipe + full re-harvest
#
# Env vars (or edit the defaults below):
#   EUROPEANA_API_KEY   free key from europeana.eu -- source is skipped without one
#   PAGE_LIMIT           pages fetched per source this run (~100 records/page).
#                         Empty string removes the cap. Default 5000 keeps a
#                         first run fast -- World Digital Library alone has
#                         200+ pages, UNESDOC/Europeana can have thousands.
set -euo pipefail
cd "$(dirname "$0")/.."

# ============================================================
# API KEYS -- paste yours between the quotes below.
# ============================================================
EUROPEANA_API_KEY="${EUROPEANA_API_KEY:-moglemon}"      # https://www.europeana.eu/ (free, instant)
export EUROPEANA_API_KEY

PAGE_LIMIT="${PAGE_LIMIT:-5000}"

SOURCE=""
ALL=false
REFRESH=false
CLEAN=false

while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --all) ALL=true; shift ;;
    --refresh) REFRESH=true; shift ;;
    --clean) CLEAN=true; shift ;;
    --page-limit) PAGE_LIMIT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$SOURCE" ] && [ "$ALL" = false ]; then
  ALL=true
fi

[ -d .venv ] || ./scripts/00_setup.sh
LAD=".venv/bin/lad"

if [ "$CLEAN" = true ]; then
  if [ -z "$SOURCE" ]; then
    echo "--clean requires --source <name> (refusing to wipe every source at once)" >&2
    exit 1
  fi
  echo "[harvest] --clean: wiping prior output for $SOURCE"
  rm -f "data/processed/$SOURCE/records.jsonl" "data/processed/$SOURCE/needs_review.jsonl"
  rm -f "data/raw/$SOURCE/checkpoint.json"
  REFRESH=true
fi

PAGE_ARGS=()
[ -n "$PAGE_LIMIT" ] && PAGE_ARGS=(--page-limit "$PAGE_LIMIT")
REFRESH_ARGS=()
[ "$REFRESH" = true ] && REFRESH_ARGS=(--refresh)

echo ""
echo "=== Harvesting ==="
if [ -n "$SOURCE" ]; then
  $LAD harvest --source "$SOURCE" "${PAGE_ARGS[@]}" "${REFRESH_ARGS[@]}"
else
  $LAD harvest --all "${PAGE_ARGS[@]}" "${REFRESH_ARGS[@]}"
fi

echo ""
echo "=== Status ==="
$LAD status
