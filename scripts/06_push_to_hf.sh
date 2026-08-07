#!/usr/bin/env bash
# Stage 6: build a local Hugging Face Hub export (Parquet + dataset card)
# and push it to the private dataset repo. Requires HF auth to already be
# set up -- this script deliberately does not touch or ask for a token:
#
#   huggingface-cli login          # interactive, or:
#   export HF_TOKEN=hf_...         # from https://huggingface.co/settings/tokens
#
#   ./scripts/06_push_to_hf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ID="SorbonneUniversity/LAD-Collected-Dataset"

[ -d .venv ] || ./scripts/00_setup.sh
LAD=".venv/bin/lad"

echo "=== Building local export (data/hf_export/) ==="
$LAD build-hf-export

echo ""
echo "=== Pushing to https://huggingface.co/datasets/$REPO_ID ==="
.venv/bin/python3 -c "
from huggingface_hub import HfApi

api = HfApi()
repo_id = '$REPO_ID'

api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(
    folder_path='data/hf_export',
    repo_id=repo_id,
    repo_type='dataset',
    commit_message='Update LAD collected dataset',
)
print(f'Done: https://huggingface.co/datasets/{repo_id}')
"
