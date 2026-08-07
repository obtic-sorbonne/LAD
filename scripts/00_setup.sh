#!/usr/bin/env bash
# Stage 0: environment setup. Idempotent -- every other script calls this
# automatically, but it's also safe to run standalone.
#
#   ./scripts/00_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer python3.11 explicitly over generic `python3`, if available. This
# matters for GPU support specifically: PyTorch's official CUDA wheels for
# very new Python versions (e.g. 3.14) are only built against CUDA >=12.6,
# but this project's target driver reports CUDA 12.4 -- there is no
# compatible GPU wheel for Python 3.14 at all, only CPU-only ones. Python
# 3.11 has cu118/cu121/cu124 wheels available, which do match a 12.4
# driver. Verified live (driver 550.127.05, CUDA 12.4, 4x
# NVIDIA A40): rebuilding the venv under python3.11 was what actually made
# `torch.cuda.is_available()` return True -- picking whatever `python3`
# happens to resolve to is not safe to assume for GPU work.
PYTHON=python3
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON=python3.11
fi

if [ ! -d .venv ]; then
  echo "[setup] Creating virtual environment with $PYTHON..."
  "$PYTHON" -m venv .venv
fi

echo "[setup] Installing dependencies..."
.venv/bin/pip install -q --upgrade pip

# GPU-compatible torch build, installed BEFORE the rest, then the main
# install run WITH the same --extra-index-url -- both steps are needed:
# installing torch alone first is not enough, because a subsequent
# `pip install -e .` without --extra-index-url re-resolves torch from the
# default PyPI index and silently replaces it with a newer non-CUDA-
# matching build (this happened live). If this fails (no
# matching GPU/driver on your machine), delete this block -- pip falls
# back to a CPU-only torch build automatically, just slower for embedding
# generation.
echo "[setup] Installing GPU-compatible torch (cu121)..."
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121

echo "[setup] Installing project dependencies..."
.venv/bin/pip install -q -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121

echo "[setup] Checking GPU availability..."
.venv/bin/python3 -c "
import torch
available = torch.cuda.is_available()
print(f'  torch {torch.__version__}, CUDA available: {available}')
if available:
    print(f'  device: {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: no GPU detected -- embedding generation will run on CPU (slower).')
"

echo "[setup] Downloading WordNet + Open Multilingual Wordnet (English/French/Arabic lemma data)..."
.venv/bin/python3 -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-2.0', quiet=True)
"

echo "[setup] Done."
