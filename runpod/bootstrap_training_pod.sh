#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/chhat-project}"
cd "$ROOT"

python3 -m venv .venv || { echo "ERROR: venv creation failed"; exit 1; }
source .venv/bin/activate

python -m pip install --upgrade pip
# Install PyTorch with CUDA 12.4 first (compatible with RunPod drivers 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Install remaining deps -- torch/torchvision already satisfied, pip won't downgrade
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
pip install scikit-learn easyocr openpyxl boto3 python-dotenv

mkdir -p backend/references/pack backend/references/box \
         backend/classifier_model/pack backend/classifier_model/box \
         backend/uploads runs

echo "Bootstrap complete."
