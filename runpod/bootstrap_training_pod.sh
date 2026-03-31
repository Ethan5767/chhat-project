#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/chhat-project}"
cd "$ROOT" || { echo "ERROR: Cannot cd to $ROOT"; exit 1; }

python3 -m venv --system-site-packages .venv || { echo "ERROR: venv creation failed"; exit 1; }
source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }

python -m pip install --upgrade pip
# Template already has PyTorch+CUDA; only install if missing
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  || pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Install remaining deps
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt || true
# Extras not in requirements.txt
pip install scikit-learn easyocr openpyxl boto3 python-dotenv
# Verify critical packages -- fail hard if torch/CUDA missing
python -c "import sklearn; import torch; assert torch.cuda.is_available(), 'CUDA not available after install'; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} sklearn OK')"

mkdir -p backend/references/pack backend/references/box \
         backend/classifier_model/pack backend/classifier_model/box \
         backend/uploads runs

echo "Bootstrap complete."
