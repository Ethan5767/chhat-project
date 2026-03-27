#!/usr/bin/env bash
set -euo pipefail

cd /workspace/chhat-project

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-learn easyocr openpyxl boto3 python-dotenv

mkdir -p backend/references backend/classifier_model backend/uploads runs

echo "Bootstrap complete."
