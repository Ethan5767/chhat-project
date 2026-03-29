#!/usr/bin/env bash
# Run on the RunPod (Linux). RF-DETR only -- no classifier, no DO Spaces.
# Expects repo bootstrapped: runpod/bootstrap_training_pod.sh
#
# Dataset layout after optional extract:
#   datasets/cigarette_packs/train/_annotations.coco.json
#   datasets/cigarette_packs/train/*.jpg
# train.py will auto-split valid/ if missing.
set -euo pipefail

ROOT="${ROOT:-/workspace/chhat-project}"
cd "$ROOT"
source .venv/bin/activate

DATASET_TAR=""
if [[ $# -gt 0 && "$1" == *.tar.gz ]]; then
  if [[ -f "$1" ]]; then
    DATASET_TAR="$1"
    shift
  fi
fi

if [[ -n "$DATASET_TAR" ]]; then
  if [[ ! -f "$DATASET_TAR" ]]; then
    echo "Dataset archive not found: $DATASET_TAR"
    exit 1
  fi
  mkdir -p datasets/cigarette_packs
  # RunPod workspace only: clear old COCO splits before extracting new archive
  rm -rf datasets/cigarette_packs/train datasets/cigarette_packs/valid datasets/cigarette_packs/test
  tar -xzf "$DATASET_TAR" -C datasets/cigarette_packs
  echo "Extracted dataset under $ROOT/datasets/cigarette_packs/"
fi

exec python train.py "$@"
