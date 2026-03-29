"""Persistent data locations.

Set CHHAT_DATA_ROOT on the server (e.g. /var/lib/chhat-project) so references,
classifiers, uploads, batch history, and RF-DETR runs stay outside the deploy
tree (/opt/chhat-project). Code deploys then only replace application code.

Unset CHHAT_DATA_ROOT: legacy layout (data under backend/)."""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def _external_data_root() -> str:
    return os.environ.get("CHHAT_DATA_ROOT", "").strip()


def using_external_data_root() -> bool:
    return bool(_external_data_root())


def _resolve_data_root() -> Path:
    raw = _external_data_root()
    if raw:
        p = Path(raw).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return BACKEND_DIR


DATA_ROOT = _resolve_data_root()


def _runs_dir() -> Path:
    if using_external_data_root():
        return DATA_ROOT / "runs"
    return PROJECT_ROOT / "runs"


RFDETR_RUNS_DIR = _runs_dir()

REFERENCES_DIR = DATA_ROOT / "references"
CLASSIFIER_BASE_DIR = DATA_ROOT / "classifier_model"
UPLOADS_DIR = DATA_ROOT / "uploads"
RESULTS_DIR = UPLOADS_DIR / "results"

BATCH_HISTORY_PATH = DATA_ROOT / "batch_history.json"
TRAINING_HISTORY_PATH = DATA_ROOT / "training_history.json"
MODEL_REGISTRY_PATH = DATA_ROOT / "model_registry.json"
VERSION_STATE_PATH = DATA_ROOT / "training_version_state.json"

for _d in (REFERENCES_DIR, CLASSIFIER_BASE_DIR, UPLOADS_DIR, RFDETR_RUNS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
