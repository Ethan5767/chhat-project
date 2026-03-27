# RunPod A100 80GB Training Setup

This folder prepares a repeatable training pod workflow for `rf-detr-cigarette`.

## 1) Create pod (recommended)

- GPU: `A100 80GB`
- Mode: `On-demand` (stable for long runs)
- Disk: `>= 200GB`
- Docker image: `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

## 2) On the pod, bootstrap once

```bash
cd /workspace
git clone https://github.com/Ethan5767/chhat-project.git
cd chhat-project
bash runpod/bootstrap_training_pod.sh
```

## 3) Configure environment

Copy `.env` (with DO Spaces credentials) to `/workspace/chhat-project/.env`.

## 4) Run one training cycle

```bash
cd /workspace/chhat-project
bash runpod/run_training_cycle.sh
```

The script will:
- ensure references are present (downloads from DO Spaces if missing)
- train classifier with OCR-fallback-ready backend
- train RF-DETR with the current repo code

## 5) Sync artifacts back

From your laptop:

```powershell
scp -r root@<POD_IP>:/workspace/chhat-project/runs "C:\Users\kimto\OneDrive\Desktop\RE-AI\rf-detr-cigarette\"
scp -r root@<POD_IP>:/workspace/chhat-project/backend/classifier_model "C:\Users\kimto\OneDrive\Desktop\RE-AI\rf-detr-cigarette\backend\"
```

## Notes

- The backend now runs OCR as fallback only for uncertain crops:
  - top-1 confidence `< 0.72`, or
  - top-1 vs top-2 margin `< 0.08`.
- Keep periodic checkpoints in `runs/` for resume safety.
