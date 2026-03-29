# RunPod A100 80GB Training Setup

This folder prepares a repeatable training pod workflow for `rf-detr-cigarette`.

**Where compute runs:** All heavy training (RF-DETR, classifier) runs **on the RunPod GPU**. Your laptop only uploads files and starts commands over **SSH** (or you paste commands in the RunPod web terminal). Nothing in these steps trains on your local machine unless you explicitly run `python train.py` locally.

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

### 4b) RF-DETR only (Roboflow COCO folder on your laptop)

Use this when you only need detector fine-tuning and already have a folder like `Cigarette pack brand.coco (...)/train/_annotations.coco.json` plus images.

**On the pod** (after bootstrap), you can unpack a tarball yourself:

```bash
cd /workspace/chhat-project
# put rfdetr_coco_train.tar.gz in this directory (contains train/ at archive root)
bash runpod/train_rfdetr_only.sh rfdetr_coco_train.tar.gz
# optional train.py args after the tarball name, e.g.:
# bash runpod/train_rfdetr_only.sh rfdetr_coco_train.tar.gz --epochs 50 --batch-size 4
```

**From Windows** (uploads `train/` from your Roboflow export, then SSHs into the pod and starts training there):

```powershell
cd C:\Users\kimto\OneDrive\Desktop\RE-AI\rf-detr-cigarette
.\runpod\upload_coco_train_and_train.ps1 `
  -SshTarget "root@YOUR_POD_IP" `
  -SshPort YOUR_SSH_PORT `
  -DatasetFolder "C:\Users\kimto\OneDrive\Desktop\RE-AI\rf-detr-cigarette\Cigarette pack brand.coco (5)"
```

Replace `YOUR_POD_IP` and `YOUR_SSH_PORT` with the values from the RunPod connect panel.

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
