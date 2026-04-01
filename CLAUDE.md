# CLAUDE.md

This file provides guidance to Claude Code when working in the rf-detr-cigarette project.

## Project Overview

Cigarette brand detection system for CHHAT. Uses RF-DETR-Medium for object detection and DINOv2 + linear classifier for brand classification. Backend is FastAPI, frontend is Streamlit.

## Architecture

- **Detection**: RF-DETR-Medium fine-tuned on cigarette pack dataset (checkpoint at `runs/checkpoint_best_ema.pth`)
- **Classification**: DINOv2-base (frozen or fine-tuned) backbone + per-packaging-type linear classifiers
- **Pipeline flow**: Image -> RF-DETR detect packs -> crop -> DINOv2 embed -> classifier predict -> output
- **Packaging types**: pack, box (separate classifiers per type)
- **Brand registry**: 29 brands, 68 products defined in `backend/brand_registry.py` (single source of truth)

## Build & Run

```bash
# Backend
cd backend && uvicorn main:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend && streamlit run app.py

# Training (RF-DETR)
python train.py

# Training (Brand classifier - local)
python brand_classifier.py --packaging-type pack --epochs 100

# Training (Brand classifier - RunPod GPU)
# Triggered via POST /train-classifier?use_runpod=true
```

## Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI server, all endpoints, RunPod GPU job orchestration |
| `backend/pipeline.py` | Detection + classification pipeline |
| `backend/brand_registry.py` | Brand/product definitions (source of truth) |
| `backend/output_format.py` | CSV output column mappings (Q12A/Q12B) |
| `brand_classifier.py` | DINOv2 classifier training script |
| `frontend/app.py` | Streamlit UI |
| `runpod/bootstrap_training_pod.sh` | Pod setup script (clones repo, installs deps) |

## RunPod SSH

RunPod's ssh.runpod.io proxy does NOT support:
- `exec_command` (Paramiko) -- only interactive shell works
- SFTP/SCP file transfers -- use DO Spaces as intermediary
- `ssh -W` TCP forwarding
- `ssh -T` (no PTY) -- proxy requires PTY

What works:
- `ssh -t user@ssh.runpod.io` from terminal (interactive)
- Paramiko `invoke_shell()` for command execution
- DO Spaces pre-signed URLs for file transfer (server -> Spaces -> pod wget; pod curl -> Spaces -> server download)
- Username format: `{podHostId}@ssh.runpod.io` (get podHostId from GraphQL `machine { podHostId }`)
- SSH key: `~/.ssh/runpod_ed25519` (must be registered in RunPod account settings)

## RunPod GPU Training

- Bootstrap timeout: 1200s (20 min) -- covers git clone + pip install
- Git clone uses `--depth 1` to skip history
- Bootstrap installs PyTorch CUDA 12.4 (compatible with RunPod driver 12.8)
- File transfer via DO Spaces (SFTP through proxy fails for large files)
- Classifier GPU fallback: tries SECURE then COMMUNITY cloud, 8+ GPU types, 5 retry rounds

## RunPod Pod Reuse

- **NEVER create or destroy RunPod pods.** Always reuse the existing pod by stopping and resuming it. If a pod is stopped, resume it. If it is running, use it as-is. Only create a new pod if no pod exists at all.
- **NEVER terminate a pod** unless the user explicitly asks to terminate a specific pod by ID
- Pods are stopped (not terminated) after jobs to preserve /workspace volume
- Pod registry at `_DATA_ROOT / "runpod_pods.json"` tracks reusable pods per job type (batch, dinov2, classifier, rfdetr)
- On next job, try `podResume` first (skip bootstrap + weight upload), fall back to create-new if GPU unavailable
- `podHostId` may change on resume -- re-queried after `podResume`
- Container disk is wiped on stop, but /workspace persists (venv, model weights, repo all survive)
- Stopped pod volume cost: ~$0.20/GB/month ($10/month for 50GB)
- Management endpoints: `GET /runpod/pods`, `POST /runpod/pods/{id}/terminate`, `POST /runpod/pods/cleanup`

## Conventions

- No emojis in code, commits, or output
- Use "product" not "SKU" (Brand -> Product hierarchy)
- `brand_registry.py` is the single source of truth for all brand/product mappings

## Sensitive Files

- `.env` -- DO Spaces credentials, RunPod API key
- `~/.ssh/runpod_ed25519` -- RunPod SSH key (server only)

## Production Server

- IP: 134.209.96.41 (16 GB RAM, 4 GB swap)
- Data: /opt/chhat-data (references, classifier models, training history)
- Code: /opt/chhat-project
- Services: `systemctl restart chhat-backend` / `systemctl restart chhat-frontend`
- Logs: `journalctl -u chhat-backend -f`
- SSH key: `~/.ssh/id_ed25519` (for server access), `~/.ssh/runpod_ed25519` (for RunPod pods)
- Nginx on port 80 with basic auth

## Known Issues and Fixes (RunPod Training)

These issues were discovered and fixed. Document here to avoid repeating them:

### Paramiko SSH Proxy
- RunPod proxy (`ssh.runpod.io`) requires `invoke_shell()` with PTY, not `exec_command`
- **Terminal width must be 4096+**: Presigned URLs are 400+ chars; width=200 truncated URLs with line breaks causing 0-byte downloads
- **Use `term="dumb"`**: Prevents ANSI escape codes that corrupt exit code and output parsing
- **Use `stty -echo`**: Prevents command echo-back that doubles URLs in output
- **Strip ANSI codes**: Use `_strip_ansi()` before parsing exit codes or file sizes from shell output
- **Use single quotes for URLs**: Presigned URLs contain `$` chars; double quotes cause bash variable expansion (e.g. `$def` in signature expands to empty string)
- **JSON progress polling**: Shell output includes markers/prompts; use `re.search(r'\{[^{}]*\}', raw)` to extract JSON, not `txt.startswith("{")`

### DO Spaces File Transfer
- Upload server -> DO Spaces works reliably
- Pod download (Spaces -> pod): Use `curl -sS -L --retry 3` instead of `wget` (more reliable)
- **Retry + size verification**: wget/curl can silently return 0 bytes; verify with `stat -c %s` and retry up to 3 times
- **URL expiry**: Set `ExpiresIn=7200` (2 hours) for GET presigned URLs
- **S3 cleanup**: `finally` block deletes S3 object; ensure download confirmed before cleanup

### Bootstrap (`runpod/bootstrap_training_pod.sh`)
- Use `--system-site-packages` for venv to inherit template's pre-installed PyTorch (~6 min vs ~15 min bootstrap)
- Only install torch if CUDA check fails (template already has it)
- Assert CUDA available at end of bootstrap; fail hard if not
- `.gitattributes` forces LF line endings for .sh files (Windows CRLF breaks bash)

### DINOv2 Finetuning (`finetune_dinov2.py`)
- **DataLoader num_workers=0 causes GPU starvation**: CPU loads images one-by-one, GPU sits idle at 0% util. Use num_workers=4 + pin_memory + persistent_workers on GPU
- **L2 normalization**: Added in forward() to match pipeline.py and brand_classifier.py inference
- **Corrupt image handling**: __getitem__ must catch Image.open errors or 1 bad image kills entire training
- **Target tensors**: Use `targets.long().to(device)` not `torch.tensor(targets)` (avoids copy warning)

### Brand Classifier (`brand_classifier.py`)
- **Must load finetuned DINOv2 backbone**: Otherwise embeddings mismatch between training and inference
- Server uploads `dinov2_finetuned_full.pth` to pod via `_scp_to()` before training starts
- **CUDA required on RunPod**: Hard-exit if `RUNPOD_POD_ID` env set but CUDA unavailable
- **Allow CPU locally**: For local dev/testing only (slow but functional)

### Path Consistency (CHHAT_DATA_ROOT)
- Production uses `CHHAT_DATA_ROOT=/opt/chhat-data` (set via systemd drop-in)
- ALL scripts must respect this: `pipeline.py`, `brand_classifier.py`, `finetune_dinov2.py`, `brand_registry.py`
- RunPod pods don't set CHHAT_DATA_ROOT; they use the default `PROJECT_ROOT / "backend"` which is correct for the pod's filesystem
- Never hardcode paths to `backend/references/` or `backend/classifier_model/`; always use the `_DATA_ROOT` variable

### RunPod GPU Batch Processing (`run_pipeline_gpu_job`)
- **Must upload model weights**: Pod needs classifier (`best_classifier.pth`, `class_mapping.json`), DINOv2 finetuned (`dinov2_finetuned_full.pth`), and RF-DETR checkpoint (`checkpoint_best_ema.pth`)
- **Upload to file paths not directories**: `_scp_to()` via DO Spaces requires full file path, not directory (curl -o /path/dir/ fails)
- **Segfault from optimize_for_inference**: `torch.compile` (used by `optimize_for_inference()`) segfaults on RunPod A100s. Skip it when on RunPod.
- **RunPod detection**: `RUNPOD_POD_ID` env var is NOT set on generic RunPod templates. Detect via `os.path.exists("/workspace")` instead.
- **Python -c quoting**: Use escaped double quotes pattern `python -c \"...\"` with single quotes for string literals inside. Mixed quoting causes shell parsing failures.
- **Timeout for large batches**: 12k rows takes ~8-10 hours. Set timeout to 48h (`48 * 3600`), not 2h.
- **CUDA_LAUNCH_BLOCKING=1**: Add to pipeline command for proper error traces instead of bare segfaults.
- **tar --no-same-owner**: Windows-created tar archives have uid 197609; Linux tar fails to set ownership. Use `--no-same-owner` on extraction.
- **tar --owner=0 --group=0**: Strip Windows UIDs when creating archives on the server.

### RunPod RF-DETR Training (`run_rfdetr_training_runpod_job`)
- Must match DINOv2 flow structure exactly: `volumeMountPath: "/workspace"`, `ports: "22/tcp"` in pod creation
- SSH probe: 20 attempts with `SSH_OK` marker, timeout=60, 15s between
- Bootstrap check: use `test -f train.py && echo OK || echo MISSING` (not checking .venv)
- Pod creation loop order: cloud -> gpu (not gpu -> cloud)
- Check `pod is not None` (not truthy string check on `pod_id`)
- Install `rfdetr[train,loggers]` as separate step after bootstrap (not in requirements.txt)
- Progress polling every 12s with regex JSON extraction
- **GPU preference**: A100 > RTX 4090/4080 > L40S (H100 removed -- overkill and expensive for RF-DETR training)
- **Pod clones from GitHub, not server**: Code changes (train.py, pipeline.py) must be pushed to GitHub before triggering RunPod training, otherwise the pod runs stale code
- **NEVER send kill signals to training PID**: `kill -USR1` or similar to inspect a running train.py process will terminate it (exit code 138). Use `nvidia-smi`, `ps aux`, `/proc/PID/status` for diagnostics instead.

### COCO Dataset Category Issues (Roboflow exports)
- Roboflow exports include an empty parent `objects` category (id=0) with zero annotations
- RF-DETR / pycocotools treats category_id=0 as valid, causing mAP=0.000 because no predictions match the phantom category
- **Always remove the `objects` category** before training: keep only `cigarette_box` and `cigarette_pack`, re-index to 0-based contiguous IDs
- Roboflow bbox values are strings (`'101.27'` not `101.27`); must cast to float before training
- When merging multiple Roboflow exports (e.g. coco v3, v4, v6), deduplicate by filename -- different versions may annotate the same images differently; prefer the highest version number

### RF-DETR mAP Progress Logging
- `train.py` now uses a custom PTL `ProgressWriterCallback` that writes mAP metrics (mAP_50_95, mAP_50, mAP_75, mAR, F1, ema variants) to the progress JSON after each validation epoch
- `_metrics_from_progress()` in main.py extracts these mAP fields for the training status API
- The old train.py only wrote epoch/total_epochs/status (no mAP), so remote monitoring showed no accuracy
- The callback is injected by importing rfdetr training internals (`RFDETRModelModule`, `RFDETRDataModule`, `build_trainer`) and appending to `trainer.callbacks` before `trainer.fit()`

### RunPod SSH Key
- SSH key: `~/.ssh/runpod_ed25519` on the server
- Must be registered in RunPod account settings (Settings > SSH Public Keys)
- When server IP changes, the key is already on the server but may need re-registering in RunPod dashboard
- Pod SSH auth uses paramiko through `ssh.runpod.io` proxy with `podHostId` as username

### URL Column Detection (`get_url_columns`)
- Client CSVs have sparse image columns (some only 2-5% of rows have URLs)
- Old 30% threshold missed most image columns; lowered to 1 URL in first 50 rows
- Also checks first row for descriptor text like "Photo Link" or "Q32 Photo Link"
- Must preserve ALL Q30/Q33 columns in output even if not detected as URL columns
