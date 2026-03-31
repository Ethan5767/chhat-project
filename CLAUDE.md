# CLAUDE.md

This file provides guidance to Claude Code when working in the rf-detr-cigarette project.

## Project Overview

Cigarette brand detection system for CHHAT. Uses RF-DETR-Medium for object detection, DINOv2 + linear classifier for brand classification, and EasyOCR for text verification. Backend is FastAPI, frontend is Streamlit.

## Architecture

- **Detection**: RF-DETR-Medium fine-tuned on cigarette pack dataset (checkpoint at `runs/checkpoint_best_ema.pth`)
- **Classification**: DINOv2-base (frozen or fine-tuned) backbone + per-packaging-type linear classifiers
- **OCR**: EasyOCR with brand-consensus fusion (OCR is primary signal, DINOv2 confirms product)
- **Pipeline flow**: Image -> RF-DETR detect packs -> crop -> DINOv2 embed -> classifier predict -> OCR verify -> output
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

## Conventions

- No emojis in code, commits, or output
- Use "product" not "SKU" (Brand -> Product hierarchy)
- OCR is primary brand signal, DINOv2 confirms product (not the other way around)
- `brand_registry.py` is the single source of truth for all brand/product mappings
- Use EasyOCR, not Tesseract

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
