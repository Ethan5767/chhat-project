# CLAUDE.md

This file provides guidance to Claude Code when working in the rf-detr-cigarette project.

## Project Overview

Cigarette brand detection system for CHHAT. Uses RF-DETR-Medium for object detection and DINOv2 + linear classifier for brand classification. Backend is FastAPI, frontend is Streamlit.

## Architecture

- **Detection**: Co-DETR Swin-L (default) or RF-DETR-Medium for object detection
- **Classification**: DINOv2-base (fine-tuned) backbone + per-packaging-type linear classifiers
- **Pipeline flow**: Image -> Co-DETR detect packs -> crop -> DINOv2 embed -> classifier predict -> output
- **Packaging types**: pack, box (separate classifiers per type)
- **Brand registry**: 29 brands, 68+ products defined in `backend/brand_registry.py` (single source of truth)
- **Current thresholds**: `CODETR_CONF_THRESHOLD=0.10`, `MIN_OUTPUT_CONFIDENCE=0.80`
- **Q33 excluded**: Store exterior photos (Q33 columns) are excluded from detection to reduce false positives
- **Co-DETR checkpoint**: `co_detr_weights/finetuned_epoch12.pth` (3.6 GB, trained on H200)
- **DINOv2 backbone**: `classifier_model/dinov2_finetuned_full.pth` (334 MB, fine-tuned on pack refs)
- **Classifier**: `classifier_model/pack/best_classifier.pth` (102 classes, ~94% val_acc)

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

### RunPod GPU Batch: Directory & Dependency Issues
- **`backend/uploads/` must be created before CSV upload**: The `mkdir -p` for uploads dir must happen BEFORE `_scp_to` for the CSV, not after (in the image cache step)
- **Weight upload must check, not assume**: Resumed pods may lack weights if a prior job errored mid-upload. Always `test -f` the classifier weight on the pod before skipping upload
- **Co-DETR needs mmdet/mmengine/mmcv**: These are NOT pre-installed on RunPod templates. The batch job must verify `import mmdet` and install if missing before running the pipeline
- **Image cache upload must filter by CSV**: Only upload cached images whose IDs appear in the CSV URLs, not the entire 26k+ cache. Parse `?ID=(\d+)` from CSV URL columns to build needed set

### mmcv Installation on RunPod (CRITICAL)
- **`mim install mmcv` and `pip install mmcv` fail** on RunPod pods due to `pkg_resources` missing in pip's build isolation environment (setuptools 82.0+ deprecated it)
- **Fix**: `pip install mmcv==2.1.0 --no-build-isolation` — uses venv's setuptools instead of isolated build env
- **Alternative**: `PIP_NO_BUILD_ISOLATION=1 mim install mmcv==2.1.0`
- mmcv builds from source for torch 2.4.1+cu124 (no prebuilt wheels available) — takes 10-30 min compiling CUDA ops
- The Co-DETR training pod (H200) used the same env (Python 3.11.10, torch 2.4.1+cu124) and had mmcv working, confirming compatibility
- **Bootstrap script** (`runpod/bootstrap_training_pod.sh`) must use `--no-build-isolation` for mmcv install
- The batch job clone must use `co-detr-migration` branch (not `main`) since Co-DETR configs and mmdet bootstrap are only on that branch (`RUNPOD_REPO_BRANCH = "co-detr-migration"`)

### Co-DETR Training Results
- Trained on H200 GPU: 36 epochs on 1266 images (1043 main + 223 Chhat), single class "cigarette"
- Epoch 36: mAP 0.770, mAP@50 0.980, mAP@75 0.835
- Config: `co_detr_weights/codetr_swinl_o365_combined.py` (Swin-L backbone, Objects365+COCO pretrained)
- Inference config: `backend/co_detr_inference_config.py` (must be in git repo for pods)
- **Batch job must upload** `co_detr_inference_config.py` alongside `pipeline.py` to the pod — it's not in the git `main` branch

### RunPod GPU Batch: Co-DETR Full Setup Checklist
All issues discovered 2026-04-05. Every item below caused a job failure when missed:

1. **Clone from `co-detr-migration` branch** (not `main`): `backend/co_detr/` module, `co_detr_inference_config.py`, and mmdet bootstrap only exist on this branch. Set `RUNPOD_REPO_BRANCH = "co-detr-migration"` and use `-b {RUNPOD_REPO_BRANCH}` in git clone.

2. **Upload `backend/co_detr/` module to pod**: Even after cloning the right branch, resumed pods from older clones (main branch) won't have it. The batch job must `test -d backend/co_detr` and upload as tarball if missing. This module contains the custom Co-DETR model definitions (codetr.py, transformer.py, etc.) required by `custom_imports` in the inference config.

3. **Upload `co_detr_inference_config.py` alongside `pipeline.py`**: The batch job uploads server's pipeline.py to override git version, but must also upload the inference config. Both go to `backend/` on the pod.

4. **`mkdir -p backend/uploads/` before CSV upload**: The uploads dir doesn't exist after a fresh clone. Must create it before `_scp_to` for the CSV file, not after (the image cache step also needs it but runs later).

5. **Check weights exist, don't assume resumed = has weights**: Use `test -f` on pod for classifier + Co-DETR checkpoint before skipping upload. A prior job may have errored mid-upload leaving the pod without weights.

6. **Install mmcv with `--no-build-isolation`**: `pip install mmcv==2.1.0 --no-build-isolation` (takes 10-30 min compiling CUDA ops). Standard pip install fails due to `pkg_resources` missing in build isolation with setuptools 82.0+. The batch job must check `import mmdet` and install if missing, with a 2400s timeout.

7. **Filter image cache by CSV**: Only tar+upload cached images whose IDs appear in the CSV URLs. Without filtering, it tries to upload all 26k+ images (10+ GB tarball).

8. **`CHHAT_DATA_ROOT` not set in direct script runs**: The systemd drop-in sets `CHHAT_DATA_ROOT=/opt/chhat-data` for the backend service, but running scripts directly via `nohup python ...` doesn't inherit it. Always pass `CHHAT_DATA_ROOT=/opt/chhat-data` when running scripts outside systemd.

9. **Server OOM with Co-DETR**: Co-DETR Swin-L + DINOv2 together use ~8.7 GB RAM. The 16 GB server swap-thrashes and OOM-kills annotation scripts. Run Co-DETR batch processing on RunPod GPU, not the server.

### RunPod GPU Batch: Additional Issues Found 2026-04-05

10. **`pip install --upgrade pip` causes I/O errors**: RunPod container disk gets `OSError: [Errno 5] Input/output error` when pip upgrades itself from 24.0 to 26.0.1. **Do NOT upgrade pip** in bootstrap -- template pip works fine.

11. **`requirements.txt || true` swallows critical failures**: The bootstrap used `pip install -r requirements.txt || true` which silently fails on mmcv, but also skips `transformers` and other deps listed after it. **Fix**: install non-mmcv deps explicitly (transformers, faiss-cpu, supervision, etc.) instead of via requirements.txt. Install mmcv/mmdet/mmengine separately.

12. **`transformers` not installed = pipeline import fails**: `from transformers import AutoImageProcessor, AutoModel` is the first import in pipeline.py. If transformers isn't installed, the pipeline crashes immediately with `ModuleNotFoundError`. Bootstrap must install it explicitly.

13. **Resumed pod from failed bootstrap has partial state**: If bootstrap crashes mid-way (e.g. pip I/O error), the pod has a cloned repo but missing deps. The resumed path skips bootstrap (only does git pull). The mmdet check step catches mmcv/mmdet, but transformers/faiss/etc. won't be checked. The bootstrap must be idempotent and complete.

14. **Git `--depth 1` clone can't switch branches**: `git fetch origin co-detr-migration && git checkout co-detr-migration` fails on a depth-1 clone of `main` because the branch ref doesn't exist locally. For resumed pods from old clones, upload files directly (co_detr module, config) rather than relying on branch switch.

15. **Batch `inference_detector` with image list causes 5h+ stalls**: mmdet's `inference_detector([img1, img2, ...])` with 24+ images causes extreme slowdown on Co-DETR Swin-L. Sequential per-image detection completes 100 rows in 12 min; batch detection stalled for 5+ hours. **Always use sequential detection** (call `detect_objects()` per image, not `detect_objects_batch()`).

16. **Q33 store exterior photos add false positives**: Q33 columns contain store exterior/wide shots. At low detection thresholds (0.05-0.10), Co-DETR detects random objects as cigarette packs in these images. **Fix**: exclude Q33 columns from detection (`url_columns = [c for c in all_url_columns if "q33" not in str(c).lower()]`). Q33 values are still preserved in the output CSV.

17. **Annotation script on pod needs DO Spaces env vars**: The annotation script (`scripts/visualize_detections.py`) uploads annotated images to DO Spaces. Pod doesn't have `.env`. Pass env vars inline: `DO_SPACES_KEY=... DO_SPACES_SECRET=... DO_SPACES_ENDPOINT=... DO_SPACES_BUCKET=... python scripts/visualize_detections.py`. Do NOT pass `CHHAT_DATA_ROOT` -- pod uses default `backend/` path.

18. **mmdet version check must use regex, not string match**: `python -c 'import mmdet; print(mmdet.__version__)'` prints `3.3.0` not the word "mmdet". Check with `re.search(r'\d+\.\d+', stdout)` not `"mmdet" in stdout`.

19. **Pod must stay running after batch for follow-up tasks**: Skip `_runpod_stop()` after successful batch completion. The annotation script needs the pod running with models loaded. Stop manually via `/runpod/pods` endpoint when done.

20. **Kill stale processes before reusing pod**: A pod left running from a failed/stuck job may have old Python processes holding GPU memory. The batch job's zombie-kill step (nvidia-smi query + kill) handles GPU processes, but the old Paramiko session's process may not show in nvidia-smi. Check `ps aux | grep python` and kill stale pipeline processes before starting a new run.

### URL Column Detection (`get_url_columns`)
- Client CSVs have sparse image columns (some only 2-5% of rows have URLs)
- Old 30% threshold missed most image columns; lowered to 1 URL in first 50 rows
- Also checks first row for descriptor text like "Photo Link" or "Q32 Photo Link"
- Must preserve ALL Q30/Q33 columns in output even if not detected as URL columns

## How to Run a Batch Processing Job

### Step 1: Prepare the CSV
- Input format: `Respondent.Serial,Q6,Q30_1,Q30_2,Q30_3,Q33_1,Q33_2,Q33_3`
- If Excel, skip header/descriptor rows: `df = pd.read_excel(file, header=0); df = df.iloc[1:]`
- For parallel processing, split into N chunks or use `/run-pipeline-parallel`

### Step 2: Single Pod Processing
```bash
# Upload CSV to server
scp input.csv root@134.209.96.41:/tmp/input.csv

# Restart backend (picks up latest code)
ssh root@134.209.96.41 'systemctl restart chhat-backend'

# Trigger batch (wait ~10s for backend to load)
ssh root@134.209.96.41 'sleep 10 && curl -s -X POST "http://127.0.0.1:8000/run-pipeline" -F "csv_file=@/tmp/input.csv" -F "use_gpu=true"'
```

### Step 3: Parallel Processing (Multiple Pods)
```bash
# Upload full CSV
scp full_batch.csv root@134.209.96.41:/tmp/full_batch.csv

# Trigger with N pods (splits CSV automatically)
ssh root@134.209.96.41 'sleep 10 && curl -s -X POST "http://127.0.0.1:8000/run-pipeline-parallel" -F "csv_file=@/tmp/full_batch.csv" -F "num_pods=5"'
```
- Each chunk gets its own pod (registry keys: `batch_0` through `batch_4`)
- Default GPU: RTX 4090 (~$0.59/hr), falls back to RTX 3090/A5000/A6000
- Pods stop automatically after each chunk completes
- Results merged into single CSV when all chunks done

### Step 4: Monitor Progress
- **Server logs**: `ssh root@134.209.96.41 'journalctl -u chhat-backend -f'`
- **Row counts on pods**: SSH to each pod and `wc -l /workspace/chhat-project/backend/uploads/*results*`
- **Pod status**: `curl RunPod GraphQL API` to check GPU utilization and uptime
- **Partial results**: `GET /download-partial/{job_id}` fetches in-progress CSV from pod
- **Speed**: RTX 4090 processes ~160 rows/min, RTX 3090 ~60 rows/min

### Step 5: Download Results
```bash
# Results auto-downloaded to server at job completion
# Find result file:
ssh root@134.209.96.41 'ls -la /opt/chhat-data/uploads/results/*{job_id}*'

# Download to local:
scp root@134.209.96.41:/opt/chhat-data/uploads/results/result_file.csv ./
```

### Expected Timelines
- **Pod bootstrap** (fresh pod): ~20-30 min (git clone + mmcv compile)
- **Pod resume** (existing pod): ~2 min (SSH + upload files)
- **mmcv compile on pod**: ~30-40 min (builds CUDA ops from source)
- **Pipeline processing**: ~160 rows/min on RTX 4090, ~60 rows/min on RTX 3090
- **5k rows on 1 pod**: ~30 min pipeline
- **24k rows on 5 pods**: ~2.5 hours total (including bootstrap)

## How to Run Annotation (Bounding Box Images)

### After batch processing completes:
1. Resume the pod (or use one that's still running)
2. Upload annotation script + patch S3 prefix
3. Run annotation with DO Spaces env vars inline
4. Download annotated CSV when done

```bash
# On server, run the annotation helper script:
ssh root@134.209.96.41 'cd /opt/chhat-project && source .venv/bin/activate && python scripts/run_annotation_on_pod.py'
```

- Annotation re-runs detection + classification on each image to draw bounding boxes
- Uploads annotated images to DO Spaces: `annotations/{prefix}/{serial}_{col}.jpg`
- Uses `--s3_prefix` arg to avoid overwriting between batches
- Speed: ~8 rows/min (slower than batch due to image drawing + S3 upload)
- 1000 rows takes ~2 hours

## How to Retrain Models

### DINOv2 Fine-tuning (run first - classifier depends on it)
```bash
ssh root@134.209.96.41 'curl -s -X POST "http://127.0.0.1:8000/finetune-dinov2?use_runpod=true"'
```
- Creates RunPod pod, uploads references, trains 30 epochs
- Downloads `dinov2_finetuned_full.pth` to server
- ~15-20 min total on RTX 4090

### Brand Classifier Training (run after DINOv2)
```bash
ssh root@134.209.96.41 'curl -s -X POST "http://127.0.0.1:8000/train-classifier?use_runpod=true"'
```
- Creates RunPod pod, uploads references + DINOv2 weights, trains 100 epochs (early stops ~30-50)
- Downloads `best_classifier.pth` + `class_mapping.json` to server
- **Known issue**: `packaging_type=all` saves to `pack/` dir (not `all/`). Fixed in main.py download path.
- ~15-20 min total on RTX 4090

### After retraining:
- Models auto-reload on server (`reload_dino + reload_classifiers`)
- Next batch processing uses updated models automatically

## Box vs Pack Detection Status

- **Co-DETR**: Single-class "cigarette" detector. Returns all detections as `class_id=0` (pack). Cannot distinguish pack from box.
- **RF-DETR**: 2-class detector (`cigarette_box=0`, `cigarette_pack=1`). CAN distinguish but lower mAP than Co-DETR.
- **Box classifier**: Does NOT exist. `/references/box/` is empty. When RF-DETR detects a box, falls back to pack classifier.
- **To fix boxes**: Need box reference images + train box classifier + either retrain Co-DETR with 2 classes or use RF-DETR for pack/box routing.

## Parallel Processing Race Conditions

- **Image cache tarball**: Each parallel thread must use unique temp path (`/tmp/image_cache_{job_id}.tar.gz`) to avoid overwriting
- **Pod registry**: Each parallel chunk uses unique key (`batch_0`, `batch_1`, etc.)
- **DO Spaces transfers**: Multiple simultaneous uploads work fine (each uses unique S3 key)
