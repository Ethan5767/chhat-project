import sys, os, traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

_BACKEND = Path(__file__).resolve().parent
_BRAND_ROOT = _BACKEND.parent
_DATA_YAML = _BRAND_ROOT / "datasets" / "yolo_cigarette_packs" / "data.yaml"
_RUNS_YOLO = _BRAND_ROOT / "runs" / "yolo"

print("[1/5] Testing torch + CUDA ...", flush=True)
import torch
print(f"       torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"       GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
else:
    print("       WARNING: CUDA not available, will fall back to CPU", flush=True)

print("[2/5] Importing ultralytics ...", flush=True)
from ultralytics import YOLO

print("[3/5] Loading yolo11n.pt ...", flush=True)
model = YOLO("yolo11n.pt")

data_yaml = str(_DATA_YAML.resolve())
print(f"[4/5] data.yaml path: {data_yaml}", flush=True)
print(f"       exists: {os.path.exists(data_yaml)}", flush=True)

train_img = _DATA_YAML.parent / "train" / "images"
val_img = _DATA_YAML.parent / "valid" / "images"
print(f"       train/images count: {len(os.listdir(train_img)) if train_img.is_dir() else 'MISSING'}", flush=True)
print(f"       valid/images count: {len(os.listdir(val_img)) if val_img.is_dir() else 'MISSING'}", flush=True)

device = 0 if torch.cuda.is_available() else "cpu"
print(f"[5/5] Starting training on device={device} ...", flush=True)

try:
    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=100,
        batch=4,
        device=device,
        workers=0,
        project=str(_RUNS_YOLO),
        name="cigarette_pack_v2_roboflow",
    )
    print("TRAINING COMPLETE!", flush=True)
except Exception as e:
    print(f"\nTRAINING FAILED: {e}", flush=True)
    traceback.print_exc()
