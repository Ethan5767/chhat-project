"""Run fine-tuned YOLO on shelf images and save annotated results."""
import sys
from pathlib import Path

from ultralytics import YOLO

_BACKEND = Path(__file__).resolve().parent
_BRAND_ROOT = _BACKEND.parent
_FT = _BRAND_ROOT / "runs" / "yolo" / "cigarette_pack_v2_roboflow" / "weights" / "best.pt"
_PREV = _BRAND_ROOT / "runs" / "yolo" / "cigarette_rf_ft" / "weights" / "best.pt"
_LEGACY = Path(r"C:\Users\kimto\runs\detect\runs\cigarette_rf_ft\weights\best.pt")
if _FT.is_file():
    MODEL = str(_FT)
elif _PREV.is_file():
    MODEL = str(_PREV)
elif _LEGACY.is_file():
    MODEL = str(_LEGACY)
else:
    MODEL = "yolo11n.pt"
IMG_DIR = _BRAND_ROOT / "datasets" / "raw_shelf_images"
OUT_DIR = _BRAND_ROOT / "runs" / "yolo" / "predictions"
CONF = 0.25
IMGSZ = 640
MAX_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 10

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not IMG_DIR.is_dir():
    raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

print(f"Loading model: {MODEL}")
model = YOLO(MODEL)

images = sorted([f for f in IMG_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
images = images[:MAX_IMAGES]
print(f"Running on {len(images)} / {len([f for f in IMG_DIR.iterdir() if f.is_file()])} images  (pass a number arg to change)\n")

results = model.predict(source=images, conf=CONF, imgsz=IMGSZ, save=True, save_txt=True,
                        project=str(OUT_DIR), name="run", device=0)

print(f"\n{'='*60}")
for r in results:
    name = Path(r.path).name
    boxes = r.boxes
    print(f"\n{name}  — {len(boxes)} detections")
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = b.conf[0].item()
        cls = int(b.cls[0].item())
        print(f"  [{cls}] conf={conf:.3f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

print(f"\nAnnotated images saved to: {OUT_DIR / 'run'}")
