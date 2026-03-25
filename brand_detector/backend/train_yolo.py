from pathlib import Path

from ultralytics import YOLO

_BACKEND = Path(__file__).resolve().parent
_BRAND_ROOT = _BACKEND.parent
_DATA_YAML = _BRAND_ROOT / "datasets" / "yolo_cigarette_packs" / "data.yaml"
_RUNS_YOLO = _BRAND_ROOT / "runs" / "yolo"


def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(_DATA_YAML),
        imgsz=640,
        epochs=100,
        batch=4,
        device=0,
        workers=0,
        project=str(_RUNS_YOLO),
        name="cigarette_autolabel_ft",
    )


if __name__ == "__main__":
    main()
