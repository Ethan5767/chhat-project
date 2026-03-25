# Local Cigarette Brand Detector

This project runs fully on a local laptop, split into:
- `backend/` (FastAPI + YOLO + DINOv2 + FAISS)
- `frontend/` (Streamlit UI)
- `datasets/` (raw shelf photos + YOLO training data — see `datasets/README.md`)
- `runs/yolo/` (training weights and prediction outputs)

## Prerequisites

- Python 3.10+
- `pip`

## Setup

```bash
cd backend
pip install -r requirements.txt
cd ../frontend
pip install -r requirements.txt
```

## Add Reference Images

Drop brand images into `backend/references/`.

Name each image after the brand label you want:
- `Marlboro.jpg`
- `Camel.png`
- etc.

The filename stem becomes the detected brand label.

## Run

Terminal 1:

```bash
cd backend
uvicorn main:app --port 8000 --reload
```

Terminal 2:

```bash
cd frontend
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

## Notes

- The backend uses `BATCH_MODE` in `backend/pipeline.py`.
- Set `BATCH_MODE = None` to process all rows in an uploaded CSV.

## Fast Auto-Label Workflow (Avoid Manual One-by-One)

Use pseudo-labels, then only correct mistakes in CVAT/Label Studio.

1) Put shelf images in:

`datasets/raw_shelf_images/`

(or download from Excel with `python download_shelf_images.py --excel ...` — default output is that folder)

2) Generate YOLO-format pseudo-labels:

```bash
cd backend
python autolabel_yolo.py --image-dir ../datasets/raw_shelf_images --model yolo11n.pt --conf 0.20 --imgsz 960
```

This creates (default output):

- `datasets/yolo_autolabel_scratch/images/train/*.jpg|png`
- `datasets/yolo_autolabel_scratch/labels/train/*.txt`
- `datasets/yolo_autolabel_scratch/data.yaml`

3) Import those `images/train` + `labels/train` into CVAT/Label Studio and correct only:
- remove bad boxes
- add missed cigarette packs
- tighten wrong boxes

4) Annotation rule:
- Label **each visible cigarette pack** (not the whole cluster/shelf block).

5) After correction, export a proper split (`train/`, `valid/`, `test/`) into `datasets/yolo_cigarette_packs/` (or add a new folder under `datasets/`) and update its `data.yaml`. Then train:

```bash
cd backend
python run_train.py
```

Or manually:

```bash
yolo detect train model=yolo11n.pt data=../datasets/yolo_cigarette_packs/data.yaml imgsz=640 epochs=50 batch=4 device=0 workers=0 project=../runs/yolo name=cigarette_rf_ft
```

Fine-tuned weights are written to `runs/yolo/<run_name>/weights/best.pt`. The backend loads `runs/yolo/cigarette_rf_ft/weights/best.pt` when present.
