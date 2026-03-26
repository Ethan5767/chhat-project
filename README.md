# RF-DETR Cigarette Brand Detection

Cigarette brand detection pipeline using RF-DETR for object detection, FAISS for reference image similarity search, and EasyOCR for text extraction. Uses a two-stage classification: bottom-half of each pack is analyzed first (brand name/logo area), falling back to full-image analysis only when confidence is below threshold.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

**Backend (FastAPI):**
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend (Streamlit):**
```bash
streamlit run frontend/app.py
```

## Structure

```
rf-detr-cigarette/
  backend/
    references/     # Reference images per brand (235 images)
    faiss_index/    # Built FAISS index (generated at runtime)
  frontend/         # Streamlit UI
  datasets/         # Training data
  runs/             # Training outputs
```
