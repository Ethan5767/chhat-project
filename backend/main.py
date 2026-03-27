import io
import json
import queue
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

try:
    from .pipeline import (
        CLASSIFIER_WEIGHTS,
        CLASS_MAPPING_JSON,
        build_index,
        get_device,
        load_dino,
        load_index,
        load_classifier,
        load_rfdetr,
        run_pipeline,
        classify_embeddings,
        _detect_brands_from_image,
        _format_brand_scores,
        label_to_product,
        _aggregate_to_products,
        MIN_OUTPUT_CONFIDENCE,
        RFDETR_CONF_THRESHOLD,
    )
except ImportError:
    from pipeline import (
        CLASSIFIER_WEIGHTS,
        CLASS_MAPPING_JSON,
        build_index,
        get_device,
        load_dino,
        load_index,
        load_classifier,
        load_rfdetr,
        run_pipeline,
        classify_embeddings,
        _detect_brands_from_image,
        _format_brand_scores,
        label_to_product,
        _aggregate_to_products,
        MIN_OUTPUT_CONFIDENCE,
        RFDETR_CONF_THRESHOLD,
    )

app = FastAPI(title="Local Cigarette Brand Detector")

_BACKEND_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = _BACKEND_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = UPLOADS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

jobs_lock = threading.Lock()
jobs: dict[str, dict] = {}


def _result_meta_path(job_id: str) -> Path:
    return RESULTS_DIR / f"{job_id}.json"


def _save_result_meta(job_id: str, result_path: Path) -> None:
    meta = {"job_id": job_id, "result": str(result_path)}
    _result_meta_path(job_id).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _load_result_meta(job_id: str) -> Path | None:
    meta_path = _result_meta_path(job_id)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        result = payload.get("result")
        if not result:
            return None
        path = Path(result)
        return path if path.exists() else None
    except Exception:
        return None


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "queue": queue.Queue(),
        }
    return job_id


def update_progress(job_id: str, current: int, total: int, message: str):
    pct = int((current / total) * 100) if total > 0 else 0
    with jobs_lock:
        job = jobs.get(job_id)
    if job:
        job["queue"].put((pct, message))


def run_build_index_job(job_id: str):
    try:
        device = get_device()
        build_index(device, progress_cb=lambda c, t, m: update_progress(job_id, c, t, f"Indexing: {m}"))
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = "INDEX_REBUILT"
    except Exception:
        err = traceback.format_exc()
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = err


def run_pipeline_job(job_id: str, csv_path: Path):
    try:
        out_path = run_pipeline(
            csv_path,
            progress_cb=lambda c, t, m: update_progress(job_id, c, t, m),
        )
        job_out_path = RESULTS_DIR / f"{job_id}_{out_path.name}"
        out_path.replace(job_out_path)
        _save_result_meta(job_id, job_out_path)
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = str(job_out_path)
    except Exception:
        err = traceback.format_exc()
        partial_path = csv_path.parent / f"{csv_path.stem}_results.csv"
        if partial_path.exists() and partial_path.stat().st_size > 0:
            job_out_path = RESULTS_DIR / f"{job_id}_{partial_path.name}"
            partial_path.replace(job_out_path)
            _save_result_meta(job_id, job_out_path)
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = err
                jobs[job_id]["result"] = str(job_out_path)
        else:
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = err


@app.on_event("startup")
def startup_event():
    device = get_device()
    print(f"[startup] device={device}")
    if not CLASSIFIER_WEIGHTS.exists() or not CLASS_MAPPING_JSON.exists():
        print("[startup] classifier not found -- run 'python brand_classifier.py' first")
        return
    try:
        load_classifier(device)
        print("[startup] classifier loaded")
    except Exception as exc:
        print(f"[startup] classifier load failed: {exc}")


@app.post("/build-index")
def build_index_endpoint():
    job_id = create_job()
    threading.Thread(target=run_build_index_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.post("/run-pipeline")
async def run_pipeline_endpoint(csv_file: UploadFile = File(...)):
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / csv_file.filename
    data = await csv_file.read()
    save_path.write_bytes(data)

    job_id = create_job()
    threading.Thread(target=run_pipeline_job, args=(job_id, save_path), daemon=True).start()
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
def progress_endpoint(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    def event_stream():
        while True:
            with jobs_lock:
                status = jobs[job_id]["status"]
                error = jobs[job_id]["error"]
            try:
                pct, message = job["queue"].get(timeout=0.5)
                yield f"data: {pct}|{message}\n\n"
            except queue.Empty:
                pass

            if status == "done":
                yield "data: DONE|\n\n"
                break
            if status == "error":
                yield f"data: ERROR|{error}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/download/{job_id}")
def download_endpoint(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    path = None
    if job and job.get("result") and job["result"] != "INDEX_REBUILT":
        path = Path(job["result"])
    if path is None or not path.exists():
        path = _load_result_meta(job_id)

    if path is None:
        raise HTTPException(status_code=404, detail="Result not available.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing.")
    return FileResponse(path=str(path), filename=path.name, media_type="text/csv")


@app.get("/index-status")
def index_status():
    if not CLASS_MAPPING_JSON.exists():
        return {"exists": False, "brand_count": 0, "brands": [], "products": []}
    with CLASS_MAPPING_JSON.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    brands = list(mapping.get("label_to_idx", {}).keys())
    products = sorted(set(label_to_product(b) for b in brands))
    return {"exists": True, "brand_count": len(brands), "brands": brands, "products": products}


@app.post("/detect-single")
async def detect_single(image_file: UploadFile = File(...)):
    """Run detection on a single image. Returns per-box brand assignments for interactive UI."""
    from PIL import Image
    import base64
    try:
        from .pipeline import (
            embed_images_batch,
            classify_embeddings,
            _build_label_profiles,
            _run_ocr_on_image,
            _ocr_brand_scores_from_items,
            _aggregate_to_products,
            CLASSIFIER_TOP_K,
            OCR_ENABLED,
            OCR_FULLIMG_ENABLED,
            OCR_STRONG_THRESHOLD,
            CLASSIFIER_HIGH_CONF,
        )
    except ImportError:
        from pipeline import (
            embed_images_batch,
            classify_embeddings,
            _build_label_profiles,
            _run_ocr_on_image,
            _ocr_brand_scores_from_items,
            _aggregate_to_products,
            CLASSIFIER_TOP_K,
            OCR_ENABLED,
            OCR_FULLIMG_ENABLED,
            OCR_STRONG_THRESHOLD,
            CLASSIFIER_HIGH_CONF,
        )

    data = await image_file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    device = get_device()
    index, labels = load_index()
    processor, model = load_dino(device)
    rfdetr_model = load_rfdetr()
    img_w, img_h = pil_img.size
    label_profiles = _build_label_profiles(labels)

    # RF-DETR detection
    detections = rfdetr_model.predict(pil_img, threshold=RFDETR_CONF_THRESHOLD)
    crops = []
    boxes_data = []
    has_detections = detections is not None and len(detections) > 0

    if has_detections:
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img_w, x2 + pad_x)
            y2 = min(img_h, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(pil_img.crop((x1, y1, x2, y2)))
            boxes_data.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_conf": round(float(conf), 3),
                "brands": [],
                "ocr_texts": [],
                "ocr_brand_scores": [],
            })
    else:
        crops.append(pil_img)
        boxes_data.append({
            "x1": 0, "y1": 0, "x2": img_w, "y2": img_h,
            "det_conf": 0.0,
            "is_full_image": True,
            "brands": [],
            "ocr_texts": [],
            "ocr_brand_scores": [],
        })

    # Classify all crops
    all_vecs = embed_images_batch(crops, processor, model, device)
    all_cls_results = classify_embeddings(all_vecs, device, top_k=CLASSIFIER_TOP_K)

    all_brand_scores = {}

    for crop_idx in range(len(crops)):
        # OCR per crop
        ocr_items = _run_ocr_on_image(crops[crop_idx]) if OCR_ENABLED else []
        ocr_texts = []
        for item in ocr_items:
            try:
                text = str(item[1]).strip()
                conf = float(item[2])
            except Exception:
                continue
            if not text or text.lower() in ("nan",) or len(text) < 2:
                continue
            ocr_texts.append({"text": text, "confidence": round(conf, 3)})
        ocr_texts.sort(key=lambda x: x["confidence"], reverse=True)
        boxes_data[crop_idx]["ocr_texts"] = ocr_texts[:8]

        # OCR brand scores
        ocr_scores = _ocr_brand_scores_from_items(ocr_items, label_profiles) if ocr_items else {}
        ocr_product_scores = _aggregate_to_products(ocr_scores)
        boxes_data[crop_idx]["ocr_brand_scores"] = [
            {"brand": b, "confidence": round(s, 3)}
            for b, s in sorted(ocr_product_scores.items(), key=lambda x: -x[1])[:5]
        ]

        # Classifier + OCR fusion for this crop
        crop_cls = dict(all_cls_results[crop_idx])
        label_profile_map = {p["label"]: p for p in label_profiles}
        ocr_families = {}
        for label, ocr_conf in ocr_scores.items():
            prof = label_profile_map.get(label, {})
            family = prof.get("brand", "")
            if family:
                ocr_families[family] = max(ocr_families.get(family, 0.0), ocr_conf)

        fused = {}
        for label, cls_conf in crop_cls.items():
            prof = label_profile_map.get(label, {})
            brand_family = prof.get("brand", "")
            ocr_fam = ocr_families.get(brand_family, 0.0) if brand_family else 0.0
            if ocr_fam >= OCR_STRONG_THRESHOLD:
                fused[label] = min(1.0, cls_conf + ocr_fam * 0.3)
            elif ocr_fam > 0:
                fused[label] = min(1.0, cls_conf + ocr_fam * 0.15)
            elif cls_conf >= CLASSIFIER_HIGH_CONF:
                fused[label] = cls_conf
            else:
                fused[label] = cls_conf * 0.7

        crop_products = _aggregate_to_products(fused)
        crop_brands = [{"brand": p, "confidence": round(c, 3)} for p, c in crop_products.items()]
        crop_brands.sort(key=lambda x: x["confidence"], reverse=True)
        boxes_data[crop_idx]["brands"] = crop_brands[:5]

        for label, conf in fused.items():
            if conf > all_brand_scores.get(label, 0.0):
                all_brand_scores[label] = conf

    # Encode image
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    all_product_scores = _aggregate_to_products(all_brand_scores)
    all_sorted = sorted(all_product_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "image_b64": img_b64,
        "image_width": img_w,
        "image_height": img_h,
        "boxes": boxes_data,
        "ocr_independent": [],
        "brands": [b for b, _ in all_sorted],
        "confidence": [round(c, 3) for _, c in all_sorted],
        "num_boxes": sum(1 for b in boxes_data if not b.get("is_full_image")),
    }


@app.post("/upload-coco")
async def upload_coco(coco_file: UploadFile = File(...)):
    """Upload a COCO JSON annotation file for RF-DETR training data."""
    if not coco_file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json COCO annotation files accepted.")

    DATASETS_DIR = _BACKEND_ROOT.parent / "datasets" / "cigarette_packs"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    data = await coco_file.read()
    try:
        coco_data = json.loads(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    n_images = len(coco_data.get("images", []))
    n_annotations = len(coco_data.get("annotations", []))

    save_path = DATASETS_DIR / coco_file.filename
    save_path.write_bytes(data)

    return {
        "status": "uploaded",
        "filename": coco_file.filename,
        "images": n_images,
        "annotations": n_annotations,
        "saved_to": str(save_path),
    }


@app.post("/generate-crops")
async def generate_crops(image_file: UploadFile = File(...)):
    """Run RF-DETR on an image and return all detected crops for labeling."""
    from PIL import Image
    import base64

    data = await image_file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    rfdetr_model = load_rfdetr()
    detections = rfdetr_model.predict(pil_img, threshold=RFDETR_CONF_THRESHOLD)

    crops = []
    if detections is not None and len(detections) > 0:
        width, height = pil_img.size
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.05), int(bh * 0.05)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil_img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=90)
            crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            crops.append({
                "index": i,
                "image_b64": crop_b64,
                "width": x2 - x1,
                "height": y2 - y1,
                "det_conf": round(float(conf), 3),
            })

    return {"num_crops": len(crops), "crops": crops}


@app.post("/add-reference")
async def add_reference(image_file: UploadFile = File(...), product_name: str = ""):
    """Add a confirmed crop as a reference image for a specific product."""
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name is required.")

    try:
        from .brand_registry import BRAND_REGISTRY
    except ImportError:
        from brand_registry import BRAND_REGISTRY

    # Validate product_name exists in registry
    valid_internals = set()
    for brand, products in BRAND_REGISTRY.items():
        for _, internal in products:
            valid_internals.add(internal)

    if product_name not in valid_internals:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown product '{product_name}'. Valid products: {sorted(valid_internals)}",
        )

    data = await image_file.read()
    from PIL import Image
    try:
        Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    REFERENCES_DIR = _BACKEND_ROOT / "references"
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    # Find next index
    import re
    existing = list(REFERENCES_DIR.glob(f"{product_name}_*.*"))
    max_idx = 0
    for p in existing:
        match = re.search(r"_(\d+)$", p.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    next_idx = max_idx + 1

    save_path = REFERENCES_DIR / f"{product_name}_{next_idx}.jpg"
    save_path.write_bytes(data)

    return {
        "status": "added",
        "product": product_name,
        "filename": save_path.name,
        "total_for_product": next_idx,
    }


@app.get("/brand-registry")
def get_brand_registry():
    """Return the full brand->products hierarchy."""
    try:
        from .brand_registry import BRAND_REGISTRY, audit_references
    except ImportError:
        from brand_registry import BRAND_REGISTRY, audit_references

    audit = audit_references()

    hierarchy = {}
    for brand, products in BRAND_REGISTRY.items():
        hierarchy[brand] = []
        for display_name, internal_name in products:
            count = audit["found"].get(internal_name, 0)
            hierarchy[brand].append({
                "display_name": display_name,
                "internal_name": internal_name,
                "reference_count": count,
            })

    return {
        "brands": hierarchy,
        "total_brands": len(BRAND_REGISTRY),
        "total_products": sum(len(p) for p in BRAND_REGISTRY.values()),
        "products_with_refs": audit.get("total_products_found", len(audit.get("found", {}))),
        "products_missing": audit.get("total_products_missing", len(audit.get("missing", []))),
        "total_images": audit.get("total_images", 0),
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": get_device()}
