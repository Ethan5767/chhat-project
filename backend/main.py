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
        INDEX_BIN,
        LABELS_JSON,
        build_index,
        get_device,
        load_dino,
        load_index,
        load_rfdetr,
        run_pipeline,
        _detect_brands_from_image,
        _format_brand_scores,
        label_to_product,
        _aggregate_to_products,
        MIN_OUTPUT_CONFIDENCE,
        RFDETR_CONF_THRESHOLD,
    )
except ImportError:
    from pipeline import (
        INDEX_BIN,
        LABELS_JSON,
        build_index,
        get_device,
        load_dino,
        load_index,
        load_rfdetr,
        run_pipeline,
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
    if not INDEX_BIN.exists() or not LABELS_JSON.exists():
        print("[startup] index not found, building now...")
        build_index(device)
        print("[startup] index built")


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
    if not LABELS_JSON.exists():
        return {"exists": False, "brand_count": 0, "brands": [], "products": []}
    with LABELS_JSON.open("r", encoding="utf-8") as f:
        brands = json.load(f)
    products = sorted(set(label_to_product(b) for b in brands))
    return {"exists": True, "brand_count": len(brands), "brands": brands, "products": products}


@app.post("/detect-single")
async def detect_single(image_file: UploadFile = File(...)):
    """Run detection on a single image. Returns per-box brand assignments for interactive UI."""
    from PIL import Image
    import numpy as np
    import base64
    try:
        from .pipeline import (
            embed_images_batch,
            distance_to_confidence,
            _build_label_profiles,
            _run_ocr_on_image,
            _ocr_brand_scores_from_items,
            label_to_product,
            _aggregate_to_products,
            FAISS_TOP_K,
            OCR_ENABLED,
            OCR_FULLIMG_ENABLED,
            DINO_PRODUCT_WEIGHT,
            OCR_BRAND_WEIGHT,
            NO_CONSENSUS_PENALTY,
            OCR_BRAND_CONFIRM_THRESHOLD,
            OCR_INDEPENDENT_WEIGHT,
            OCR_INDEPENDENT_MIN_SCORE,
        )
    except ImportError:
        from pipeline import (
            embed_images_batch,
            distance_to_confidence,
            _build_label_profiles,
            _run_ocr_on_image,
            _ocr_brand_scores_from_items,
            label_to_product,
            _aggregate_to_products,
            FAISS_TOP_K,
            OCR_ENABLED,
            OCR_FULLIMG_ENABLED,
            DINO_PRODUCT_WEIGHT,
            OCR_BRAND_WEIGHT,
            NO_CONSENSUS_PENALTY,
            OCR_BRAND_CONFIRM_THRESHOLD,
            OCR_INDEPENDENT_WEIGHT,
            OCR_INDEPENDENT_MIN_SCORE,
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

    # ── RF-DETR detection ──
    detections = rfdetr_model.predict(pil_img, threshold=RFDETR_CONF_THRESHOLD)

    crops: list[Image.Image] = []
    boxes_data: list[dict] = []
    has_detections = detections is not None and len(detections) > 0

    if has_detections:
        xyxy = detections.xyxy  # already numpy
        confs = detections.confidence  # already numpy
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(1, min(x2, img_w))
            y2 = max(1, min(y2, img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(pil_img.crop((x1, y1, x2, y2)))
            boxes_data.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_conf": round(float(confs[i]), 3),
                "brands": [],   # filled below
            })
    else:
        crops.append(pil_img)
        boxes_data.append({
            "x1": 0, "y1": 0, "x2": img_w, "y2": img_h,
            "det_conf": 0.0,
            "is_full_image": True,
            "brands": [],
        })

    # ── Per-crop OCR ──
    label_profiles = _build_label_profiles(labels)
    label_profile_map = {p["label"]: p for p in label_profiles}

    crop_ocr_items = []
    if OCR_ENABLED:
        for crop in crops:
            crop_ocr_items.append(_run_ocr_on_image(crop))
    else:
        crop_ocr_items = [[] for _ in crops]

    # Pre-compute OCR brand scores per crop so we can both (a) fuse and (b) show OCR debug.
    # easyocr detail=1 returns items shaped like: (bbox, text, confidence)
    crop_ocr_scores_list: list[dict[str, float]] = []
    if OCR_ENABLED:
        for ocr_items in crop_ocr_items:
            crop_ocr_scores_list.append(_ocr_brand_scores_from_items(ocr_items, label_profiles))
    else:
        crop_ocr_scores_list = [{} for _ in crops]

    # ── Full-image OCR ──
    fullimg_ocr_items = []
    if OCR_ENABLED and OCR_FULLIMG_ENABLED and has_detections:
        fullimg_ocr_items = _run_ocr_on_image(pil_img)
    elif OCR_ENABLED and not has_detections:
        fullimg_ocr_items = crop_ocr_items[0] if crop_ocr_items else []

    # ── DINO batch embed ──
    all_vecs = embed_images_batch(crops, processor, model, device)

    # ── Collect raw DINO scores and per-crop OCR scores ──
    k = max(1, min(FAISS_TOP_K, len(labels)))
    dino_best: dict[str, float] = {}
    crop_dino_results: list[list[tuple[str, float]]] = []

    for crop_idx in range(len(crops)):
        vec = all_vecs[crop_idx].reshape(1, -1)
        distances, indices = index.search(vec, k=k)
        crop_results = []
        for rank in range(k):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(labels):
                continue
            label = labels[idx]
            dist = float(distances[0][rank])
            conf = distance_to_confidence(dist)
            crop_results.append((label, conf))
            if conf > dino_best.get(label, 0.0):
                dino_best[label] = conf
        crop_dino_results.append(crop_results)

    # Collect OCR scores across all crops (global max, used by the existing fusion logic).
    ocr_best: dict[str, float] = {}
    if OCR_ENABLED:
        for crop_ocr_scores in crop_ocr_scores_list:
            for brand, conf in crop_ocr_scores.items():
                if conf > ocr_best.get(brand, 0.0):
                    ocr_best[brand] = conf
    fullimg_scores = {}
    if fullimg_ocr_items:
        fullimg_scores = _ocr_brand_scores_from_items(fullimg_ocr_items, label_profiles)
        for brand, conf in fullimg_scores.items():
            if conf > ocr_best.get(brand, 0.0):
                ocr_best[brand] = conf

    # Build OCR family-level lookup
    ocr_family_best: dict[str, float] = {}
    for label, ocr_conf in ocr_best.items():
        prof = label_profile_map.get(label, {})
        family = prof.get("brand", "")
        if family:
            ocr_family_best[family] = max(ocr_family_best.get(family, 0.0), ocr_conf)

    # ── Brand-consensus fusion per crop (for box tooltips) ──
    all_brand_scores: dict[str, float] = {}

    for crop_idx in range(len(crops)):
        # ── OCR debug per crop ──
        ocr_items_for_box = crop_ocr_items[crop_idx] if OCR_ENABLED else []
        ocr_texts = []
        for item in ocr_items_for_box:
            try:
                text = str(item[1]).strip()
                conf = float(item[2])
            except Exception:
                continue
            if not text or text.lower() in ("nan",):
                continue
            if len(text) < 2:
                continue
            ocr_texts.append({"text": text, "confidence": round(conf, 3)})
        # Sort by OCR confidence and keep short preview.
        ocr_texts.sort(key=lambda x: x["confidence"], reverse=True)
        boxes_data[crop_idx]["ocr_texts"] = ocr_texts[:8]

        ocr_brand_scores_map = crop_ocr_scores_list[crop_idx] if OCR_ENABLED else {}
        ocr_product_scores = _aggregate_to_products(ocr_brand_scores_map)
        ocr_brand_scores_sorted = sorted(ocr_product_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        boxes_data[crop_idx]["ocr_brand_scores"] = [
            {"brand": brand, "confidence": round(score, 3)} for brand, score in ocr_brand_scores_sorted
        ]

        crop_brands_raw = {}
        for label, dino_conf in crop_dino_results[crop_idx]:
            prof = label_profile_map.get(label, {})
            brand_family = prof.get("brand", "")
            ocr_family_score = ocr_family_best.get(brand_family, 0.0) if brand_family else 0.0
            ocr_label_score = ocr_best.get(label, 0.0)

            if ocr_family_score >= OCR_BRAND_CONFIRM_THRESHOLD:
                if ocr_label_score >= OCR_BRAND_CONFIRM_THRESHOLD:
                    final_conf = DINO_PRODUCT_WEIGHT * dino_conf + OCR_BRAND_WEIGHT * ocr_label_score
                else:
                    final_conf = DINO_PRODUCT_WEIGHT * dino_conf + OCR_BRAND_WEIGHT * ocr_family_score * 0.5
            else:
                final_conf = dino_conf * NO_CONSENSUS_PENALTY

            final_conf = min(1.0, final_conf)
            crop_brands_raw[label] = max(crop_brands_raw.get(label, 0.0), final_conf)
            if final_conf > all_brand_scores.get(label, 0.0):
                all_brand_scores[label] = final_conf

        # Aggregate to product level for this box
        crop_products = _aggregate_to_products(crop_brands_raw)
        crop_brands = [{"brand": p, "confidence": round(c, 3)} for p, c in crop_products.items()]
        crop_brands.sort(key=lambda x: x["confidence"], reverse=True)
        boxes_data[crop_idx]["brands"] = crop_brands[:3]

    # ── OCR independent brands (not found by DINO) ──
    ocr_independent: list[dict] = []
    for label, ocr_conf in ocr_best.items():
        if label in all_brand_scores:
            continue
        if ocr_conf >= OCR_INDEPENDENT_MIN_SCORE:
            indep_conf = min(1.0, ocr_conf * OCR_INDEPENDENT_WEIGHT)
            product = label_to_product(label)
            ocr_independent.append({"brand": product, "confidence": round(indep_conf, 3), "source": "ocr"})
            if indep_conf > all_brand_scores.get(label, 0.0):
                all_brand_scores[label] = indep_conf

    # ── Encode original (clean) image as base64 ──
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    import base64
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Aggregate all brands to product level, sorted ──
    all_product_scores = _aggregate_to_products(all_brand_scores)
    all_sorted = sorted(all_product_scores.items(), key=lambda x: x[1], reverse=True)
    all_brands = [b for b, _ in all_sorted]
    all_confs = [round(c, 3) for _, c in all_sorted]

    return {
        "image_b64": img_b64,
        "image_width": img_w,
        "image_height": img_h,
        "boxes": boxes_data,
        "ocr_independent": ocr_independent,
        "brands": all_brands,
        "confidence": all_confs,
        "num_boxes": sum(1 for b in boxes_data if not b.get("is_full_image")),
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": get_device()}
