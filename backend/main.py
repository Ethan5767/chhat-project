import io
import hashlib
import json
import queue
import threading
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
        PACKAGING_TYPES,
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
        PACKAGING_TYPES,
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
            OCR_FALLBACK_THRESHOLD,
            OCR_FALLBACK_MARGIN,
            OCR_STRONG_THRESHOLD,
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
            OCR_FALLBACK_THRESHOLD,
            OCR_FALLBACK_MARGIN,
            OCR_STRONG_THRESHOLD,
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
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
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
            pkg_type = "pack"
            if class_ids is not None and len(class_ids) > i:
                pkg_type = "box" if int(class_ids[i]) == 1 else "pack"
            boxes_data.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_conf": round(float(conf), 3),
                "packaging_type": pkg_type,
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
            "packaging_type": "pack",
            "brands": [],
            "ocr_texts": [],
            "ocr_brand_scores": [],
        })

    # Classify crops grouped by packaging type
    all_vecs = embed_images_batch(crops, processor, model, device)
    crop_pkg_types = [b.get("packaging_type", "pack") for b in boxes_data]

    type_indices: dict[str, list[int]] = {}
    for idx, pkg_type in enumerate(crop_pkg_types):
        type_indices.setdefault(pkg_type, []).append(idx)

    all_cls_results: list[list[tuple[str, float]]] = [[] for _ in crops]
    for pkg_type, indices in type_indices.items():
        try:
            type_vecs = all_vecs[indices]
            type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type=pkg_type)
            for local_idx, global_idx in enumerate(indices):
                all_cls_results[global_idx] = type_results[local_idx]
        except FileNotFoundError:
            # Fall back to pack classifier
            type_vecs = all_vecs[indices]
            type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type="pack")
            for local_idx, global_idx in enumerate(indices):
                all_cls_results[global_idx] = type_results[local_idx]

    all_brand_scores = {}

    for crop_idx in range(len(crops)):
        crop_cls_ranked = all_cls_results[crop_idx]
        crop_cls = dict(crop_cls_ranked)
        top1 = float(crop_cls_ranked[0][1]) if crop_cls_ranked else 0.0
        top2 = float(crop_cls_ranked[1][1]) if len(crop_cls_ranked) > 1 else 0.0
        margin = top1 - top2
        should_run_ocr = OCR_ENABLED and (top1 < OCR_FALLBACK_THRESHOLD or margin < OCR_FALLBACK_MARGIN)

        # OCR fallback per crop
        ocr_items = _run_ocr_on_image(crops[crop_idx]) if should_run_ocr else []
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

        # Classifier + OCR fallback fusion for this crop
        label_profile_map = {p["label"]: p for p in label_profiles}
        ocr_families = {}
        for label, ocr_conf in ocr_scores.items():
            prof = label_profile_map.get(label, {})
            family = prof.get("brand", "")
            if family:
                ocr_families[family] = max(ocr_families.get(family, 0.0), ocr_conf)

        fused = {}
        for label, cls_conf in crop_cls.items():
            out_conf = float(cls_conf)
            if should_run_ocr:
                prof = label_profile_map.get(label, {})
                brand_family = prof.get("brand", "")
                ocr_fam = ocr_families.get(brand_family, 0.0) if brand_family else 0.0
                if ocr_fam >= OCR_STRONG_THRESHOLD:
                    out_conf = min(1.0, out_conf + ocr_fam * 0.25)
                elif ocr_fam > 0:
                    out_conf = min(1.0, out_conf + ocr_fam * 0.10)
            fused[label] = out_conf

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
    """Upload a COCO JSON annotation file or ZIP export for RF-DETR training data."""
    filename = (coco_file.filename or "").strip()
    lower = filename.lower()
    if not (lower.endswith(".json") or lower.endswith(".zip")):
        raise HTTPException(status_code=400, detail="Only .json or .zip COCO files accepted.")

    DATASETS_DIR = _BACKEND_ROOT.parent / "datasets" / "cigarette_packs"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    data = await coco_file.read()
    if lower.endswith(".json"):
        try:
            coco_data = json.loads(data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON file.")

        n_images = len(coco_data.get("images", []))
        n_annotations = len(coco_data.get("annotations", []))
        save_path = DATASETS_DIR / filename
        save_path.write_bytes(data)
        return {
            "status": "uploaded",
            "file_type": "json",
            "filename": filename,
            "images": n_images,
            "annotations": n_annotations,
            "saved_to": str(save_path),
        }

    try:
        zf = zipfile.ZipFile(BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")

    members = [m for m in zf.namelist() if not m.endswith("/")]
    ann_candidates = [m for m in members if m.lower().endswith("_annotations.coco.json")]
    if not ann_candidates:
        raise HTTPException(status_code=400, detail="ZIP must include _annotations.coco.json.")

    def _safe_parts(member_name: str) -> list[str]:
        return [p for p in member_name.replace("\\", "/").split("/") if p not in ("", ".", "..")]

    def _extract_member(member_name: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member_name) as src, out_path.open("wb") as dst:
            dst.write(src.read())

    target_splits = ("train", "valid", "test")
    split_ann_map: dict[str, str] = {}
    for ann in ann_candidates:
        parts = [p.lower() for p in _safe_parts(ann)]
        for split in target_splits:
            if split in parts:
                split_ann_map[split] = ann
    if not split_ann_map:
        split_ann_map["train"] = ann_candidates[0]

    extracted_images = 0
    extracted_annotations = 0
    extracted_files = 0
    saved_dirs: list[str] = []

    for split, ann_member in split_ann_map.items():
        split_dir = DATASETS_DIR / split
        saved_dirs.append(str(split_dir))
        ann_out = split_dir / "_annotations.coco.json"
        _extract_member(ann_member, ann_out)
        extracted_files += 1

        try:
            coco_data = json.loads(ann_out.read_text(encoding="utf-8"))
        except Exception:
            continue

        extracted_images += len(coco_data.get("images", []))
        extracted_annotations += len(coco_data.get("annotations", []))

        ann_parts = _safe_parts(ann_member)
        ann_prefix = "/".join(ann_parts[:-1])
        for img in coco_data.get("images", []):
            img_name = img.get("file_name", "")
            if not img_name:
                continue
            safe_img_name = "/".join(_safe_parts(img_name))
            if not safe_img_name:
                continue
            prefixed_candidate = f"{ann_prefix}/{safe_img_name}" if ann_prefix else safe_img_name
            if prefixed_candidate in members:
                _extract_member(prefixed_candidate, split_dir / safe_img_name)
                extracted_files += 1
            elif safe_img_name in members:
                _extract_member(safe_img_name, split_dir / safe_img_name)
                extracted_files += 1

    return {
        "status": "uploaded",
        "file_type": "zip",
        "filename": filename,
        "images": extracted_images,
        "annotations": extracted_annotations,
        "extracted_files": extracted_files,
        "splits": sorted(split_ann_map.keys()),
        "saved_to": saved_dirs,
    }


@app.post("/download-roboflow-coco")
def download_roboflow_coco(url: str = Form(...), clean: bool = Form(False)):
    """Download a Roboflow raw dataset URL and extract COCO files/images."""
    from urllib.parse import parse_qs, urlparse
    import requests
    import shutil

    parsed = urlparse(url.strip())
    if "roboflow.com" not in parsed.netloc:
        raise HTTPException(status_code=400, detail="URL must be from roboflow.com")
    if "key" not in parse_qs(parsed.query):
        raise HTTPException(status_code=400, detail="Roboflow URL must include ?key=...")

    datasets_dir = _BACKEND_ROOT.parent / "datasets" / "cigarette_packs"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if clean and datasets_dir.exists():
        shutil.rmtree(datasets_dir)
        datasets_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url.strip(), timeout=120)
        resp.raise_for_status()
        payload = resp.content
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {exc}")

    # Reuse upload extraction path by treating payload as uploaded zip
    try:
        zf = zipfile.ZipFile(BytesIO(payload))
    except Exception:
        raise HTTPException(status_code=400, detail="Downloaded file is not a valid ZIP.")

    members = [m for m in zf.namelist() if not m.endswith("/")]
    ann_candidates = [m for m in members if m.lower().endswith("_annotations.coco.json")]
    if not ann_candidates:
        raise HTTPException(status_code=400, detail="ZIP must include _annotations.coco.json.")

    def _safe_parts(member_name: str) -> list[str]:
        return [p for p in member_name.replace("\\", "/").split("/") if p not in ("", ".", "..")]

    def _extract_member(member_name: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member_name) as src, out_path.open("wb") as dst:
            dst.write(src.read())

    target_splits = ("train", "valid", "test")
    split_ann_map: dict[str, str] = {}
    for ann in ann_candidates:
        parts = [p.lower() for p in _safe_parts(ann)]
        for split in target_splits:
            if split in parts:
                split_ann_map[split] = ann
    if not split_ann_map:
        split_ann_map["train"] = ann_candidates[0]

    extracted_images = 0
    extracted_annotations = 0
    extracted_files = 0
    saved_dirs: list[str] = []

    for split, ann_member in split_ann_map.items():
        split_dir = datasets_dir / split
        saved_dirs.append(str(split_dir))
        ann_out = split_dir / "_annotations.coco.json"
        _extract_member(ann_member, ann_out)
        extracted_files += 1
        try:
            coco_data = json.loads(ann_out.read_text(encoding="utf-8"))
        except Exception:
            continue

        extracted_images += len(coco_data.get("images", []))
        extracted_annotations += len(coco_data.get("annotations", []))

        ann_parts = _safe_parts(ann_member)
        ann_prefix = "/".join(ann_parts[:-1])
        for img in coco_data.get("images", []):
            img_name = img.get("file_name", "")
            if not img_name:
                continue
            safe_img_name = "/".join(_safe_parts(img_name))
            if not safe_img_name:
                continue
            prefixed_candidate = f"{ann_prefix}/{safe_img_name}" if ann_prefix else safe_img_name
            if prefixed_candidate in members:
                _extract_member(prefixed_candidate, split_dir / safe_img_name)
                extracted_files += 1
            elif safe_img_name in members:
                _extract_member(safe_img_name, split_dir / safe_img_name)
                extracted_files += 1

    return {
        "status": "downloaded",
        "source": "roboflow_url",
        "images": extracted_images,
        "annotations": extracted_annotations,
        "extracted_files": extracted_files,
        "splits": sorted(split_ann_map.keys()),
        "saved_to": saved_dirs,
    }


@app.post("/generate-crops")
async def generate_crops(image_file: UploadFile = File(...)):
    """Run RF-DETR on an image, classify each crop, and return for labeling."""
    from PIL import Image
    import base64
    try:
        from .pipeline import (
            embed_images_batch, classify_embeddings,
        )
        from .brand_registry import get_brand, resolve_internal_name
    except ImportError:
        from pipeline import (
            embed_images_batch, classify_embeddings,
        )
        from brand_registry import get_brand, resolve_internal_name

    data = await image_file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    device = get_device()
    rfdetr_model = load_rfdetr()
    detections = rfdetr_model.predict(pil_img, threshold=RFDETR_CONF_THRESHOLD)

    crop_images = []
    crop_meta = []
    if detections is not None and len(detections) > 0:
        width, height = pil_img.size
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.05), int(bh * 0.05)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil_img.crop((x1, y1, x2, y2))
            crop_images.append(crop)
            pkg_type = "pack"
            if class_ids is not None and len(class_ids) > i:
                pkg_type = "box" if int(class_ids[i]) == 1 else "pack"
            crop_meta.append({"index": i, "w": x2 - x1, "h": y2 - y1, "conf": float(conf), "packaging_type": pkg_type})

    suggested_labels = []
    if crop_images:
        try:
            processor, model = load_dino(device)

            vecs = embed_images_batch(crop_images, processor, model, device)

            # Group by packaging type for classification
            type_indices: dict[str, list[int]] = {}
            for idx, meta in enumerate(crop_meta):
                pkg = meta["packaging_type"]
                type_indices.setdefault(pkg, []).append(idx)

            per_crop_results: list[tuple[str, float]] = [("unknown", 0.0)] * len(crop_images)
            for pkg_type, indices in type_indices.items():
                try:
                    type_vecs = vecs[indices]
                    cls_results = classify_embeddings(type_vecs, device, top_k=3, packaging_type=pkg_type)
                    for local_idx, global_idx in enumerate(indices):
                        if cls_results[local_idx]:
                            per_crop_results[global_idx] = cls_results[local_idx][0]
                except FileNotFoundError:
                    pass  # No classifier for this type yet

            for crop_idx, top_pred in enumerate(per_crop_results):
                internal_name = resolve_internal_name(top_pred[0])
                cls_conf = top_pred[1]
                brand = get_brand(internal_name)
                suggested_labels.append({
                    "internal_name": internal_name,
                    "brand": brand,
                    "confidence": round(cls_conf, 3),
                })
        except Exception:
            suggested_labels = [{"internal_name": "", "brand": "", "confidence": 0.0}] * len(crop_images)

    # Build response
    crops = []
    for idx, (crop, meta) in enumerate(zip(crop_images, crop_meta)):
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
        crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        suggestion = suggested_labels[idx] if idx < len(suggested_labels) else {}
        crops.append({
            "index": meta["index"],
            "image_b64": crop_b64,
            "width": meta["w"],
            "height": meta["h"],
            "det_conf": round(meta["conf"], 3),
            "packaging_type": meta["packaging_type"],
            "suggested_brand": suggestion.get("brand", ""),
            "suggested_product": suggestion.get("internal_name", ""),
            "suggested_confidence": suggestion.get("confidence", 0.0),
        })

    return {"num_crops": len(crops), "crops": crops}


@app.post("/add-reference")
async def add_reference(
    image_file: UploadFile = File(...),
    product_name: str = Form(...),
    packaging_type: str = Form("pack"),
):
    """Add a confirmed crop as a reference image for a specific product and packaging type."""
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name is required.")
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'.")

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

    TYPE_DIR = _BACKEND_ROOT / "references" / packaging_type
    TYPE_DIR.mkdir(parents=True, exist_ok=True)

    import re
    existing = list(TYPE_DIR.glob(f"{product_name}_*.*"))
    max_idx = 0
    for p in existing:
        match = re.search(r"_(\d+)$", p.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    next_idx = max_idx + 1

    save_path = TYPE_DIR / f"{product_name}_{next_idx}.jpg"
    save_path.write_bytes(data)

    return {
        "status": "added",
        "product": product_name,
        "packaging_type": packaging_type,
        "filename": save_path.name,
        "total_for_product": next_idx,
    }


@app.get("/brand-registry")
def get_brand_registry():
    """Return the full brand->products hierarchy with per-type reference counts."""
    try:
        from .brand_registry import BRAND_REGISTRY, audit_references
    except ImportError:
        from brand_registry import BRAND_REGISTRY, audit_references

    audit = audit_references()

    hierarchy = {}
    for brand, products in BRAND_REGISTRY.items():
        hierarchy[brand] = []
        for display_name, internal_name in products:
            found_entry = audit["found"].get(internal_name, {})
            # found_entry is now {pkg_type: count} dict
            pack_count = found_entry.get("pack", 0) if isinstance(found_entry, dict) else found_entry
            box_count = found_entry.get("box", 0) if isinstance(found_entry, dict) else 0
            hierarchy[brand].append({
                "display_name": display_name,
                "internal_name": internal_name,
                "reference_count": pack_count + box_count,
                "pack_count": pack_count,
                "box_count": box_count,
            })

    return {
        "brands": hierarchy,
        "total_brands": len(BRAND_REGISTRY),
        "total_products": sum(len(p) for p in BRAND_REGISTRY.values()),
        "products_with_refs": audit.get("total_products_found", len(audit.get("found", {}))),
        "products_missing": audit.get("total_products_missing", len(audit.get("missing", []))),
        "total_images": audit.get("total_images", 0),
        "per_type": audit.get("per_type", {}),
    }


@app.get("/reference-image/{packaging_type}/{filename}")
def get_reference_image(packaging_type: str, filename: str):
    """Serve a reference image by packaging type and filename."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _BACKEND_ROOT / "references" / packaging_type
    path = REFERENCES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.delete("/reference-image/{packaging_type}/{filename}")
def delete_reference_image(packaging_type: str, filename: str):
    """Delete a reference image by packaging type and filename."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _BACKEND_ROOT / "references" / packaging_type
    path = REFERENCES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if not path.resolve().parent == REFERENCES_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid path")
    path.unlink()
    return {"status": "deleted", "packaging_type": packaging_type, "filename": filename}


@app.get("/reference-images/{product_name}")
def list_reference_images(product_name: str, packaging_type: str = "pack"):
    """List all reference image filenames for a product in a packaging type subfolder."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _BACKEND_ROOT / "references" / packaging_type
    files = sorted(REFERENCES_DIR.glob(f"{product_name}_*.*")) if REFERENCES_DIR.exists() else []
    return {
        "product": product_name,
        "packaging_type": packaging_type,
        "count": len(files),
        "filenames": [f.name for f in files],
    }


@app.get("/dataset-status")
def dataset_status():
    """Check if COCO dataset splits exist for RF-DETR training."""
    ds_root = _BACKEND_ROOT.parent / "datasets" / "cigarette_packs"
    splits = {}
    for split in ("train", "valid", "test"):
        ann = ds_root / split / "_annotations.coco.json"
        if ann.exists():
            import json as json_mod
            try:
                data = json_mod.loads(ann.read_text(encoding="utf-8"))
                splits[split] = {
                    "exists": True,
                    "images": len(data.get("images", [])),
                    "annotations": len(data.get("annotations", [])),
                }
            except Exception:
                splits[split] = {"exists": True, "images": 0, "annotations": 0}
        else:
            splits[split] = {"exists": False, "images": 0, "annotations": 0}
    ready = splits.get("train", {}).get("exists", False) and splits.get("valid", {}).get("exists", False)
    return {"ready": ready, "splits": splits}


_TRAINING_PROGRESS_DIR = _BACKEND_ROOT.parent
_training_jobs: dict[str, dict] = {}
_training_processes: dict[str, object] = {}
_training_lock = threading.Lock()
_TRAINING_HISTORY_PATH = _BACKEND_ROOT / "training_history.json"
_MODEL_REGISTRY_PATH = _BACKEND_ROOT / "model_registry.json"
_VERSION_STATE_PATH = _BACKEND_ROOT / "training_version_state.json"
DEFAULT_TRAINING_VERSION = "v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_training_history() -> list[dict]:
    if not _TRAINING_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(_TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _save_training_history(items: list[dict]) -> None:
    _TRAINING_HISTORY_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_training_history(entry: dict) -> None:
    with _training_lock:
        history = _load_training_history()
        history.append(entry)
        _save_training_history(history)


def _update_training_history(job_id: str, patch: dict) -> None:
    with _training_lock:
        history = _load_training_history()
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("job_id") == job_id:
                history[i].update(patch)
                _save_training_history(history)
                return


def _load_model_registry() -> list[dict]:
    if not _MODEL_REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(_MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _save_model_registry(items: list[dict]) -> None:
    _MODEL_REGISTRY_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_model_registry(entry: dict) -> None:
    with _training_lock:
        rows = _load_model_registry()
        rows.append(entry)
        _save_model_registry(rows)


def _update_model_registry(job_id: str, patch: dict) -> None:
    with _training_lock:
        rows = _load_model_registry()
        for i in range(len(rows) - 1, -1, -1):
            if rows[i].get("job_id") == job_id:
                rows[i].update(patch)
                _save_model_registry(rows)
                return


def _parse_version_num(version: str) -> int:
    if isinstance(version, str) and version.startswith("v") and version[1:].isdigit():
        return int(version[1:])
    return 1


def _next_version(version: str) -> str:
    return f"v{_parse_version_num(version) + 1}"


def _load_version_state() -> dict:
    if not _VERSION_STATE_PATH.exists():
        return {"current_version": DEFAULT_TRAINING_VERSION, "last_trained_version": None}
    try:
        payload = json.loads(_VERSION_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"current_version": DEFAULT_TRAINING_VERSION, "last_trained_version": None}
        payload.setdefault("current_version", DEFAULT_TRAINING_VERSION)
        payload.setdefault("last_trained_version", None)
        return payload
    except Exception:
        return {"current_version": DEFAULT_TRAINING_VERSION, "last_trained_version": None}


def _save_version_state(state: dict) -> None:
    _VERSION_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_current_training_version() -> str:
    return _load_version_state().get("current_version", DEFAULT_TRAINING_VERSION)


def _mark_training_completed(version: str) -> str:
    with _training_lock:
        state = _load_version_state()
        state["last_trained_version"] = version
        if state.get("current_version") == version:
            state["current_version"] = _next_version(version)
        _save_version_state(state)
        return state.get("current_version", DEFAULT_TRAINING_VERSION)


def _hash_dataset_dir(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    for p in files:
        rel = str(p.relative_to(path)).replace("\\", "/")
        stat = p.stat()
        h.update(rel.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()


def _dataset_hash_for_type(model_type: str) -> str:
    if model_type in ("classifier", "dinov2_finetune"):
        return _hash_dataset_dir(_BACKEND_ROOT / "references")
    if model_type == "rfdetr":
        return _hash_dataset_dir(_BACKEND_ROOT.parent / "datasets" / "cigarette_packs")
    return "unknown"


def _hparam_signature(model_type: str, version: str, params: dict) -> str:
    payload = {
        "model_type": model_type,
        "version": version,
        "params": params,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _find_duplicate_completed_run(model_type: str, version: str, dataset_hash: str, hparam_signature: str) -> dict | None:
    rows = _load_model_registry()
    for row in reversed(rows):
        if (
            row.get("model_type") == model_type
            and row.get("version") == version
            and row.get("dataset_hash") == dataset_hash
            and row.get("hparam_signature") == hparam_signature
            and row.get("status") == "done"
        ):
            return row
    return None


def _metrics_from_progress(progress: dict) -> dict:
    if not isinstance(progress, dict):
        return {}
    out = {}
    if "val_acc" in progress:
        out["val_acc"] = progress.get("val_acc")
    if "best_val_acc" in progress:
        out["best_val_acc"] = progress.get("best_val_acc")
    if "train_acc" in progress:
        out["train_acc"] = progress.get("train_acc")
    if "train_loss" in progress:
        out["train_loss"] = progress.get("train_loss")
    if "epoch" in progress:
        out["epoch"] = progress.get("epoch")
    if "total_epochs" in progress:
        out["total_epochs"] = progress.get("total_epochs")
    return out


def _run_training_job(job_id: str, script: str, args: list[str]):
    """Run a training script as a subprocess with progress file polling."""
    import subprocess
    import sys
    import time

    progress_file = _TRAINING_PROGRESS_DIR / f".training_progress_{job_id}.json"
    full_args = [sys.executable, str(_TRAINING_PROGRESS_DIR / script),
                 "--progress-file", str(progress_file)] + args

    try:
        with _training_lock:
            _training_jobs[job_id].update({
                "status": "running",
                "progress": {},
                "error": None,
                "last_update": _now_iso(),
            })
        _update_model_registry(job_id, {"status": "running", "last_update": _now_iso()})

        process = subprocess.Popen(
            full_args, cwd=str(_TRAINING_PROGRESS_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        with _training_lock:
            _training_processes[job_id] = process
            _training_jobs[job_id]["pid"] = process.pid

        # Poll progress file while process runs
        while process.poll() is None:
            time.sleep(2)
            if progress_file.exists():
                try:
                    progress = json.loads(progress_file.read_text())
                    with _training_lock:
                        _training_jobs[job_id]["progress"] = progress
                        _training_jobs[job_id]["last_update"] = _now_iso()
                    # Push to SSE queue if job has one
                    with jobs_lock:
                        job = jobs.get(job_id)
                    if job:
                        epoch = progress.get("epoch", 0)
                        total = progress.get("total_epochs", 1)
                        val_acc = progress.get("val_acc", 0)
                        pct = int((epoch / total) * 100) if total > 0 else 0
                        job["queue"].put((pct, f"Epoch {epoch}/{total} | Val acc: {val_acc:.3f}"))
                    _update_training_history(job_id, {
                        "status": "running",
                        "progress": progress,
                        "last_update": _now_iso(),
                    })
                    _update_model_registry(job_id, {
                        "status": "running",
                        "progress": progress,
                        "last_update": _now_iso(),
                        **_metrics_from_progress(progress),
                    })
                except Exception:
                    pass

        # Final read
        if progress_file.exists():
            try:
                final_progress = json.loads(progress_file.read_text())
                with _training_lock:
                    _training_jobs[job_id]["progress"] = final_progress
                    _training_jobs[job_id]["last_update"] = _now_iso()
                _update_training_history(job_id, {
                    "progress": final_progress,
                    "last_update": _now_iso(),
                })
                _update_model_registry(job_id, {
                    "progress": final_progress,
                    "last_update": _now_iso(),
                    **_metrics_from_progress(final_progress),
                })
            except Exception:
                pass
            progress_file.unlink(missing_ok=True)

        stdout = process.stdout.read() if process.stdout else ""

        if process.returncode == 0:
            with _training_lock:
                _training_jobs[job_id]["status"] = "done"
                _training_jobs[job_id]["end_time"] = _now_iso()
                completed_version = _training_jobs[job_id].get("version", DEFAULT_TRAINING_VERSION)
            with jobs_lock:
                job = jobs.get(job_id)
            if job:
                job["queue"].put((100, "Training complete"))
                jobs[job_id]["status"] = "done"
                jobs[job_id]["result"] = "TRAINING_COMPLETE"
            next_version = _mark_training_completed(str(completed_version))
            _update_training_history(job_id, {
                "status": "done",
                "end_time": _now_iso(),
                "next_version": next_version,
            })
            with _training_lock:
                progress = _training_jobs.get(job_id, {}).get("progress", {})
            _update_model_registry(job_id, {
                "status": "done",
                "end_time": _now_iso(),
                "next_version": next_version,
                **_metrics_from_progress(progress),
            })
        else:
            with _training_lock:
                stop_requested = bool(_training_jobs.get(job_id, {}).get("stop_requested"))
            if stop_requested:
                with _training_lock:
                    _training_jobs[job_id]["status"] = "stopped"
                    _training_jobs[job_id]["error"] = None
                    _training_jobs[job_id]["end_time"] = _now_iso()
                _update_training_history(job_id, {
                    "status": "stopped",
                    "end_time": _now_iso(),
                })
                _update_model_registry(job_id, {
                    "status": "stopped",
                    "end_time": _now_iso(),
                })
                return
            with _training_lock:
                _training_jobs[job_id]["status"] = "error"
                _training_jobs[job_id]["error"] = stdout[-2000:]
                _training_jobs[job_id]["end_time"] = _now_iso()
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = stdout[-2000:]
            _update_training_history(job_id, {
                "status": "error",
                "error": stdout[-2000:],
                "end_time": _now_iso(),
            })
            _update_model_registry(job_id, {
                "status": "error",
                "error": stdout[-2000:],
                "end_time": _now_iso(),
            })

    except Exception:
        err = traceback.format_exc()
        with _training_lock:
            _training_jobs[job_id]["status"] = "error"
            _training_jobs[job_id]["error"] = err
            _training_jobs[job_id]["end_time"] = _now_iso()
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = err
        _update_training_history(job_id, {
            "status": "error",
            "error": err,
            "end_time": _now_iso(),
        })
        _update_model_registry(job_id, {
            "status": "error",
            "error": err,
            "end_time": _now_iso(),
        })
    finally:
        with _training_lock:
            _training_processes.pop(job_id, None)


@app.post("/train-classifier")
def train_classifier_endpoint(
    epochs: int = 100,
    batch_size: int = 64,
    embed_batch_size: int = 8,
    lr: float = 0.001,
    version: str = "",
    force: bool = False,
):
    """Train the brand classifier (frozen DINOv2 + MLP head). Works on CPU."""
    version = (version or _get_current_training_version()).strip() or DEFAULT_TRAINING_VERSION
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "embed_batch_size": embed_batch_size,
        "lr": lr,
    }
    dataset_hash = _dataset_hash_for_type("classifier")
    hp_sig = _hparam_signature("classifier", version, params)
    if not force:
        dup = _find_duplicate_completed_run("classifier", version, dataset_hash, hp_sig)
        if dup:
            return {
                "skipped": True,
                "reason": "Matching completed run found for same dataset+settings",
                "existing_job_id": dup.get("job_id"),
                "version": version,
                "best_val_acc": dup.get("best_val_acc"),
            }

    job_id = create_job()
    start_time = _now_iso()
    args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--embed-batch-size", str(embed_batch_size),
        "--lr", str(lr),
    ]
    with _training_lock:
        _training_jobs[job_id] = {
            "job_id": job_id,
            "type": "classifier",
            "version": version,
            "model_type": "classifier",
            "dataset_hash": dataset_hash,
            "hparam_signature": hp_sig,
            "status": "queued",
            "params": params,
            "progress": {},
            "error": None,
            "start_time": start_time,
            "last_update": start_time,
            "end_time": None,
        }
        snapshot = dict(_training_jobs[job_id])
    _append_training_history(snapshot)
    _append_model_registry(snapshot)
    threading.Thread(
        target=_run_training_job,
        args=(job_id, "brand_classifier.py", args),
        daemon=True,
    ).start()
    return {"job_id": job_id, "type": "classifier", "epochs": epochs, "version": version, "skipped": False}


@app.post("/train-rfdetr")
def train_rfdetr_endpoint(
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 0.0001,
    roboflow_url: str = "",
    clean_dataset: bool = False,
    version: str = "",
    force: bool = False,
):
    """Train RF-DETR detection model. Requires GPU for reasonable speed."""
    version = (version or _get_current_training_version()).strip() or DEFAULT_TRAINING_VERSION
    dataset_import = None
    if roboflow_url.strip():
        dataset_import = download_roboflow_coco(url=roboflow_url.strip(), clean=clean_dataset)

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dataset_source": "roboflow_url" if roboflow_url.strip() else "existing_local",
    }
    dataset_hash = _dataset_hash_for_type("rfdetr")
    hp_sig = _hparam_signature("rfdetr", version, params)
    if not force:
        dup = _find_duplicate_completed_run("rfdetr", version, dataset_hash, hp_sig)
        if dup:
            return {
                "skipped": True,
                "reason": "Matching completed run found for same dataset+settings",
                "existing_job_id": dup.get("job_id"),
                "version": version,
            }

    job_id = create_job()
    start_time = _now_iso()
    args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
    ]
    with _training_lock:
        _training_jobs[job_id] = {
            "job_id": job_id,
            "type": "rfdetr",
            "version": version,
            "model_type": "rfdetr",
            "dataset_hash": dataset_hash,
            "hparam_signature": hp_sig,
            "status": "queued",
            "params": params,
            "dataset_import": dataset_import,
            "progress": {},
            "error": None,
            "start_time": start_time,
            "last_update": start_time,
            "end_time": None,
        }
        snapshot = dict(_training_jobs[job_id])
    _append_training_history(snapshot)
    _append_model_registry(snapshot)
    threading.Thread(
        target=_run_training_job,
        args=(job_id, "train.py", args),
        daemon=True,
    ).start()
    return {
        "job_id": job_id,
        "type": "rfdetr",
        "epochs": epochs,
        "version": version,
        "skipped": False,
        "dataset_import": dataset_import,
    }


@app.post("/finetune-dinov2")
def finetune_dinov2_endpoint(
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 0.00001,
    unfreeze_layers: int = 4,
    version: str = "",
    force: bool = False,
):
    """Fine-tune DINOv2 backbone. Requires 16GB+ VRAM (RunPod recommended)."""
    version = (version or _get_current_training_version()).strip() or DEFAULT_TRAINING_VERSION
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "unfreeze_layers": unfreeze_layers,
    }
    dataset_hash = _dataset_hash_for_type("dinov2_finetune")
    hp_sig = _hparam_signature("dinov2_finetune", version, params)
    if not force:
        dup = _find_duplicate_completed_run("dinov2_finetune", version, dataset_hash, hp_sig)
        if dup:
            return {
                "skipped": True,
                "reason": "Matching completed run found for same dataset+settings",
                "existing_job_id": dup.get("job_id"),
                "version": version,
                "best_val_acc": dup.get("best_val_acc"),
            }

    job_id = create_job()
    start_time = _now_iso()
    args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--unfreeze-layers", str(unfreeze_layers),
    ]
    with _training_lock:
        _training_jobs[job_id] = {
            "job_id": job_id,
            "type": "dinov2_finetune",
            "version": version,
            "model_type": "dinov2_finetune",
            "dataset_hash": dataset_hash,
            "hparam_signature": hp_sig,
            "status": "queued",
            "params": params,
            "progress": {},
            "error": None,
            "start_time": start_time,
            "last_update": start_time,
            "end_time": None,
        }
        snapshot = dict(_training_jobs[job_id])
    _append_training_history(snapshot)
    _append_model_registry(snapshot)
    threading.Thread(
        target=_run_training_job,
        args=(job_id, "finetune_dinov2.py", args),
        daemon=True,
    ).start()
    return {"job_id": job_id, "type": "dinov2_finetune", "epochs": epochs, "version": version, "skipped": False}


@app.get("/training-status/{job_id}")
def training_status(job_id: str):
    """Get training progress for a running job."""
    with _training_lock:
        if job_id in _training_jobs:
            return _training_jobs[job_id]
    raise HTTPException(status_code=404, detail="Training job not found")


@app.post("/training-stop/{job_id}")
def training_stop(job_id: str):
    """Stop a running training job by terminating its subprocess."""
    with _training_lock:
        job = _training_jobs.get(job_id)
        process = _training_processes.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        status = job.get("status", "")
        if status not in ("queued", "running", "stopping"):
            return {"job_id": job_id, "status": status, "message": "Job is not running."}
        job["stop_requested"] = True
        job["status"] = "stopping"
        job["last_update"] = _now_iso()

    if process is None:
        with _training_lock:
            _training_jobs[job_id]["status"] = "stopped"
            _training_jobs[job_id]["end_time"] = _now_iso()
        _update_training_history(job_id, {"status": "stopped", "end_time": _now_iso()})
        _update_model_registry(job_id, {"status": "stopped", "end_time": _now_iso()})
        return {"job_id": job_id, "status": "stopped", "message": "Queued job cancelled."}

    try:
        process.terminate()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to terminate process: {exc}")

    return {"job_id": job_id, "status": "stopping", "message": "Termination requested."}


@app.get("/training-progress/{job_id}")
def training_progress_stream(job_id: str):
    """SSE stream for live training progress updates."""
    import time

    def event_stream():
        last_payload = None
        while True:
            with _training_lock:
                state = _training_jobs.get(job_id)
                payload = json.dumps(state or {"status": "not_found"}, ensure_ascii=False)

            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload

            if not state or state.get("status") in ("done", "error"):
                break

            time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/training-history")
def training_history(limit: int = 30):
    """Return recent training jobs (latest first)."""
    history = _load_training_history()
    history = list(reversed(history))
    return {"count": len(history), "items": history[: max(1, min(limit, 200))]}


@app.get("/model-registry")
def model_registry(limit: int = 100):
    rows = _load_model_registry()
    rows = list(reversed(rows))
    version_state = _load_version_state()
    return {
        "count": len(rows),
        "items": rows[: max(1, min(limit, 500))],
        "current_version": version_state.get("current_version", DEFAULT_TRAINING_VERSION),
        "last_trained_version": version_state.get("last_trained_version"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": get_device()}
