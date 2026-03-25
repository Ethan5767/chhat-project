import json
import io
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import faiss
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from rapidfuzz import fuzz
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO

BATCH_MODE: Optional[int] = None
SAVE_INTERVAL = 50
REFERENCES_DIR = Path("./references")
INDEX_DIR = Path("./faiss_index")
INDEX_BIN = INDEX_DIR / "index.bin"
LABELS_JSON = INDEX_DIR / "labels.json"
DINO_MODEL_ID = "facebook/dinov2-small"
_BRAND_ROOT = Path(__file__).resolve().parent.parent
_YOLO_FT = _BRAND_ROOT / "runs" / "yolo" / "cigarette_pack_v2_roboflow" / "weights" / "best.pt"
# Previous run name (optional fallback)
_YOLO_FT_PREV = _BRAND_ROOT / "runs" / "yolo" / "cigarette_rf_ft" / "weights" / "best.pt"
_YOLO_FT_LEGACY = Path(r"C:\Users\kimto\runs\detect\runs\cigarette_rf_ft\weights\best.pt")
if _YOLO_FT.is_file():
    YOLO_MODEL_ID = str(_YOLO_FT)
elif _YOLO_FT_PREV.is_file():
    YOLO_MODEL_ID = str(_YOLO_FT_PREV)
elif _YOLO_FT_LEGACY.is_file():
    YOLO_MODEL_ID = str(_YOLO_FT_LEGACY)
else:
    YOLO_MODEL_ID = "yolo11n.pt"
DOWNLOAD_TIMEOUT = 15
YOLO_CONF_THRESHOLD = 0.25
URL_COLUMN_PREFIXES = ("Q", )  # matches any Q-column; actual filtering is done by URL content
OCR_ENABLED = True
OCR_WEIGHT = 0.60
OCR_MIN_TOKEN_LEN = 3
MIN_OUTPUT_CONFIDENCE = 0.70
FAISS_TOP_K = 5
OCR_BRAND_MATCH_BOOST = 0.20
OCR_VARIANT_MATCH_BOOST = 0.08

_dino_processor = None
_dino_model = None
_yolo_model = None
_ocr_reader = None

logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dino(device: str):
    global _dino_processor, _dino_model
    if _dino_processor is None or _dino_model is None:
        _dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModel.from_pretrained(DINO_MODEL_ID)
        _dino_model.eval()
    _dino_model.to(device)
    return _dino_processor, _dino_model


def load_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_ID)
    return _yolo_model


def load_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _ocr_reader


def embed_image(pil_img: Image.Image, processor, model, device: str) -> np.ndarray:
    img = pil_img.convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        vec = cls_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_images_batch(pil_imgs: list[Image.Image], processor, model, device: str) -> np.ndarray:
    """Embed multiple images in a single forward pass. Returns (N, dim) L2-normalised."""
    if not pil_imgs:
        return np.empty((0, 384), dtype=np.float32)
    imgs = [img.convert("RGB") for img in pil_imgs]
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        vecs = cls_embeddings.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = vecs / norms
    return vecs


def build_index(device: str, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(REFERENCES_DIR.glob(ext))
        image_paths.extend(REFERENCES_DIR.glob(ext.upper()))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise FileNotFoundError(
            f"No reference images found in {REFERENCES_DIR.resolve()}. "
            "Add files like Marlboro.jpg, Camel.png first."
        )

    processor, model = load_dino(device)
    embeddings = []
    labels = []
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img:
            vec = embed_image(img, processor, model, device)
        embeddings.append(vec)
        labels.append(image_path.stem)

        if progress_cb:
            progress_cb(idx, total, image_path.name)

    matrix = np.vstack(embeddings).astype(np.float32)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_BIN))
    with LABELS_JSON.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def load_index():
    if not INDEX_BIN.exists() or not LABELS_JSON.exists():
        raise FileNotFoundError(
            f"Missing index files. Expected {INDEX_BIN.resolve()} and {LABELS_JSON.resolve()}. "
            "Run /build-index first."
        )
    index = faiss.read_index(str(INDEX_BIN))
    with LABELS_JSON.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return index, labels


def distance_to_confidence(dist: float) -> float:
    return float(math.exp(-float(dist)))


def download_image(url: str) -> Optional[Image.Image]:
    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img
    except Exception as exc:
        logger.warning("Failed to download image from %s: %s", url, exc)
        return None


def _looks_like_url(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return False
    return text.startswith("http://") or text.startswith("https://")


def get_url_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect columns that contain image URLs by scanning the first rows."""
    matches = []
    sample = df.head(min(20, len(df)))
    for col in list(df.columns):
        col_str = str(col).strip()
        if "photo" in col_str.lower() or "link" in col_str.lower() or "image" in col_str.lower() or "url" in col_str.lower():
            matches.append(col)
            continue
        url_count = sample[col].apply(_looks_like_url).sum()
        if url_count >= max(1, len(sample) * 0.3):
            matches.append(col)
    return matches


def _valid_url_value(value) -> bool:
    return _looks_like_url(value)


def _normalize_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_label_profiles(labels: list[str]) -> list[dict]:
    profiles = []
    for label in labels:
        norm_label = _normalize_text(label)
        base_label = re.sub(r"\s+\d+$", "", norm_label).strip()
        parts = [p for p in base_label.split() if p]
        brand = parts[0] if parts else base_label
        variant = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        tokens = {t for t in parts if len(t) >= OCR_MIN_TOKEN_LEN}
        profiles.append(
            {
                "label": label,
                "norm": norm_label,
                "base": base_label,
                "brand": brand,
                "variant": variant,
                "tokens": tokens,
            }
        )
    return profiles


def _run_ocr_on_image(image: Image.Image) -> list:
    """Run EasyOCR once on an image and return the raw result items."""
    try:
        reader = load_ocr()
    except Exception as exc:
        logger.warning("OCR disabled because EasyOCR failed to load: %s", exc)
        return []
    try:
        np_img = np.array(image.convert("RGB"))
        return reader.readtext(np_img, detail=1, paragraph=False)
    except Exception as exc:
        logger.warning("OCR read failed: %s", exc)
        return []


def _ocr_brand_scores_from_items(ocr_items: list, label_profiles: list[dict]) -> dict[str, float]:
    brand_scores: dict[str, float] = {}
    for item in ocr_items:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        text_conf = float(item[2])
        norm_text = _normalize_text(text)
        if not norm_text:
            continue

        text_tokens = [t for t in norm_text.split() if len(t) >= OCR_MIN_TOKEN_LEN]
        if not text_tokens:
            continue

        for prof in label_profiles:
            score = 0.0

            token_overlap = 0
            for tok in text_tokens:
                if tok in prof["tokens"]:
                    token_overlap += 1
            if token_overlap > 0:
                score = max(score, min(1.0, text_conf * (0.55 + 0.15 * token_overlap)))

            fuzzy_base = fuzz.partial_ratio(norm_text, prof["base"]) / 100.0 if prof["base"] else 0.0
            fuzzy_full = fuzz.partial_ratio(norm_text, prof["norm"]) / 100.0 if prof["norm"] else 0.0
            fuzzy = max(fuzzy_base, fuzzy_full)
            score = max(score, text_conf * fuzzy)

            if score > brand_scores.get(prof["label"], 0.0):
                brand_scores[prof["label"]] = float(score)

    return brand_scores


def _ocr_family_hints_from_items(ocr_items: list, label_profiles: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    brand_hints: dict[str, float] = {}
    variant_hints: dict[str, float] = {}
    for item in ocr_items:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        text_conf = float(item[2])
        norm_text = _normalize_text(text)
        if not norm_text:
            continue

        for prof in label_profiles:
            brand = prof.get("brand", "")
            if brand:
                brand_fuzzy = fuzz.partial_ratio(norm_text, brand) / 100.0
                if brand_fuzzy >= 0.70:
                    score = text_conf * brand_fuzzy
                    if score > brand_hints.get(brand, 0.0):
                        brand_hints[brand] = float(score)

            variant = prof.get("variant", "")
            if variant:
                variant_fuzzy = fuzz.partial_ratio(norm_text, variant) / 100.0
                if variant_fuzzy >= 0.75:
                    score = text_conf * variant_fuzzy
                    if score > variant_hints.get(variant, 0.0):
                        variant_hints[variant] = float(score)

    return brand_hints, variant_hints


def _detect_brands_from_image(
    image: Image.Image,
    yolo_model,
    processor,
    model,
    device: str,
    index,
    labels: list[str],
) -> dict[str, float]:
    label_profiles = _build_label_profiles(labels)
    label_profile_map = {p["label"]: p for p in label_profiles}
    results = yolo_model.predict(image, conf=YOLO_CONF_THRESHOLD, verbose=False)
    crops: list[Image.Image] = []

    boxes = None
    if results and len(results) > 0:
        boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        width, height = image.size
        xyxy = boxes.xyxy.detach().cpu().numpy()
        for box in xyxy:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(1, min(x2, width))
            y2 = max(1, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
    else:
        crops.append(image)

    # OCR: run ONCE per crop, reuse items for both hints and brand scores
    crop_ocr_items: list[list] = []
    if OCR_ENABLED:
        for crop in crops:
            crop_ocr_items.append(_run_ocr_on_image(crop))
    else:
        crop_ocr_items = [[] for _ in crops]

    # DINO: batch-embed all crops in a single GPU forward pass
    all_vecs = embed_images_batch(crops, processor, model, device)

    # FAISS search + OCR family-hint fusion per crop
    dino_best: dict[str, float] = {}
    k = max(1, min(FAISS_TOP_K, len(labels)))
    for crop_idx in range(len(crops)):
        ocr_items = crop_ocr_items[crop_idx]
        if OCR_ENABLED:
            brand_hints, variant_hints = _ocr_family_hints_from_items(ocr_items, label_profiles)
        else:
            brand_hints, variant_hints = {}, {}

        vec = all_vecs[crop_idx].reshape(1, -1)
        distances, indices = index.search(vec, k=k)
        for rank in range(k):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(labels):
                continue
            label = labels[idx]
            dist = float(distances[0][rank])
            conf = distance_to_confidence(dist)

            prof = label_profile_map.get(label, {})
            family_boost = brand_hints.get(prof.get("brand", ""), 0.0) * OCR_BRAND_MATCH_BOOST
            variant_boost = variant_hints.get(prof.get("variant", ""), 0.0) * OCR_VARIANT_MATCH_BOOST
            fused_rank_conf = min(1.0, conf + family_boost + variant_boost)

            if fused_rank_conf > dino_best.get(label, 0.0):
                dino_best[label] = fused_rank_conf

    if not OCR_ENABLED:
        return dino_best

    # OCR brand scores from the SAME pre-computed items (no second OCR call)
    ocr_best: dict[str, float] = {}
    for ocr_items in crop_ocr_items:
        crop_ocr_scores = _ocr_brand_scores_from_items(ocr_items, label_profiles)
        for brand, conf in crop_ocr_scores.items():
            if conf > ocr_best.get(brand, 0.0):
                ocr_best[brand] = conf

    fused: dict[str, float] = {}
    all_labels = set(dino_best.keys()) | set(ocr_best.keys())
    for brand in all_labels:
        dino_conf = dino_best.get(brand, 0.0)
        ocr_conf = ocr_best.get(brand, 0.0)
        fused_conf = 1.0 - ((1.0 - dino_conf) * (1.0 - (OCR_WEIGHT * ocr_conf)))
        fused[brand] = float(min(1.0, max(dino_conf, fused_conf)))

    return fused


def _format_brand_scores(brand_best: dict[str, float], min_confidence: float = 0.0) -> tuple[list[str], list[float]]:
    filtered = {k: v for k, v in brand_best.items() if float(v) >= float(min_confidence)}
    if not filtered:
        return ["NO_DETECTION"], [0.0]
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    detected_brands = [b for b, _ in sorted_items]
    confidence_scores = [round(float(c), 3) for _, c in sorted_items]
    return detected_brands, confidence_scores


def _compute_row_union_brands(col_brand_maps: list[dict[str, float]]) -> tuple[str, str]:
    """
    OR across all image columns: union every label that meets MIN_OUTPUT_CONFIDENCE.
    If the same label appears in multiple columns, keep the maximum confidence.
    Returns JSON array strings for final_brand and overall_confidence (aligned lists).
    """
    if not col_brand_maps:
        return json.dumps(["NO_DETECTION"], ensure_ascii=False), json.dumps([0.0], ensure_ascii=False)

    merged: dict[str, float] = {}
    for m in col_brand_maps:
        for label, conf in m.items():
            if float(conf) >= MIN_OUTPUT_CONFIDENCE:
                merged[label] = max(merged.get(label, 0.0), float(conf))

    if not merged:
        return json.dumps(["NO_DETECTION"], ensure_ascii=False), json.dumps([0.0], ensure_ascii=False)

    sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    brands = [b for b, _ in sorted_items]
    confs = [round(float(c), 3) for _, c in sorted_items]
    return json.dumps(brands, ensure_ascii=False), json.dumps(confs, ensure_ascii=False)


def _download_row_images(
    row, url_columns: list[str],
) -> dict[str, Optional[Image.Image]]:
    """Download all URL-column images for a single row concurrently."""
    results: dict[str, Optional[Image.Image]] = {}
    urls_to_download: dict[str, str] = {}

    for col in url_columns:
        col_name = str(col)
        value = row[col]
        if _valid_url_value(value):
            urls_to_download[col_name] = str(value).strip()
        else:
            results[col_name] = None

    if not urls_to_download:
        return results

    with ThreadPoolExecutor(max_workers=min(8, len(urls_to_download))) as executor:
        future_to_col = {
            executor.submit(download_image, url): col_name
            for col_name, url in urls_to_download.items()
        }
        for future in as_completed(future_to_col):
            col_name = future_to_col[future]
            try:
                results[col_name] = future.result()
            except Exception:
                results[col_name] = None

    return results


def run_pipeline(csv_path, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Path:
    device = get_device()
    index, labels = load_index()
    processor, model = load_dino(device)
    yolo_model = load_yolo()

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    original_len = len(df)
    url_columns = get_url_columns(df)
    if not url_columns:
        raise ValueError(
            "No URL columns found. Columns should contain 'photo', 'link', 'image', or 'url' in the header, "
            "or have at least 30% of rows containing http(s) URLs."
        )

    process_len = original_len if BATCH_MODE is None else min(BATCH_MODE, original_len)
    out_path = csv_path.parent / f"{csv_path.stem}_results.csv"
    id_col = df.columns[0]

    # Resume: load existing partial results and skip already-processed rows
    start_row = 0
    completed_rows: list[dict] = []
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            start_row = len(existing_df)
            completed_rows = existing_df.to_dict("records")
            logger.info("Resuming from row %d (%d rows already done)", start_row, start_row)
        except Exception:
            start_row = 0
            completed_rows = []

    def _build_result_row(row_data, col_brand_maps, per_col_results):
        final_brand, overall_conf = _compute_row_union_brands(col_brand_maps)
        result = {id_col: row_data[id_col], "final_brand": final_brand, "overall_confidence": overall_conf}
        for col in url_columns:
            cn = str(col)
            result[cn] = row_data[col]
            det, scr = per_col_results.get(cn, (["NO_DETECTION"], [0.0]))
            result[f"{cn}_detected_brands"] = json.dumps(det, ensure_ascii=False)
            result[f"{cn}_confidence"] = json.dumps(scr, ensure_ascii=False)
        return result

    def _error_result_row(row_data):
        result = {
            id_col: row_data[id_col],
            "final_brand": json.dumps(["ERROR"], ensure_ascii=False),
            "overall_confidence": json.dumps([0.0], ensure_ascii=False),
        }
        for col in url_columns:
            cn = str(col)
            result[cn] = row_data[col]
            result[f"{cn}_detected_brands"] = json.dumps(["ERROR"], ensure_ascii=False)
            result[f"{cn}_confidence"] = json.dumps([0.0], ensure_ascii=False)
        return result

    def _flush(rows):
        pd.DataFrame(rows).to_csv(out_path, index=False)

    for row_idx in range(start_row, process_len):
        row = df.iloc[row_idx]
        try:
            images = _download_row_images(row, url_columns)
            col_brand_maps: list[dict[str, float]] = []
            per_col_results: dict[str, tuple[list[str], list[float]]] = {}

            for col in url_columns:
                col_name = str(col)
                image = images.get(col_name)
                if image is None:
                    per_col_results[col_name] = (["NO_DETECTION"], [0.0])
                    continue

                col_best = _detect_brands_from_image(
                    image=image,
                    yolo_model=yolo_model,
                    processor=processor,
                    model=model,
                    device=device,
                    index=index,
                    labels=labels,
                )
                col_detected, col_scores = _format_brand_scores(col_best, min_confidence=MIN_OUTPUT_CONFIDENCE)
                per_col_results[col_name] = (col_detected, col_scores)
                col_brand_maps.append(col_best)

            result_row = _build_result_row(row, col_brand_maps, per_col_results)
        except Exception as exc:
            logger.error("Row %d failed: %s", row_idx, exc, exc_info=True)
            result_row = _error_result_row(row)

        completed_rows.append(result_row)

        # Incremental save every SAVE_INTERVAL rows or on the last row
        if (row_idx + 1) % SAVE_INTERVAL == 0 or row_idx + 1 == process_len:
            _flush(completed_rows)

        if progress_cb:
            progress_cb(row_idx + 1, process_len, f"Processed row {row_idx + 1}/{process_len}")

    # Append NOT_PROCESSED markers for rows beyond process_len (BATCH_MODE)
    remaining = original_len - process_len
    if remaining > 0:
        for ri in range(process_len, original_len):
            row = df.iloc[ri]
            result = {
                id_col: row[id_col],
                "final_brand": json.dumps(["NOT_PROCESSED"], ensure_ascii=False),
                "overall_confidence": json.dumps([], ensure_ascii=False),
            }
            for col in url_columns:
                cn = str(col)
                result[cn] = row[col]
                result[f"{cn}_detected_brands"] = json.dumps(["NOT_PROCESSED"], ensure_ascii=False)
                result[f"{cn}_confidence"] = json.dumps([], ensure_ascii=False)
            completed_rows.append(result)
        _flush(completed_rows)

    return out_path
