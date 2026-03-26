import json
import io
import logging
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

BATCH_MODE: Optional[int] = None
SAVE_INTERVAL = 50
_BACKEND_ROOT = Path(__file__).resolve().parent
REFERENCES_DIR = _BACKEND_ROOT / "references"
INDEX_DIR = _BACKEND_ROOT / "faiss_index"
INDEX_BIN = INDEX_DIR / "index.bin"
LABELS_JSON = INDEX_DIR / "labels.json"
DINO_MODEL_ID = "facebook/dinov2-base"
_PROJECT_ROOT = _BACKEND_ROOT.parent
_RFDETR_CHECKPOINT_DIR = _PROJECT_ROOT / "runs"

DOWNLOAD_TIMEOUT = 15
RFDETR_CONF_THRESHOLD = 0.25
URL_COLUMN_PREFIXES = ("Q", )  # matches any Q-column; actual filtering is done by URL content
OCR_ENABLED = True
OCR_MIN_TOKEN_LEN = 3
MIN_OUTPUT_CONFIDENCE = 0.70
FAISS_TOP_K = 5
OCR_FULLIMG_ENABLED = True             # run OCR on full image as fallback

# Brand-consensus fusion: both OCR and DINO must agree on the brand family
DINO_PRODUCT_WEIGHT = 0.60            # DINO weight for product ranking when brand is confirmed
OCR_BRAND_WEIGHT = 0.40               # OCR weight when both agree on brand
NO_CONSENSUS_PENALTY = 0.70           # multiplier when DINO finds brand but OCR doesn't confirm
OCR_BRAND_CONFIRM_THRESHOLD = 0.30    # min OCR brand-family score to count as "brand confirmed"
OCR_INDEPENDENT_WEIGHT = 0.55         # weight for OCR-only detections (strong text match can reach ~0.75)
OCR_INDEPENDENT_MIN_SCORE = 0.65      # OCR brand score must be >= this to introduce a brand
MULTI_REF_BOOST_PER_HIT = 0.015      # confidence boost per additional matching reference photo

_dino_processor = None
_dino_model = None
_rfdetr_model = None
_ocr_reader = None


def label_to_product(label: str) -> str:
    """Strip the trailing photo number to get the product name.

    'esse_change_1'  -> 'esse_change'
    'malboro_red_12' -> 'malboro_red'
    'other_4_24'     -> 'other_4'
    '555_original_6' -> '555_original'
    """
    parts = label.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label


def _aggregate_to_products(label_scores: dict[str, float]) -> dict[str, float]:
    """Aggregate per-reference-photo scores to per-product scores.

    For each product, keeps the best confidence and boosts it based on
    how many distinct reference photos matched (more photos = more evidence).
    """
    product_best: dict[str, float] = {}
    product_hits: dict[str, int] = {}

    for label, conf in label_scores.items():
        product = label_to_product(label)
        if conf > product_best.get(product, 0.0):
            product_best[product] = conf
        product_hits[product] = product_hits.get(product, 0) + 1

    # Boost confidence when multiple reference photos match the same product
    for product in product_best:
        hits = product_hits[product]
        if hits >= 2:
            boost = min(1.15, 1.0 + MULTI_REF_BOOST_PER_HIT * (hits - 1))
            product_best[product] = min(1.0, product_best[product] * boost)

    return product_best

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


def _find_best_checkpoint() -> Optional[Path]:
    if not _RFDETR_CHECKPOINT_DIR.exists():
        return None
    candidates = list(_RFDETR_CHECKPOINT_DIR.rglob("best*.pth"))
    if not candidates:
        candidates = list(_RFDETR_CHECKPOINT_DIR.rglob("*.pth"))
    if not candidates:
        candidates = list(_RFDETR_CHECKPOINT_DIR.rglob("best*.pt"))
    if not candidates:
        candidates = list(_RFDETR_CHECKPOINT_DIR.rglob("*.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def load_rfdetr():
    global _rfdetr_model
    if _rfdetr_model is not None:
        return _rfdetr_model
    from rfdetr import RFDETRMedium
    checkpoint = _find_best_checkpoint()
    if checkpoint:
        logger.info("Loading fine-tuned RF-DETR from %s", checkpoint)
        _rfdetr_model = RFDETRMedium(pretrain_weights=str(checkpoint))
    else:
        logger.info("No fine-tuned checkpoint found, using pre-trained RF-DETR-M")
        _rfdetr_model = RFDETRMedium()
    try:
        _rfdetr_model.optimize_for_inference()
        logger.info("RF-DETR optimized for inference")
    except Exception as exc:
        logger.warning("Could not optimize RF-DETR for inference: %s", exc)
    return _rfdetr_model


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
        return np.empty((0, 768), dtype=np.float32)
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
    index = faiss.IndexFlatIP(matrix.shape[1])
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
    """With IndexFlatIP on L2-normalised vectors, 'dist' is cosine similarity [0,1]."""
    return float(max(0.0, min(1.0, dist)))


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



def _detect_brands_from_image(
    image: Image.Image,
    rfdetr_model,
    processor,
    model,
    device: str,
    index,
    labels: list[str],
) -> dict[str, float]:
    label_profiles = _build_label_profiles(labels)
    label_profile_map = {p["label"]: p for p in label_profiles}
    detections = rfdetr_model.predict(image, threshold=RFDETR_CONF_THRESHOLD)
    crops: list[Image.Image] = []

    has_detections = len(detections) > 0 if detections is not None else False

    if has_detections:
        width, height = image.size
        xyxy = detections.xyxy
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

    num_boxes = len(detections) if has_detections else 0

    # OCR: run ONCE per crop, reuse items for both hints and brand scores
    crop_ocr_items: list[list] = []
    if OCR_ENABLED:
        for crop in crops:
            crop_ocr_items.append(_run_ocr_on_image(crop))
    else:
        crop_ocr_items = [[] for _ in crops]

    # Full-image OCR: run on the entire image to catch text the detector missed
    fullimg_ocr_items: list = []
    if OCR_ENABLED and OCR_FULLIMG_ENABLED and has_detections:
        fullimg_ocr_items = _run_ocr_on_image(image)
    elif OCR_ENABLED and not has_detections:
        fullimg_ocr_items = crop_ocr_items[0] if crop_ocr_items else []

    # DINO: batch-embed all crops in a single GPU forward pass
    all_vecs = embed_images_batch(crops, processor, model, device)

    # FAISS search per crop -- collect raw DINO scores with multi-crop aggregation
    dino_best: dict[str, float] = {}
    k = max(1, min(FAISS_TOP_K, len(labels)))

    # Track how many crops vote for each brand family (top-1 match per crop)
    brand_family_votes: dict[str, int] = {}

    for crop_idx in range(len(crops)):
        vec = all_vecs[crop_idx].reshape(1, -1)
        distances, indices = index.search(vec, k=k)

        # Count vote for top-1 brand family of this crop
        top_idx = int(indices[0][0])
        if 0 <= top_idx < len(labels):
            top_label = labels[top_idx]
            top_prof = label_profile_map.get(top_label, {})
            top_family = top_prof.get("brand", "")
            if top_family:
                brand_family_votes[top_family] = brand_family_votes.get(top_family, 0) + 1

        for rank in range(k):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(labels):
                continue
            label = labels[idx]
            dist = float(distances[0][rank])
            conf = distance_to_confidence(dist)
            if conf > dino_best.get(label, 0.0):
                dino_best[label] = conf

    # Multi-crop boost: if multiple crops agree on a brand family, boost scores
    # This reflects that seeing a brand across multiple shelf positions is strong evidence
    if has_detections and num_boxes >= 3:
        for label in list(dino_best.keys()):
            prof = label_profile_map.get(label, {})
            family = prof.get("brand", "")
            votes = brand_family_votes.get(family, 0)
            if votes >= 2:
                boost = min(1.15, 1.0 + 0.05 * (votes - 1))
                dino_best[label] = min(1.0, dino_best[label] * boost)

    if not OCR_ENABLED:
        return _aggregate_to_products(dino_best)

    # OCR brand scores from crop-level OCR items
    ocr_best: dict[str, float] = {}
    for ocr_items in crop_ocr_items:
        crop_ocr_scores = _ocr_brand_scores_from_items(ocr_items, label_profiles)
        for brand, conf in crop_ocr_scores.items():
            if conf > ocr_best.get(brand, 0.0):
                ocr_best[brand] = conf

    # Full-image OCR brand scores (catches text the detector missed)
    if fullimg_ocr_items:
        fullimg_ocr_scores = _ocr_brand_scores_from_items(fullimg_ocr_items, label_profiles)
        for brand, conf in fullimg_ocr_scores.items():
            if conf > ocr_best.get(brand, 0.0):
                ocr_best[brand] = conf

    # -- Brand-consensus fusion --
    # Both OCR and DINO must agree on the brand family for full confidence.
    # DINO gets higher weight for product-level (variant) ranking within
    # a confirmed brand; OCR gets meaningful weight for brand confirmation.
    fused: dict[str, float] = {}

    # Build a map: brand_family -> best OCR score across all labels in that family
    ocr_family_best: dict[str, float] = {}
    for label, ocr_conf in ocr_best.items():
        prof = label_profile_map.get(label, {})
        family = prof.get("brand", "")
        if family:
            ocr_family_best[family] = max(ocr_family_best.get(family, 0.0), ocr_conf)

    for label, dino_conf in dino_best.items():
        prof = label_profile_map.get(label, {})
        brand_family = prof.get("brand", "")

        # Check if OCR confirms this brand family
        ocr_family_score = ocr_family_best.get(brand_family, 0.0) if brand_family else 0.0
        ocr_label_score = ocr_best.get(label, 0.0)

        if ocr_family_score >= OCR_BRAND_CONFIRM_THRESHOLD:
            # Brand confirmed by both OCR and DINO -- weighted fusion
            # DINO gets higher weight for product specificity (which variant)
            # OCR confirms the brand family is correct
            fused_conf = DINO_PRODUCT_WEIGHT * dino_conf + OCR_BRAND_WEIGHT * ocr_label_score
            # If OCR matched the family but not this exact label, still give partial credit
            if ocr_label_score < OCR_BRAND_CONFIRM_THRESHOLD and ocr_family_score >= OCR_BRAND_CONFIRM_THRESHOLD:
                fused_conf = DINO_PRODUCT_WEIGHT * dino_conf + OCR_BRAND_WEIGHT * ocr_family_score * 0.5
            fused[label] = float(min(1.0, fused_conf))
        else:
            # DINO found brand but OCR didn't confirm -- penalize
            fused[label] = float(min(1.0, dino_conf * NO_CONSENSUS_PENALTY))

    # OCR independent detections: introduce brands DINO did NOT find
    if num_boxes == 0:
        independent_scale = 1.5
    elif num_boxes <= 2:
        independent_scale = 1.2
    else:
        independent_scale = 1.0
    for label, ocr_conf in ocr_best.items():
        if label in fused:
            continue
        prof = label_profile_map.get(label, {})
        brand_family = prof.get("brand", "")
        # Check if DINO found any label in this brand family
        dino_has_family = any(
            label_profile_map.get(d_label, {}).get("brand", "") == brand_family
            for d_label in dino_best
        ) if brand_family else False

        if dino_has_family:
            # DINO found same family but different product -- OCR variant with DINO family boost
            dino_family_conf = max(
                (d_conf for d_label, d_conf in dino_best.items()
                 if label_profile_map.get(d_label, {}).get("brand", "") == brand_family),
                default=0.0,
            )
            fused_conf = OCR_BRAND_WEIGHT * ocr_conf + DINO_PRODUCT_WEIGHT * dino_family_conf * 0.4
            fused[label] = float(min(1.0, fused_conf))
        elif ocr_conf >= OCR_INDEPENDENT_MIN_SCORE:
            # Pure OCR detection, no DINO support at all
            indep_conf = min(1.0, ocr_conf * OCR_INDEPENDENT_WEIGHT * independent_scale)
            fused[label] = float(indep_conf)

    # Aggregate to product level: esse_change_1 + esse_change_2 -> esse_change
    # Multiple matching reference photos boost confidence
    return _aggregate_to_products(fused)


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
    rfdetr_model = load_rfdetr()

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
                    rfdetr_model=rfdetr_model,
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
