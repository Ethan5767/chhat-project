import json
import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from rapidfuzz import fuzz
from transformers import AutoImageProcessor, AutoModel

BATCH_MODE: Optional[int] = None
SAVE_INTERVAL = 50
_BACKEND_ROOT = Path(__file__).resolve().parent
# Persistent data root -- survives code deploys. Defaults to backend/ for local dev.
_DATA_ROOT = Path(os.environ.get("CHHAT_DATA_ROOT", str(_BACKEND_ROOT)))
REFERENCES_DIR = _DATA_ROOT / "references"
CLASSIFIER_BASE_DIR = _DATA_ROOT / "classifier_model"
PACKAGING_TYPES = ("pack", "box")

# Per-type paths (backward compat aliases pointing to pack)
CLASSIFIER_DIR = CLASSIFIER_BASE_DIR / "pack"
CLASSIFIER_WEIGHTS = CLASSIFIER_DIR / "best_classifier.pth"
CLASS_MAPPING_JSON = CLASSIFIER_DIR / "class_mapping.json"
# Legacy flat layout (pre pack/box split): startup and endpoints use these paths
if not (CLASSIFIER_WEIGHTS.exists() and CLASS_MAPPING_JSON.exists()):
    _legacy_w = CLASSIFIER_BASE_DIR / "best_classifier.pth"
    _legacy_m = CLASSIFIER_BASE_DIR / "class_mapping.json"
    if _legacy_w.exists() and _legacy_m.exists():
        CLASSIFIER_WEIGHTS = _legacy_w
        CLASS_MAPPING_JSON = _legacy_m
DINO_MODEL_ID = "facebook/dinov2-base"
# Written by finetune_dinov2.py; backbone weights only (dino.*), shared across pack/box brand classifiers.
DINO_FINETUNED_FULL_PATH = CLASSIFIER_BASE_DIR / "dinov2_finetuned_full.pth"
_PROJECT_ROOT = _BACKEND_ROOT.parent
_RFDETR_CHECKPOINT_DIR = _DATA_ROOT / "runs" if (_DATA_ROOT / "runs").exists() else _PROJECT_ROOT / "runs"

DOWNLOAD_TIMEOUT = 15
RFDETR_CONF_THRESHOLD = 0.15  # low threshold to catch packs in small shelf images
OCR_ENABLED = True
OCR_MIN_TOKEN_LEN = 3
MIN_OUTPUT_CONFIDENCE = 0.40  # classifier softmax over 43 classes produces lower per-class probabilities
CLASSIFIER_TOP_K = 5
OCR_FULLIMG_ENABLED = True

# Simplified 3-tier OCR fusion:
# Tier 1: Classifier confident (>= 0.85) -> trust classifier
# Tier 2: Classifier moderate (>= 0.50) -> check OCR for confirmation
# Tier 3: Classifier lost (< 0.50) -> OCR can override
CLASSIFIER_HIGH_CONF = 0.85
CLASSIFIER_LOW_CONF = 0.50
OCR_FALLBACK_THRESHOLD = 0.72
OCR_FALLBACK_MARGIN = 0.08
OCR_STRONG_THRESHOLD = 0.60
OCR_BOOST = 0.05              # confidence boost when OCR agrees with classifier
OCR_INDEPENDENT_MIN_SCORE = 0.70

_dino_processor = None
_dino_model = None
_rfdetr_model = None
_ocr_reader = None
# Per-type classifiers: {"pack": (classifier, mapping), "box": (classifier, mapping)}
_classifiers: dict[str, tuple] = {}
# Legacy single-classifier aliases (for backward compat)
_brand_classifier = None
_class_mapping = None


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


class BrandClassifier(nn.Module):
    """Linear classifier head on DINOv2 embeddings. Must match brand_classifier.py."""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
            boost = min(1.15, 1.0 + 0.015 * (hits - 1))
            product_best[product] = min(1.0, product_best[product] * boost)

    return product_best

logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_finetuned_dino_backbone(model: torch.nn.Module) -> bool:
    """Load ViT backbone from finetune_dinov2.py full checkpoint (keys prefixed with 'dino.')."""
    path = DINO_FINETUNED_FULL_PATH
    if not path.is_file():
        return False
    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    except Exception as exc:
        logger.warning("Could not read finetuned DINO checkpoint %s: %s — using base weights", path, exc)
        return False
    prefix = "dino."
    dino_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not dino_state:
        logger.warning("Finetuned checkpoint %s has no %s* keys — using base weights", path, prefix)
        return False
    incomp = model.load_state_dict(dino_state, strict=False)
    logger.info(
        "Loaded DINOv2 backbone from %s (missing_keys=%d unexpected_keys=%d)",
        path,
        len(incomp.missing_keys),
        len(incomp.unexpected_keys),
    )
    return True


def load_dino(device: str):
    global _dino_processor, _dino_model
    if _dino_processor is None or _dino_model is None:
        _dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModel.from_pretrained(DINO_MODEL_ID)
        _dino_model.eval()
        if not _apply_finetuned_dino_backbone(_dino_model):
            logger.info("DINOv2: using Hugging Face base weights (no finetuned checkpoint at %s)", DINO_FINETUNED_FULL_PATH)
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


def reload_rfdetr():
    """Force-reload RF-DETR from the latest checkpoint. Call after training completes."""
    global _rfdetr_model
    _rfdetr_model = None
    return load_rfdetr()


def reload_classifiers():
    """Force-reload all brand classifiers. Call after classifier or DINOv2 training."""
    global _classifiers, _brand_classifier, _class_mapping
    _classifiers = {}
    _brand_classifier = None
    _class_mapping = None
    device = get_device()
    for pkg_type in PACKAGING_TYPES:
        try:
            load_classifier(device, packaging_type=pkg_type)
        except FileNotFoundError:
            logger.info("No %s classifier found (skipping)", pkg_type)
    logger.info("Brand classifiers reloaded")


def reload_dino():
    """Force-reload DINOv2 backbone. Call after DINOv2 fine-tuning."""
    global _dino_processor, _dino_model
    _dino_processor = None
    _dino_model = None
    device = get_device()
    try:
        load_dino(device)
        logger.info("DINOv2 model reloaded")
    except Exception as exc:
        logger.error("Failed to reload DINOv2: %s", exc)
        raise


def load_ocr():
    """Load EasyOCR reader (English). Cached after first call."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _ocr_reader


EMBED_DIM = 1536  # CLS (768) + mean-pooled patches (768)


def _pad_to_square(img: Image.Image) -> Image.Image:
    """Pad image to square with gray fill to avoid aspect-ratio distortion."""
    w, h = img.size
    if w == h:
        return img
    max_side = max(w, h)
    padded = Image.new("RGB", (max_side, max_side), (128, 128, 128))
    padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return padded


def embed_image(pil_img: Image.Image, processor, model, device: str) -> np.ndarray:
    img = _pad_to_square(pil_img.convert("RGB"))
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]       # (1, 768)
        patch_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # (1, 768)
        combined = torch.cat([cls_token, patch_mean], dim=1)  # (1, 1536)
        vec = combined.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_images_batch(pil_imgs: list[Image.Image], processor, model, device: str) -> np.ndarray:
    """Embed multiple images in a single forward pass. Returns (N, dim) L2-normalised."""
    if not pil_imgs:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    imgs = [_pad_to_square(img.convert("RGB")) for img in pil_imgs]
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]       # (N, 768)
        patch_means = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # (N, 768)
        combined = torch.cat([cls_tokens, patch_means], dim=1)  # (N, 1536)
        vecs = combined.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = vecs / norms
    return vecs


def build_index(device: str, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> None:
    """Train brand classifiers for each packaging type that has reference images."""
    import subprocess
    import sys

    train_script = _PROJECT_ROOT / "brand_classifier.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    trained_types = []
    for idx, pkg_type in enumerate(PACKAGING_TYPES):
        type_dir = REFERENCES_DIR / pkg_type
        if not type_dir.exists():
            continue
        # Check if there are any images
        has_images = any(type_dir.glob("*.[jJ][pP][gG]")) or any(type_dir.glob("*.[jJ][pP][eE][gG]")) or any(type_dir.glob("*.[pP][nN][gG]"))
        if not has_images:
            if progress_cb:
                progress_cb(
                    int((idx + 1) / len(PACKAGING_TYPES) * 100),
                    100,
                    f"Skipping {pkg_type} classifier (no reference images)",
                )
            continue

        if progress_cb:
            progress_cb(
                int(idx / len(PACKAGING_TYPES) * 50),
                100,
                f"Training {pkg_type} classifier...",
            )

        result = subprocess.run(
            [sys.executable, str(train_script), "--packaging-type", pkg_type,
             "--epochs", "100", "--embed-batch-size", "8"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"{pkg_type} classifier training failed:\n{result.stderr}")

        trained_types.append(pkg_type)

    if progress_cb:
        progress_cb(100, 100, f"Classifiers trained: {', '.join(trained_types)}")

    # Reload classifiers
    global _classifiers
    _classifiers = {}
    for pkg_type in trained_types:
        try:
            load_classifier(device, packaging_type=pkg_type)
        except FileNotFoundError:
            pass


def load_classifier(device: str = None, packaging_type: str = "pack"):
    """Load the trained brand classifier and class mapping for a packaging type."""
    global _classifiers, _brand_classifier, _class_mapping

    if packaging_type in _classifiers:
        return _classifiers[packaging_type]

    type_dir = CLASSIFIER_BASE_DIR / packaging_type
    weights_path = type_dir / "best_classifier.pth"
    mapping_path = type_dir / "class_mapping.json"

    if not weights_path.exists() or not mapping_path.exists():
        # Fallback: check legacy flat structure (pre-migration)
        legacy_weights = CLASSIFIER_BASE_DIR / "best_classifier.pth"
        legacy_mapping = CLASSIFIER_BASE_DIR / "class_mapping.json"
        if packaging_type == "pack" and legacy_weights.exists() and legacy_mapping.exists():
            weights_path = legacy_weights
            mapping_path = legacy_mapping
        else:
            raise FileNotFoundError(
                f"Missing {packaging_type} classifier model. Expected {weights_path} and {mapping_path}. "
                f"Run 'python brand_classifier.py --packaging-type {packaging_type}' or /build-index first."
            )

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    num_classes = mapping["num_classes"]
    embed_dim = mapping["embed_dim"]
    hidden_dim = mapping.get("hidden_dim", 512)

    classifier = BrandClassifier(embed_dim, num_classes, hidden_dim)

    if device is None:
        device = get_device()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    classifier.to(device)
    classifier.eval()

    _classifiers[packaging_type] = (classifier, mapping)

    # Keep legacy aliases pointing to pack classifier
    if packaging_type == "pack":
        _brand_classifier = classifier
        _class_mapping = mapping

    logger.info("Loaded %s brand classifier: %d classes", packaging_type, num_classes)
    return classifier, mapping


def classify_embeddings(
    embeddings: np.ndarray,
    device: str,
    top_k: int = CLASSIFIER_TOP_K,
    packaging_type: str = "pack",
) -> list[list[tuple[str, float]]]:
    """Run the brand classifier on pre-computed DINOv2 embeddings.

    Args:
        embeddings: (N, embed_dim) numpy array of L2-normalized embeddings
        device: torch device
        top_k: number of top predictions to return per sample
        packaging_type: "pack" or "box" -- determines which classifier to use

    Returns:
        List of N lists, each containing top_k (label, confidence) tuples.
    """
    classifier, mapping = load_classifier(device, packaging_type=packaging_type)
    idx_to_label = mapping["idx_to_label"]

    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32).to(device)
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)

    results = []
    for i in range(len(embeddings)):
        topk_vals, topk_idxs = torch.topk(probs[i], min(top_k, probs.shape[1]))
        sample_results = []
        for val, idx in zip(topk_vals, topk_idxs):
            label_key = str(idx.item())
            label = idx_to_label.get(label_key, f"unknown_{label_key}")
            sample_results.append((label, float(val.item())))
        results.append(sample_results)

    return results


# Keep for backward compatibility with main.py imports
def load_index(packaging_type: str = "pack"):
    """Load classifier and return (classifier, labels) for backward compatibility."""
    device = get_device()
    classifier, mapping = load_classifier(device, packaging_type=packaging_type)
    labels = list(mapping["label_to_idx"].keys())
    return classifier, labels


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
    """Run EasyOCR on an image. Returns [(bbox, text, confidence), ...]."""
    try:
        reader = load_ocr()
    except Exception as exc:
        logger.warning("OCR disabled because EasyOCR failed to load: %s", exc)
        return []
    try:
        img_np = np.array(image.convert("RGB"))
        results = reader.readtext(img_np)
        return results  # already [(bbox, text, conf)]
    except Exception as exc:
        logger.warning("OCR read failed: %s", exc)
        return []


def _ocr_brand_scores_from_items(ocr_items: list, label_profiles: list[dict]) -> dict[str, float]:
    """Match OCR text to brand labels using fuzzy matching.

    Uses token-level matching with fuzzy per-token comparison so that
    "La pn" matches "lapin", "MEVWS" matches "mevius", etc.
    Requires the OCR text to be at least 60% similar to the brand token
    to count as a match, preventing false positives like "image" -> "esse".
    """
    brand_scores: dict[str, float] = {}

    # Filter out common non-brand text (error pages, generic words)
    noise_words = {"this", "image", "cannot", "displayed", "found", "error",
                   "the", "and", "for", "are", "not", "with", "that", "from",
                   "have", "has", "was", "were", "been", "being", "page"}

    for item in ocr_items:
        if len(item) < 3:
            continue
        try:
            text = str(item[1]).strip()
            text_conf = float(item[2])
        except (IndexError, TypeError, ValueError):
            continue
        norm_text = _normalize_text(text)
        if not norm_text or len(norm_text) < 2:
            continue

        text_tokens = [t for t in norm_text.split() if len(t) >= 2]
        # Skip if all tokens are noise
        meaningful_tokens = [t for t in text_tokens if t not in noise_words]
        if not meaningful_tokens:
            continue

        for prof in label_profiles:
            brand_name = prof["brand"]  # e.g. "mevius", "esse", "555"
            brand_tokens = prof["tokens"]  # e.g. {"mevius", "original"}

            # Strategy 1: Exact token match (highest confidence)
            exact_matches = 0
            for tok in meaningful_tokens:
                if tok in brand_tokens:
                    exact_matches += 1
            if exact_matches > 0:
                score = min(1.0, text_conf * (0.70 + 0.10 * exact_matches))
                if score > brand_scores.get(prof["label"], 0.0):
                    brand_scores[prof["label"]] = float(score)
                continue

            # Strategy 2: Fuzzy token match (for OCR errors like "La pn" -> "lapin")
            # Compare each OCR token against each brand token individually
            # Also try joining the full OCR text as one string for split-word matches
            best_token_fuzzy = 0.0

            # Try the full normalized text against brand name (catches "la pn" -> "lapin")
            joined = norm_text.replace(" ", "")
            if len(joined) >= 3:
                brand_ratio = fuzz.ratio(joined, brand_name) / 100.0
                if brand_ratio >= 0.60:
                    best_token_fuzzy = max(best_token_fuzzy, brand_ratio)

            for ocr_tok in meaningful_tokens:
                if len(ocr_tok) < 3:
                    continue
                # Compare against brand name
                brand_ratio = fuzz.ratio(ocr_tok, brand_name) / 100.0
                if brand_ratio >= 0.60:
                    best_token_fuzzy = max(best_token_fuzzy, brand_ratio)

                # Compare against each brand token
                for bt in brand_tokens:
                    if len(bt) < 3:
                        continue
                    tok_ratio = fuzz.ratio(ocr_tok, bt) / 100.0
                    if tok_ratio >= 0.60:
                        best_token_fuzzy = max(best_token_fuzzy, tok_ratio)

            if best_token_fuzzy >= 0.60:
                score = text_conf * best_token_fuzzy * 0.85
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
    """Detect and classify cigarette brands in an image.

    Pipeline (classifier-first with OCR fallback):
      1. RF-DETR detects bounding boxes with class_id (0=pack, 1=box)
      2. Crops are grouped by packaging type
      3. Each group is classified by its type-specific DINOv2 classifier
      4. EasyOCR runs only on uncertain crops (low confidence / low margin)
      5. OCR signals boost matching classifier families for uncertain crops
    """
    # Load labels for each available packaging type
    type_labels = {}
    type_label_profiles = {}
    for pkg_type in PACKAGING_TYPES:
        try:
            _, pkg_labels = load_index(packaging_type=pkg_type)
            type_labels[pkg_type] = pkg_labels
            type_label_profiles[pkg_type] = _build_label_profiles(pkg_labels)
        except FileNotFoundError:
            pass

    # Fallback: if no type-specific classifiers, use legacy labels
    if not type_labels:
        type_labels["pack"] = labels
        type_label_profiles["pack"] = _build_label_profiles(labels)

    detections = rfdetr_model.predict(image, threshold=RFDETR_CONF_THRESHOLD)
    crops: list[Image.Image] = []
    crop_types: list[str] = []  # "pack" or "box" per crop

    has_detections = len(detections) > 0 if detections is not None else False

    if has_detections:
        width, height = image.size
        xyxy = detections.xyxy
        # class_id: 0 = pack (or single-class legacy), 1 = box
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))

            # Determine packaging type from class_id
            if class_ids is not None and len(class_ids) > i:
                cid = int(class_ids[i])
                crop_types.append("box" if cid == 1 else "pack")
            else:
                crop_types.append("pack")
    else:
        crops.append(image)
        crop_types.append("pack")

    # Group crops by packaging type for batch classification
    type_crop_indices: dict[str, list[int]] = {}
    for idx, pkg_type in enumerate(crop_types):
        effective_type = pkg_type if pkg_type in type_labels else "pack"
        type_crop_indices.setdefault(effective_type, []).append(idx)

    # Embed all crops at once (DINOv2 is shared)
    all_vecs = embed_images_batch(crops, processor, model, device)

    # Classify per type
    all_cls_results: list[list[tuple[str, float]]] = [[] for _ in crops]

    for pkg_type, indices in type_crop_indices.items():
        if not indices:
            continue
        type_vecs = all_vecs[indices]
        type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type=pkg_type)
        for local_idx, global_idx in enumerate(indices):
            all_cls_results[global_idx] = type_results[local_idx]

    # Build combined label profiles for OCR matching (union of all types)
    combined_labels = set()
    for pkg_labels in type_labels.values():
        combined_labels.update(pkg_labels)
    combined_label_profiles = _build_label_profiles(list(combined_labels))
    label_profile_map = {p["label"]: p for p in combined_label_profiles}

    def _needs_ocr_fallback(crop_results: list[tuple[str, float]]) -> bool:
        if not OCR_ENABLED:
            return False
        if not crop_results:
            return True
        top1 = float(crop_results[0][1])
        top2 = float(crop_results[1][1]) if len(crop_results) > 1 else 0.0
        margin = top1 - top2
        return top1 < OCR_FALLBACK_THRESHOLD or margin < OCR_FALLBACK_MARGIN

    ocr_needed = [_needs_ocr_fallback(r) for r in all_cls_results]

    # OCR fallback only for uncertain crops
    per_crop_ocr_scores: list[dict[str, float]] = [{} for _ in crops]
    ocr_best: dict[str, float] = {}
    for idx, crop in enumerate(crops):
        if not ocr_needed[idx]:
            continue
        ocr_items = _run_ocr_on_image(crop)
        crop_ocr_scores = _ocr_brand_scores_from_items(ocr_items, combined_label_profiles)
        per_crop_ocr_scores[idx] = crop_ocr_scores
        for label, conf in crop_ocr_scores.items():
            if conf > ocr_best.get(label, 0.0):
                ocr_best[label] = conf

    # Optional full-image OCR context
    fullimg_ocr_families: dict[str, float] = {}
    if OCR_FULLIMG_ENABLED and has_detections and any(ocr_needed):
        fullimg_scores = _ocr_brand_scores_from_items(_run_ocr_on_image(image), combined_label_profiles)
        for label, conf in fullimg_scores.items():
            if conf > ocr_best.get(label, 0.0):
                ocr_best[label] = conf
            family = label_profile_map.get(label, {}).get("brand", "")
            if family:
                fullimg_ocr_families[family] = max(fullimg_ocr_families.get(family, 0.0), conf)

    # Fuse classifier with OCR fallback
    fused: dict[str, float] = {}
    for crop_idx, crop_results in enumerate(all_cls_results):
        crop_ocr_scores = per_crop_ocr_scores[crop_idx]
        crop_ocr_families: dict[str, float] = {}
        for label, ocr_conf in crop_ocr_scores.items():
            family = label_profile_map.get(label, {}).get("brand", "")
            if family:
                crop_ocr_families[family] = max(crop_ocr_families.get(family, 0.0), ocr_conf)

        for label, cls_conf in crop_results:
            out_conf = float(cls_conf)
            if ocr_needed[crop_idx]:
                brand_family = label_profile_map.get(label, {}).get("brand", "")
                fam_conf = 0.0
                if brand_family:
                    fam_conf = max(
                        crop_ocr_families.get(brand_family, 0.0),
                        fullimg_ocr_families.get(brand_family, 0.0),
                    )
                if fam_conf >= OCR_STRONG_THRESHOLD:
                    out_conf = min(1.0, out_conf + fam_conf * 0.15)
                elif fam_conf > 0:
                    out_conf = min(1.0, out_conf + fam_conf * 0.05)
            if out_conf > fused.get(label, 0.0):
                fused[label] = out_conf

    # OCR-only brands
    for label, ocr_conf in ocr_best.items():
        if label in fused:
            continue
        if ocr_conf >= OCR_INDEPENDENT_MIN_SCORE:
            fused[label] = min(1.0, ocr_conf * 0.70)

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
    """Process a CSV of image URLs and output brand/product detection in Q12A/Q12B format."""
    try:
        from .output_format import build_q12_row, get_output_columns
    except ImportError:
        from output_format import build_q12_row, get_output_columns

    device = get_device()
    classifier, labels = load_index()
    processor, model = load_dino(device)
    rfdetr_model = load_rfdetr()
    index = classifier

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

    # Resume support
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

    def _flush(rows):
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    for row_idx in range(start_row, process_len):
        row = df.iloc[row_idx]
        try:
            images = _download_row_images(row, url_columns)

            # Collect all detected products across all images for this row
            all_products: dict[str, float] = {}
            for col in url_columns:
                col_name = str(col)
                image = images.get(col_name)
                if image is None:
                    continue
                col_best = _detect_brands_from_image(
                    image=image, rfdetr_model=rfdetr_model,
                    processor=processor, model=model,
                    device=device, index=index, labels=labels,
                )
                for product, conf in col_best.items():
                    if conf >= MIN_OUTPUT_CONFIDENCE:
                        if conf > all_products.get(product, 0.0):
                            all_products[product] = conf

            # Sort by confidence and get product names
            detected = [p for p, _ in sorted(all_products.items(), key=lambda x: -x[1])]

            # Build Q12A/Q12B formatted row
            q12_row = build_q12_row(detected)
            result_row = {"Respondent.Serial": row[id_col], "Q6": ""}
            result_row.update(q12_row)

            # Preserve photo URL columns
            for col in url_columns:
                col_name = str(col)
                result_row[col_name] = row[col] if not pd.isna(row[col]) else ""

        except Exception as exc:
            logger.error("Row %d failed: %s", row_idx, exc, exc_info=True)
            result_row = {"Respondent.Serial": row[id_col], "Q6": ""}

        completed_rows.append(result_row)

        if (row_idx + 1) % SAVE_INTERVAL == 0 or row_idx + 1 == process_len:
            _flush(completed_rows)

        if progress_cb:
            progress_cb(row_idx + 1, process_len, f"Processed row {row_idx + 1}/{process_len}")

    # NOT_PROCESSED markers for remaining rows
    if BATCH_MODE is not None:
        for ri in range(process_len, original_len):
            row = df.iloc[ri]
            result_row = {"Respondent.Serial": row[id_col], "Q6": "NOT_PROCESSED"}
            completed_rows.append(result_row)
        _flush(completed_rows)

    return out_path
