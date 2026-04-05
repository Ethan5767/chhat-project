import gc
import json
import io
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
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
RFDETR_CONF_THRESHOLD = 0.35
MIN_OUTPUT_CONFIDENCE = 0.80
CLASSIFIER_TOP_K = 5

import threading as _threading

_dino_processor = None
_dino_model = None
_rfdetr_model = None
_rfdetr_models: dict[str, object] = {}  # model_size -> loaded model instance
# Per-type classifiers: {"pack": (classifier, mapping), "box": (classifier, mapping)}
_classifiers: dict[str, tuple] = {}
_model_load_lock = _threading.RLock()  # Thread-safe model loading
# Legacy single-classifier aliases (for backward compat)
_brand_classifier = None
_class_mapping = None

# Co-DETR globals
CODETR_CONF_THRESHOLD = 0.10
DEFAULT_DETECTOR = os.environ.get("CHHAT_DETECTOR_BACKEND", "codetr")
_codetr_model = None
_codetr_model_lock = _threading.Lock()


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

    For each product, keeps the highest confidence score across all matching
    reference photos.
    """
    product_best: dict[str, float] = {}

    for label, conf in label_scores.items():
        product = label_to_product(label)
        if conf > product_best.get(product, 0.0):
            product_best[product] = conf

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


def _find_best_checkpoint(model_size: str = "medium") -> Optional[Path]:
    if not _RFDETR_CHECKPOINT_DIR.exists():
        return None
    # Look in size-specific subdirectory first (e.g. runs/medium/, runs/large/)
    size_dir = _RFDETR_CHECKPOINT_DIR / model_size
    search_dirs = [size_dir, _RFDETR_CHECKPOINT_DIR] if size_dir.exists() else [_RFDETR_CHECKPOINT_DIR]
    for d in search_dirs:
        for pattern in ("best*.pth", "*.pth", "best*.pt", "*.pt"):
            # Only search immediate children (not recursively into other size dirs)
            candidates = [f for f in d.glob(pattern) if f.is_file()]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def load_rfdetr(model_size: str = "medium"):
    """Load an RF-DETR model by size. Caches each size separately.

    Args:
        model_size: One of "nano", "small", "base", "medium", "large", "xlarge", "2xlarge".
    """
    global _rfdetr_model, _rfdetr_models
    size = model_size.lower().strip()
    if size not in ("nano", "small", "base", "medium", "large", "xlarge", "2xlarge"):
        size = "medium"

    # Return cached model if available
    if size in _rfdetr_models:
        _rfdetr_model = _rfdetr_models[size]
        return _rfdetr_model

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
    model_classes = {"nano": RFDETRNano, "small": RFDETRSmall, "base": RFDETRBase, "medium": RFDETRMedium, "large": RFDETRLarge}

    # XL/2XL require rfdetr[plus]
    try:
        from rfdetr import RFDETRXLarge, RFDETR2XLarge
        model_classes["xlarge"] = RFDETRXLarge
        model_classes["2xlarge"] = RFDETR2XLarge
    except ImportError:
        if size in ("xlarge", "2xlarge"):
            logger.error("RF-DETR %s requires rfdetr[plus]. Install with: pip install 'rfdetr[plus]'", size)
            size = "medium"

    cls = model_classes[size]

    checkpoint = _find_best_checkpoint(model_size=size)
    kwargs = {}
    if size in ("xlarge", "2xlarge"):
        kwargs["accept_platform_model_license"] = True
    if checkpoint:
        logger.info("Loading fine-tuned RF-DETR-%s from %s", size, checkpoint)
        model = cls(pretrain_weights=str(checkpoint), **kwargs)
    else:
        logger.info("No fine-tuned checkpoint found, using pre-trained RF-DETR-%s", size)
        model = cls(**kwargs)

    # Skip optimize_for_inference on RunPod -- torch.compile can segfault on some CUDA setups
    _is_runpod = os.environ.get("RUNPOD_POD_ID") or os.path.exists("/workspace")
    if not _is_runpod:
        try:
            model.optimize_for_inference()
            logger.info("RF-DETR-%s optimized for inference", size)
        except Exception as exc:
            logger.warning("Could not optimize RF-DETR-%s for inference: %s", size, exc)
    else:
        logger.info("RF-DETR-%s: skipping optimize_for_inference on RunPod", size)

    _rfdetr_models[size] = model
    _rfdetr_model = model
    return model


def reload_rfdetr(model_size: str = "medium"):
    """Force-reload RF-DETR from the latest checkpoint. Call after training completes."""
    global _rfdetr_model, _rfdetr_models
    _rfdetr_model = None
    _rfdetr_models.pop(model_size, None)
    return load_rfdetr(model_size=model_size)


def load_codetr():
    """Load Co-DETR Swin-L model using mmdet. Cached after first load."""
    global _codetr_model
    with _codetr_model_lock:
        if _codetr_model is not None:
            return _codetr_model

        try:
            from mmdet.apis import init_detector
        except ImportError:
            raise ImportError(
                "mmdet is required for Co-DETR. Install with: "
                "pip install mmdet mmengine mmcv"
            )

        # Find checkpoint
        ckpt_candidates = [
            _DATA_ROOT / "co_detr_weights" / "finetuned_epoch12.pth",
            _PROJECT_ROOT / "co_detr_weights" / "finetuned_epoch12.pth",
        ]
        checkpoint = None
        for c in ckpt_candidates:
            if c.is_file():
                checkpoint = str(c)
                break
        if checkpoint is None:
            raise FileNotFoundError(
                "Co-DETR checkpoint not found. Expected finetuned_epoch12.pth in "
                "co_detr_weights/ under project root or CHHAT_DATA_ROOT."
            )

        # Find config
        config_candidates = [
            _BACKEND_ROOT / "co_detr_inference_config.py",
            _PROJECT_ROOT / "backend" / "co_detr_inference_config.py",
        ]
        config_path = None
        for c in config_candidates:
            if c.is_file():
                config_path = str(c)
                break
        if config_path is None:
            raise FileNotFoundError("Co-DETR inference config not found")

        device = get_device()
        logger.info("Loading Co-DETR Swin-L from %s on %s...", checkpoint, device)
        # PyTorch 2.6+ defaults weights_only=True which breaks mmengine checkpoints
        # that contain HistoryBuffer objects. Temporarily patch torch.load.
        import functools
        _orig_torch_load = torch.load
        torch.load = functools.partial(_orig_torch_load, weights_only=False)
        try:
            model = init_detector(config_path, checkpoint, device=device)
        finally:
            torch.load = _orig_torch_load
        model.eval()
        logger.info("Co-DETR Swin-L loaded successfully")

        _codetr_model = model
        return _codetr_model


def reload_codetr():
    """Force-reload Co-DETR model."""
    global _codetr_model
    with _codetr_model_lock:
        _codetr_model = None
    return load_codetr()


def detect_objects(image: Image.Image, backend: str = None, threshold: float = None, model_size: str = "medium"):
    """Unified detection interface. Returns list of dicts with keys:
    xyxy (list[4]), confidence (float), class_id (int).

    Args:
        image: PIL Image
        backend: "codetr" or "rfdetr". Defaults to DEFAULT_DETECTOR.
        threshold: Confidence threshold. Defaults to per-detector default.
        model_size: RF-DETR model size (ignored for Co-DETR).
    """
    if backend is None:
        backend = DEFAULT_DETECTOR

    if backend == "codetr":
        if threshold is None:
            threshold = CODETR_CONF_THRESHOLD
        return _detect_codetr(image, threshold)
    else:
        if threshold is None:
            threshold = RFDETR_CONF_THRESHOLD
        return _detect_rfdetr(image, threshold, model_size=model_size)


def _detect_rfdetr(image: Image.Image, threshold: float, model_size: str = "medium"):
    """RF-DETR detection, returns normalized format."""
    rfdetr_model = load_rfdetr(model_size=model_size)
    detections = rfdetr_model.predict(image, threshold=threshold)
    results = []
    if detections is not None and len(detections) > 0:
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            cid = 0
            if class_ids is not None and len(class_ids) > i:
                cid = int(class_ids[i])
            results.append({
                "xyxy": [float(v) for v in box],
                "confidence": float(conf),
                "class_id": cid,
            })
    return results


def _detect_codetr(image: Image.Image, threshold: float):
    """Co-DETR detection, returns normalized format."""
    return _detect_codetr_batch([image], threshold)[0]


def _detect_codetr_batch(images: list[Image.Image], threshold: float) -> list[list[dict]]:
    """Batch Co-DETR detection. Returns list of detection lists (one per image)."""
    model = load_codetr()
    img_arrays = [np.array(img)[:, :, ::-1] for img in images]  # RGB -> BGR

    from mmdet.apis import inference_detector
    batch_results = inference_detector(model, img_arrays)

    all_detections = []
    for result in batch_results:
        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        dets = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                dets.append({
                    "xyxy": [float(v) for v in bboxes[i]],
                    "confidence": float(scores[i]),
                    "class_id": 0,
                })
        all_detections.append(dets)
    return all_detections


def detect_objects_batch(images: list[Image.Image], backend: str = None,
                         threshold: float = None) -> list[list[dict]]:
    """Batch detection interface. Returns list of detection lists (one per image)."""
    if backend is None:
        backend = DEFAULT_DETECTOR
    if backend == "codetr":
        return _detect_codetr_batch(images, threshold or CODETR_CONF_THRESHOLD)
    else:
        # RF-DETR doesn't support batch natively, fall back to sequential
        t = threshold or RFDETR_CONF_THRESHOLD
        return [_detect_rfdetr(img, t) for img in images]


def reload_classifiers():
    """Force-reload all brand classifiers. Call after classifier or DINOv2 training."""
    global _classifiers, _brand_classifier, _class_mapping, _cached_type_labels
    _classifiers = {}
    _brand_classifier = None
    _class_mapping = None
    _cached_type_labels = None  # Reset so batch pipeline picks up new labels
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


EMBED_BATCH_SIZE = 8  # limit peak tensor memory on low-RAM servers
# H100-optimised batch size -- set by run_pipeline when CUDA available
_EMBED_BATCH_SIZE_OVERRIDE: Optional[int] = None


def _get_embed_batch_size() -> int:
    if _EMBED_BATCH_SIZE_OVERRIDE is not None:
        return _EMBED_BATCH_SIZE_OVERRIDE
    return EMBED_BATCH_SIZE


def _preprocess_crops_parallel(pil_imgs: list[Image.Image], max_workers: int = 8) -> list[Image.Image]:
    """Pad-to-square + RGB convert in parallel threads."""
    def _prep(img):
        return _pad_to_square(img.convert("RGB"))
    if len(pil_imgs) <= 4:
        return [_prep(img) for img in pil_imgs]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(pil_imgs))) as pool:
        return list(pool.map(_prep, pil_imgs))


def embed_images_batch(pil_imgs: list[Image.Image], processor, model, device: str) -> np.ndarray:
    """Embed images in chunks to limit peak memory. Returns (N, dim) L2-normalised."""
    if not pil_imgs:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    batch_size = _get_embed_batch_size()
    imgs = _preprocess_crops_parallel(pil_imgs)
    all_vecs = []
    use_amp = device == "cuda"
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i:i + batch_size]
        with torch.no_grad():
            inputs = processor(images=chunk, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]       # (B, 768)
            patch_means = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # (B, 768)
            combined = torch.cat([cls_tokens, patch_means], dim=1)  # (B, 1536)
            all_vecs.append(combined.float().detach().cpu().numpy().astype(np.float32))
            del inputs, outputs, cls_tokens, patch_means, combined
    vecs = np.concatenate(all_vecs, axis=0)
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

    with _model_load_lock:
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
        idx_to_label = mapping.get("idx_to_label", {})
        if len(idx_to_label) != num_classes:
            raise ValueError(
                f"Class mapping mismatch for {packaging_type}: idx_to_label has {len(idx_to_label)} "
                f"entries but num_classes={num_classes}. Retrain with /train-classifier."
            )

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
        k = min(top_k, probs.shape[1])
        topk_vals, topk_idxs = torch.topk(probs, k, dim=1)  # (N, k) batched
        topk_vals_np = topk_vals.cpu().numpy()
        topk_idxs_np = topk_idxs.cpu().numpy()

    results = []
    for i in range(len(embeddings)):
        sample_results = []
        for j in range(k):
            label_key = str(topk_idxs_np[i, j])
            label = idx_to_label.get(label_key, f"unknown_{label_key}")
            sample_results.append((label, float(topk_vals_np[i, j])))
        results.append(sample_results)

    return results


# Keep for backward compatibility with main.py imports
def load_index(packaging_type: str = "pack"):
    """Load classifier and return (classifier, labels) for backward compatibility."""
    device = get_device()
    classifier, mapping = load_classifier(device, packaging_type=packaging_type)
    labels = list(mapping["label_to_idx"].keys())
    return classifier, labels


# Image cache directory -- set by run_pipeline or externally before processing
IMAGE_CACHE_DIR: Optional[Path] = None


def _extract_image_id(url: str) -> Optional[str]:
    """Extract the ID parameter from an Ipsos image URL."""
    m = re.search(r'[?&]ID=(\d+)', url, re.IGNORECASE)
    return m.group(1) if m else None


def download_image(url: str) -> Optional[Image.Image]:
    # Check local cache first
    if IMAGE_CACHE_DIR is not None:
        img_id = _extract_image_id(url)
        if img_id:
            cached_path = IMAGE_CACHE_DIR / f"{img_id}.jpg"
            if cached_path.exists():
                try:
                    return Image.open(cached_path).convert("RGB")
                except Exception:
                    pass  # fall through to network download
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
    """Auto-detect columns that contain image URLs by scanning rows.

    Uses column name matching first (photo/link/image/url), then falls back
    to scanning for columns with at least 1 URL in the first 50 rows.
    Also checks the first data row for 'Photo' or 'Link' descriptor text.
    """
    matches = []
    sample = df.head(min(50, len(df)))
    for col in list(df.columns):
        col_str = str(col).strip()
        # Check column name
        if "photo" in col_str.lower() or "link" in col_str.lower() or "image" in col_str.lower() or "url" in col_str.lower():
            matches.append(col)
            continue
        # Check first row for descriptor text like "Photo Link"
        first_val = str(df[col].iloc[0]) if len(df) > 0 else ""
        if "photo" in first_val.lower() or "link" in first_val.lower() or "image" in first_val.lower():
            matches.append(col)
            continue
        # Check if any rows contain URLs (low threshold to catch sparse columns)
        url_count = sample[col].apply(_looks_like_url).sum()
        if url_count >= 1:
            matches.append(col)
    return matches


def _valid_url_value(value) -> bool:
    return _looks_like_url(value)


def _detect_brands_from_image(
    image: Image.Image,
    rfdetr_model,
    processor,
    model,
    device: str,
    index,
    labels: list[str],
    detector_backend: str = None,
    det_threshold: float = None,
) -> dict[str, float]:
    """Detect and classify cigarette brands in an image.

    Pipeline:
      1. RF-DETR detects bounding boxes with class_id (0=pack, 1=box)
      2. Crops are grouped by packaging type
      3. Each group is classified by its type-specific DINOv2 classifier
    """
    # Load labels for each available packaging type
    type_labels = {}
    for pkg_type in PACKAGING_TYPES:
        try:
            _, pkg_labels = load_index(packaging_type=pkg_type)
            type_labels[pkg_type] = pkg_labels
        except FileNotFoundError:
            pass

    # Fallback: if no type-specific classifiers, use legacy labels
    if not type_labels:
        type_labels["pack"] = labels

    det_results = detect_objects(image, backend=detector_backend, threshold=det_threshold)
    crops: list[Image.Image] = []
    crop_types: list[str] = []  # "pack" or "box" per crop

    if det_results:
        width, height = image.size
        for det in det_results:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            crop_types.append("box" if det["class_id"] == 1 else "pack")
    else:
        crops.append(image)
        crop_types.append("pack")

    # Group crops by packaging type for batch classification
    type_crop_indices: dict[str, list[int]] = {}
    for idx, pkg_type in enumerate(crop_types):
        effective_type = pkg_type if pkg_type in type_labels else "pack"
        type_crop_indices.setdefault(effective_type, []).append(idx)

    # Embed crops in chunks (DINOv2 is shared)
    all_vecs = embed_images_batch(crops, processor, model, device)

    # Classify per type; box crops fall back to pack classifier if box confidence is low
    BOX_FALLBACK_THRESHOLD = 0.50  # if box top-1 confidence below this, use pack classifier instead
    all_cls_results: list[list[tuple[str, float]]] = [[] for _ in crops]

    for pkg_type, indices in type_crop_indices.items():
        if not indices:
            continue
        type_vecs = all_vecs[indices]
        type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type=pkg_type)
        for local_idx, global_idx in enumerate(indices):
            all_cls_results[global_idx] = type_results[local_idx]

    # Box -> pack fallback: if box classifier confidence is low, re-classify with pack
    if "box" in type_crop_indices and "pack" in type_labels:
        box_indices_to_retry = []
        for global_idx in type_crop_indices.get("box", []):
            results = all_cls_results[global_idx]
            top_conf = results[0][1] if results else 0.0
            if top_conf < BOX_FALLBACK_THRESHOLD:
                box_indices_to_retry.append(global_idx)
        if box_indices_to_retry:
            retry_vecs = all_vecs[box_indices_to_retry]
            pack_results = classify_embeddings(retry_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type="pack")
            for local_idx, global_idx in enumerate(box_indices_to_retry):
                pack_top = pack_results[local_idx][0][1] if pack_results[local_idx] else 0.0
                box_top = all_cls_results[global_idx][0][1] if all_cls_results[global_idx] else 0.0
                if pack_top > box_top:
                    all_cls_results[global_idx] = pack_results[local_idx]

    # Aggregate classifier results directly
    fused: dict[str, float] = {}
    for crop_results in all_cls_results:
        for label, cls_conf in crop_results:
            out_conf = float(cls_conf)
            if out_conf > fused.get(label, 0.0):
                fused[label] = out_conf

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


def _process_single_row(row, url_columns, id_col, df_columns, rfdetr_model, processor, model, device, index, labels, build_q12_row, images=None, detector_backend: str = None):
    """Process a single row: download images (if not prefetched), detect brands, return result dict."""
    try:
        if images is None:
            images = _download_row_images(row, url_columns)

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
                detector_backend=detector_backend,
            )
            for product, conf in col_best.items():
                if conf >= MIN_OUTPUT_CONFIDENCE:
                    if conf > all_products.get(product, 0.0):
                        all_products[product] = conf

        detected = [p for p, _ in sorted(all_products.items(), key=lambda x: -x[1])]
        q12_row = build_q12_row(detected)
        result_row = {"Respondent.Serial": row[id_col], "Q6": ""}
        result_row.update(q12_row)

        photo_cols = [c for c in df_columns if c in url_columns or
                      any(k in str(c).lower() for k in ("q30", "q33", "photo", "link", "image"))]
        for col in photo_cols:
            col_name = str(col)
            result_row[col_name] = row[col] if not pd.isna(row[col]) else ""

        if all_products:
            avg_conf = sum(all_products.values()) / len(all_products)
            result_row["overall_confidence"] = round(avg_conf, 3)
        else:
            result_row["overall_confidence"] = 0.0

    except Exception as exc:
        logger.error("Row failed: %s", exc, exc_info=True)
        result_row = {"Respondent.Serial": row[id_col], "Q6": "", "overall_confidence": 0.0}

    return result_row


# ---------------------------------------------------------------------------
# GPU-optimised multi-row batch processing
# ---------------------------------------------------------------------------
MULTI_ROW_BATCH = 8   # number of rows to process in one GPU mega-batch
PREFETCH_AHEAD = 48   # prefetch images for upcoming rows


_cached_type_labels: Optional[dict] = None


def _get_type_labels_cached(labels):
    """Cache type labels across batches to avoid reloading every call."""
    global _cached_type_labels
    with _model_load_lock:
        if _cached_type_labels is not None:
            return _cached_type_labels
        type_labels = {}
        for pkg_type in PACKAGING_TYPES:
            try:
                _, pkg_labels = load_index(packaging_type=pkg_type)
                type_labels[pkg_type] = pkg_labels
            except FileNotFoundError:
                pass
        if not type_labels:
            type_labels["pack"] = labels
        _cached_type_labels = type_labels
        return type_labels


def _process_rows_batched(
    rows_with_images: list[tuple],  # [(row, images_dict), ...]
    url_columns, id_col, df_columns, rfdetr_model, processor, model,
    device, index, labels, build_q12_row,
    detector_backend: str = None,
) -> list[dict]:
    """Process multiple rows at once, batching ALL crops across ALL images across ALL rows
    through a single DINOv2 + classifier pass for maximum GPU utilization."""

    type_labels = _get_type_labels_cached(labels)

    # Phase 1: Detection on all images, collect all crops
    row_col_crops: list[dict[str, list[tuple[Image.Image, str]]]] = []
    all_crops: list[Image.Image] = []
    all_crop_types: list[str] = []
    crop_to_rowcol: list[tuple[int, str, int]] = []

    for row_idx, (row, images) in enumerate(rows_with_images):
        col_crops: dict[str, list[tuple[Image.Image, str]]] = {}
        for col in url_columns:
            col_name = str(col)
            image = images.get(col_name) if images else None
            if image is None:
                col_crops[col_name] = []
                continue

            det_results = detect_objects(image, backend=detector_backend)

            crops_for_col: list[tuple[Image.Image, str]] = []
            if det_results:
                width, height = image.size
                for det in det_results:
                    x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
                    bw, bh = x2 - x1, y2 - y1
                    pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(width, x2 + pad_x)
                    y2 = min(height, y2 + pad_y)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = image.crop((x1, y1, x2, y2))
                    ctype = "box" if det["class_id"] == 1 else "pack"
                    crops_for_col.append((crop, ctype))
            else:
                crops_for_col.append((image, "pack"))

            for cidx, (crop, ctype) in enumerate(crops_for_col):
                crop_to_rowcol.append((row_idx, col_name, cidx))
                all_crops.append(crop)
                all_crop_types.append(ctype)

            col_crops[col_name] = crops_for_col
        row_col_crops.append(col_crops)

    if not all_crops:
        # No images at all -- return empty result rows
        results = []
        for row_idx, (row, _) in enumerate(rows_with_images):
            result_row = {"Respondent.Serial": row[id_col], "Q6": "", "overall_confidence": 0.0}
            results.append(result_row)
        return results

    # Phase 2: Mega-batch DINOv2 embedding of ALL crops at once
    all_vecs = embed_images_batch(all_crops, processor, model, device)

    # Phase 3: Classify all embeddings grouped by packaging type
    type_crop_indices: dict[str, list[int]] = {}
    for idx, ctype in enumerate(all_crop_types):
        effective = ctype if ctype in type_labels else "pack"
        type_crop_indices.setdefault(effective, []).append(idx)

    all_cls_results: list[list[tuple[str, float]]] = [[] for _ in all_crops]
    for pkg_type, indices in type_crop_indices.items():
        if not indices:
            continue
        type_vecs = all_vecs[indices]
        type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type=pkg_type)
        for local_idx, global_idx in enumerate(indices):
            all_cls_results[global_idx] = type_results[local_idx]

    # Box -> pack fallback
    BOX_FALLBACK_THRESHOLD = 0.50
    if "box" in type_crop_indices and "pack" in type_labels:
        box_retry = [gi for gi in type_crop_indices.get("box", [])
                     if not all_cls_results[gi] or all_cls_results[gi][0][1] < BOX_FALLBACK_THRESHOLD]
        if box_retry:
            retry_vecs = all_vecs[box_retry]
            pack_results = classify_embeddings(retry_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type="pack")
            for li, gi in enumerate(box_retry):
                pack_top = pack_results[li][0][1] if pack_results[li] else 0.0
                box_top = all_cls_results[gi][0][1] if all_cls_results[gi] else 0.0
                if pack_top > box_top:
                    all_cls_results[gi] = pack_results[li]

    # Phase 4: Map results back to rows and build output
    # Group crop results by (row_idx, col_name)
    row_col_results: dict[tuple[int, str], list[tuple[str, float]]] = defaultdict(list)
    for flat_idx, (row_idx, col_name, _) in enumerate(crop_to_rowcol):
        for label, conf in all_cls_results[flat_idx]:
            row_col_results[(row_idx, col_name)].append((label, conf))

    result_rows = []
    for row_idx, (row, _) in enumerate(rows_with_images):
        try:
            all_products: dict[str, float] = {}
            for col in url_columns:
                col_name = str(col)
                crop_results = row_col_results.get((row_idx, col_name), [])
                if not crop_results:
                    continue
                fused: dict[str, float] = {}
                for label, conf in crop_results:
                    if float(conf) > fused.get(label, 0.0):
                        fused[label] = float(conf)
                products = _aggregate_to_products(fused)
                for product, pconf in products.items():
                    if pconf >= MIN_OUTPUT_CONFIDENCE:
                        if pconf > all_products.get(product, 0.0):
                            all_products[product] = pconf

            detected = [p for p, _ in sorted(all_products.items(), key=lambda x: -x[1])]
            q12_row = build_q12_row(detected)
            result_row = {"Respondent.Serial": row[id_col], "Q6": ""}
            result_row.update(q12_row)

            photo_cols = [c for c in df_columns if c in url_columns or
                          any(k in str(c).lower() for k in ("q30", "q33", "photo", "link", "image"))]
            for col in photo_cols:
                col_name = str(col)
                result_row[col_name] = row[col] if not pd.isna(row[col]) else ""

            if all_products:
                avg_conf = sum(all_products.values()) / len(all_products)
                result_row["overall_confidence"] = round(avg_conf, 3)
            else:
                result_row["overall_confidence"] = 0.0
        except Exception as exc:
            logger.error("Row %d failed in batch: %s", row_idx, exc, exc_info=True)
            result_row = {"Respondent.Serial": row[id_col], "Q6": "", "overall_confidence": 0.0}

        result_rows.append(result_row)

    # Free GPU memory
    del all_vecs, all_crops, all_cls_results
    return result_rows


def run_pipeline(csv_path, progress_cb: Optional[Callable[[int, int, str], None]] = None, detector_backend: str = None) -> Path:
    """Process a CSV of image URLs and output brand/product detection in Q12A/Q12B format.

    Optimization: prefetches images for upcoming rows in background threads while
    the GPU processes the current row, overlapping network I/O with GPU compute.
    """
    try:
        from .output_format import build_q12_row, get_output_columns
    except ImportError:
        from output_format import build_q12_row, get_output_columns

    device = get_device()
    # On RunPod pods, require GPU -- abort early if only CPU available
    if os.environ.get("RUNPOD_POD_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/workspace"):
        if device == "cpu":
            raise RuntimeError("GPU required but not available. Aborting to avoid slow CPU processing.")
    logger.info("Pipeline device: %s", device)

    # H100/A100 GPU optimisation: large batch size for DINOv2 embeddings
    global _EMBED_BATCH_SIZE_OVERRIDE
    if device == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            if vram_gb >= 70:       # H100 80GB, A100 80GB
                _EMBED_BATCH_SIZE_OVERRIDE = 128
            elif vram_gb >= 30:     # A100 40GB, A6000
                _EMBED_BATCH_SIZE_OVERRIDE = 64
            elif vram_gb >= 16:     # RTX 4090, L4
                _EMBED_BATCH_SIZE_OVERRIDE = 32
            else:
                _EMBED_BATCH_SIZE_OVERRIDE = 16
            logger.info("GPU VRAM %.1f GB -> embed_batch_size=%d", vram_gb, _EMBED_BATCH_SIZE_OVERRIDE)
        except Exception:
            _EMBED_BATCH_SIZE_OVERRIDE = 32

    classifier, labels = load_index()
    processor, model = load_dino(device)
    _active_backend = detector_backend or DEFAULT_DETECTOR
    if _active_backend == "codetr":
        load_codetr()
        rfdetr_model = None
    else:
        rfdetr_model = load_rfdetr()
    index = classifier

    csv_path = Path(csv_path)

    # Auto-detect image cache: check for image_cache/ next to the CSV
    global IMAGE_CACHE_DIR
    candidate_cache = csv_path.parent / "image_cache"
    if candidate_cache.is_dir() and any(candidate_cache.glob("*.jpg")):
        IMAGE_CACHE_DIR = candidate_cache
        logger.info("Image cache found: %s (%d files)", candidate_cache, len(list(candidate_cache.glob("*.jpg"))))
    else:
        logger.info("No image cache found, will download from network")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    original_len = len(df)
    all_url_columns = get_url_columns(df)
    if not all_url_columns:
        raise ValueError(
            "No URL columns found. Columns should contain 'photo', 'link', 'image', or 'url' in the header, "
            "or have at least 1 URL in the first 50 rows."
        )
    # Only process Q30 (shelf/product photos) for detection. Q33 (store exterior) adds false positives.
    # Q33 columns are still preserved in the output CSV.
    url_columns = [c for c in all_url_columns if "q33" not in str(c).lower()]
    if not url_columns:
        url_columns = all_url_columns  # fallback if no Q30 columns found
    logger.info("Detection columns: %s (skipped store photos: %s)",
                url_columns, [c for c in all_url_columns if c not in url_columns])

    process_len = original_len if BATCH_MODE is None else min(BATCH_MODE, original_len)
    out_path = csv_path.parent / f"{csv_path.stem}_results.csv"
    id_col = df.columns[0]

    # Resume support
    start_row = 0
    completed_rows: list[dict] = []
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path, encoding="utf-8-sig")
            start_row = len(existing_df)
            completed_rows = existing_df.to_dict("records")
            logger.info("Resuming from row %d (%d rows already done)", start_row, start_row)
        except Exception:
            start_row = 0
            completed_rows = []

    def _flush(rows):
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    # Prefetch: submit image downloads for upcoming rows in a background pool
    batch_size = MULTI_ROW_BATCH if device == "cuda" else 1
    prefetch_pool = ThreadPoolExecutor(max_workers=PREFETCH_AHEAD, thread_name_prefix="prefetch")
    prefetch_futures: dict[int, any] = {}

    def _submit_prefetch(row_idx):
        if row_idx < process_len and row_idx not in prefetch_futures:
            row = df.iloc[row_idx]
            prefetch_futures[row_idx] = prefetch_pool.submit(_download_row_images, row, url_columns)

    # Seed the prefetch queue
    for i in range(start_row, min(start_row + PREFETCH_AHEAD * batch_size, process_len)):
        _submit_prefetch(i)

    import time as _time_mod
    _pipeline_start = _time_mod.monotonic()
    _last_log_time = _pipeline_start

    row_idx = start_row
    while row_idx < process_len:
        # Collect a batch of rows with their prefetched images
        batch_end = min(row_idx + batch_size, process_len)
        rows_with_images = []
        for ri in range(row_idx, batch_end):
            row = df.iloc[ri]
            images = None
            if ri in prefetch_futures:
                try:
                    images = prefetch_futures.pop(ri).result(timeout=120)
                except Exception:
                    images = None
            if images is None:
                images = _download_row_images(row, url_columns)
            rows_with_images.append((row, images))

        # Clean up consumed futures
        for k in list(prefetch_futures.keys()):
            if k < batch_end:
                prefetch_futures.pop(k, None)

        # Submit prefetch for upcoming rows
        for i in range(batch_end, min(batch_end + PREFETCH_AHEAD * batch_size, process_len)):
            _submit_prefetch(i)

        # Process the batch
        if batch_size > 1 and device == "cuda":
            batch_results = _process_rows_batched(
                rows_with_images, url_columns, id_col, df.columns,
                rfdetr_model, processor, model, device, index, labels, build_q12_row,
                detector_backend=_active_backend,
            )
        else:
            batch_results = []
            for row, images in rows_with_images:
                r = _process_single_row(
                    row, url_columns, id_col, df.columns, rfdetr_model,
                    processor, model, device, index, labels, build_q12_row, images,
                    detector_backend=_active_backend,
                )
                batch_results.append(r)

        completed_rows.extend(batch_results)

        # Free memory
        rows_with_images = None
        batch_results = None
        if batch_end % 100 < batch_size:
            gc.collect()

        if batch_end % SAVE_INTERVAL == 0 or batch_end == process_len:
            _flush(completed_rows)

        # Timing log every row
        _now = _time_mod.monotonic()
        _elapsed = _now - _pipeline_start
        _row_time = _now - _last_log_time
        _rows_done = batch_end - start_row
        _avg = _elapsed / _rows_done if _rows_done else 0
        _remaining = (process_len - batch_end) * _avg
        logger.info(
            "Row %d/%d done (%.1fs this batch, %.1fs avg/row, %.0fs elapsed, ~%.0fs remaining)",
            batch_end, process_len, _row_time, _avg, _elapsed, _remaining,
        )
        _last_log_time = _now

        if progress_cb:
            progress_cb(batch_end, process_len, f"Processed row {batch_end}/{process_len}")

        row_idx = batch_end

    prefetch_pool.shutdown(wait=False)

    # NOT_PROCESSED markers for remaining rows
    if BATCH_MODE is not None:
        for ri in range(process_len, original_len):
            row = df.iloc[ri]
            result_row = {"Respondent.Serial": row[id_col], "Q6": "NOT_PROCESSED"}
            completed_rows.append(result_row)
        _flush(completed_rows)

    return out_path
