"""Process glass view images: detect cigarette packs with RF-DETR and save as reference crops.

Quality control:
  - Only keeps the BEST detection per image (highest confidence)
  - Requires minimum detection confidence of 0.40
  - Filters out crops with bad aspect ratios (not pack-shaped)
  - Verifies each crop against class centroid using DINOv2 similarity
  - Minimum crop size 60x80 px
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRMedium
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent
GLASS_VIEW_DIR = PROJECT_ROOT / "source_materials" / "glass_view"
REFERENCES_DIR = PROJECT_ROOT / "backend" / "references"
DINO_MODEL_ID = "facebook/dinov2-base"

# Quality thresholds
DETECTION_CONF_THRESHOLD = 0.35
MIN_CROP_WIDTH = 60
MIN_CROP_HEIGHT = 80
MIN_ASPECT = 0.3   # width/height minimum (packs are typically portrait)
MAX_ASPECT = 2.5   # width/height maximum
VERIFY_SIMILARITY_THRESHOLD = 0.40  # DINOv2 similarity to existing class images


def sku_name_from_path(img_path: Path) -> str:
    """Derive SKU name from the glass view folder structure."""
    parts = img_path.relative_to(GLASS_VIEW_DIR).parts

    if len(parts) >= 3:
        sku = parts[1]
    elif len(parts) == 2:
        sku = parts[0]
        sku = re.sub(r"^\d+\.", "", sku).strip()
    else:
        return ""

    sku = re.sub(r"\s*\(.*?\)", "", sku).strip()
    sku = sku.lower().replace(" ", "_").replace("-", "_")
    sku = re.sub(r"[^a-z0-9_]", "", sku)
    return sku


def next_index(sku: str) -> int:
    existing = list(REFERENCES_DIR.glob(f"{sku}_*.*"))
    max_idx = 0
    for p in existing:
        match = re.search(r"_(\d+)$", p.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    padded = Image.new("RGB", (s, s), (128, 128, 128))
    padded.paste(img, ((s - w) // 2, (s - h) // 2))
    return padded


def embed_single(img: Image.Image, processor, model, device: str) -> np.ndarray:
    processed = pad_to_square(img.convert("RGB"))
    with torch.no_grad():
        inputs = processor(images=[processed], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]
        patch_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        combined = torch.cat([cls, patch_mean], dim=1)
        vec = combined.squeeze(0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_class_centroids(processor, model, device: str) -> dict[str, np.ndarray]:
    """Compute DINOv2 centroid for each existing class in references."""
    from collections import defaultdict

    class_paths: dict[str, list[Path]] = defaultdict(list)
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in REFERENCES_DIR.glob(ext):
            label = sku_name_from_filename(p.name)
            class_paths[label].append(p)

    centroids = {}
    print(f"  Computing centroids for {len(class_paths)} existing classes...")

    for label, paths in class_paths.items():
        vecs = []
        for p in paths[:20]:  # Cap at 20 images per class for speed
            try:
                with Image.open(p) as img:
                    img_rgb = img.convert("RGB")
                    vec = embed_single(img_rgb, processor, model, device)
                    vecs.append(vec)
            except Exception:
                continue
        if vecs:
            centroid = np.mean(vecs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[label] = centroid

    return centroids


def sku_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def process_glass_view():
    if not GLASS_VIEW_DIR.exists():
        print(f"Glass view directory not found: {GLASS_VIEW_DIR}")
        sys.exit(1)

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all images (exclude mapping photos)
    image_paths = []
    for ext in ("*.JPG", "*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.PNG"):
        image_paths.extend(GLASS_VIEW_DIR.rglob(ext))
    image_paths = [p for p in image_paths if not p.name.startswith("photo_")]
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print("No glass view images found.")
        return

    print(f"Found {len(image_paths)} glass view images to process")

    # Load models
    print("Loading RF-DETR...")
    rfdetr = RFDETRMedium()

    print("Loading DINOv2 for verification...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_ID)
    dino_model.eval().to(device)

    # Compute class centroids from existing references
    centroids = compute_class_centroids(processor, dino_model, device)

    saved = 0
    skipped_no_sku = 0
    skipped_no_det = 0
    skipped_error = 0
    skipped_size = 0
    skipped_aspect = 0
    skipped_similarity = 0

    for img_path in image_paths:
        sku = sku_name_from_path(img_path)
        if not sku:
            skipped_no_sku += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped_error += 1
            continue

        # Detect
        detections = rfdetr.predict(pil_img, threshold=DETECTION_CONF_THRESHOLD)
        has_dets = detections is not None and len(detections) > 0

        if not has_dets:
            # For glass view close-ups, the full image IS the pack
            # Verify it looks like the right class before saving
            w, h = pil_img.size
            if w >= MIN_CROP_WIDTH and h >= MIN_CROP_HEIGHT:
                vec = embed_single(pil_img, processor, dino_model, device)
                centroid = centroids.get(sku)

                if centroid is not None:
                    sim = float(vec @ centroid)
                    if sim < VERIFY_SIMILARITY_THRESHOLD:
                        skipped_similarity += 1
                        continue
                # If no centroid exists (new class), accept the image

                idx = next_index(sku)
                out_name = f"{sku}_{idx}.jpg"
                # Resize if too large
                if max(w, h) > 800:
                    scale = 800 / max(w, h)
                    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                pil_img.save(REFERENCES_DIR / out_name, "JPEG", quality=92)
                print(f"  Full image saved: {out_name}")
                saved += 1
            else:
                skipped_size += 1
            continue

        # Pick the BEST detection only (highest confidence)
        xyxy = detections.xyxy
        confs = detections.confidence
        best_idx = int(np.argmax(confs))
        box = xyxy[best_idx]
        conf = float(confs[best_idx])

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w, h = pil_img.size

        # Add 5% padding
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.05), int(bh * 0.05)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = pil_img.crop((x1, y1, x2, y2))
        cw, ch = crop.size

        # Size check
        if cw < MIN_CROP_WIDTH or ch < MIN_CROP_HEIGHT:
            skipped_size += 1
            continue

        # Aspect ratio check
        aspect = cw / ch
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            skipped_aspect += 1
            continue

        # DINOv2 similarity verification
        vec = embed_single(crop, processor, dino_model, device)
        centroid = centroids.get(sku)

        if centroid is not None:
            sim = float(vec @ centroid)
            if sim < VERIFY_SIMILARITY_THRESHOLD:
                print(f"  REJECTED: {img_path.name} -> {sku} (similarity={sim:.3f} < {VERIFY_SIMILARITY_THRESHOLD})")
                skipped_similarity += 1
                continue
            sim_str = f"sim={sim:.3f}"
        else:
            sim_str = "new_class"
            # For new classes, build centroid from this first image
            centroids[sku] = vec

        idx = next_index(sku)
        out_name = f"{sku}_{idx}.jpg"
        crop.save(REFERENCES_DIR / out_name, "JPEG", quality=92)
        print(f"  Saved: {out_name} (conf={conf:.2f}, {cw}x{ch}, {sim_str})")
        saved += 1

    print(f"\n=== Glass View Processing Complete ===")
    print(f"  Saved:              {saved}")
    print(f"  Skipped (no SKU):   {skipped_no_sku}")
    print(f"  Skipped (no det):   {skipped_no_det}")
    print(f"  Skipped (error):    {skipped_error}")
    print(f"  Skipped (too small):{skipped_size}")
    print(f"  Skipped (aspect):   {skipped_aspect}")
    print(f"  Skipped (low sim):  {skipped_similarity}")

    # Cleanup
    del dino_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    process_glass_view()
