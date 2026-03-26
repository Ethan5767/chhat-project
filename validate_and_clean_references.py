"""Validate and clean reference images using DINOv2 similarity.

For each class:
  1. Compute DINOv2 embeddings for all images in that class
  2. Compute the class centroid (mean embedding)
  3. Flag images whose similarity to centroid is below threshold (outliers)
  4. Report class statistics

Also filters by:
  - Minimum size (30x30 px)
  - Aspect ratio sanity (not ultra-wide strips that are clearly not packs)
"""
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCES_DIR = PROJECT_ROOT / "backend" / "references"
QUARANTINE_DIR = PROJECT_ROOT / "backend" / "references_quarantined"
DINO_MODEL_ID = "facebook/dinov2-base"

# Thresholds
MIN_WIDTH = 40
MIN_HEIGHT = 40
MAX_ASPECT_RATIO = 5.0   # width/height or height/width > this = suspicious
SIMILARITY_THRESHOLD = 0.55  # cosine similarity to class centroid; below = outlier


def label_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    padded = Image.new("RGB", (s, s), (128, 128, 128))
    padded.paste(img, ((s - w) // 2, (s - h) // 2))
    return padded


def embed_batch(imgs: list[Image.Image], processor, model, device: str) -> np.ndarray:
    processed = [pad_to_square(img.convert("RGB")) for img in imgs]
    with torch.no_grad():
        inputs = processor(images=processed, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        patch_means = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        combined = torch.cat([cls_tokens, patch_means], dim=1)
        vecs = combined.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def main():
    print("=== Reference Image Validation ===\n")

    # Collect all images by class
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(REFERENCES_DIR.glob(ext))
        image_paths.extend(REFERENCES_DIR.glob(ext.upper()))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print("No reference images found.")
        return

    # Group by class
    class_images: dict[str, list[Path]] = defaultdict(list)
    for p in image_paths:
        label = label_from_filename(p.name)
        class_images[label].append(p)

    print(f"Total: {len(image_paths)} images across {len(class_images)} classes\n")

    # --- Pass 1: Size and aspect ratio filtering ---
    print("--- Pass 1: Size & aspect ratio filtering ---")
    to_quarantine: list[tuple[Path, str]] = []

    for label, paths in class_images.items():
        for p in paths:
            try:
                img = Image.open(p)
                w, h = img.size
                img.close()
            except Exception:
                to_quarantine.append((p, "cannot_open"))
                continue

            if w < MIN_WIDTH or h < MIN_HEIGHT:
                to_quarantine.append((p, f"too_small_{w}x{h}"))
                continue

            aspect = max(w / h, h / w)
            if aspect > MAX_ASPECT_RATIO:
                to_quarantine.append((p, f"bad_aspect_{w}x{h}_ratio_{aspect:.1f}"))

    if to_quarantine:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        for p, reason in to_quarantine:
            dest = QUARANTINE_DIR / p.name
            shutil.move(str(p), str(dest))
            print(f"  Quarantined: {p.name} ({reason})")
        # Rebuild class_images after removal
        for p, _ in to_quarantine:
            label = label_from_filename(p.name)
            if p in class_images[label]:
                class_images[label].remove(p)
    else:
        print("  All images pass size/aspect checks.")

    print(f"\n  Quarantined (pass 1): {len(to_quarantine)}")
    remaining = sum(len(v) for v in class_images.values())
    print(f"  Remaining: {remaining}\n")

    # --- Pass 2: DINOv2 similarity validation ---
    print("--- Pass 2: DINOv2 intra-class similarity ---")
    print("Loading DINOv2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModel.from_pretrained(DINO_MODEL_ID)
    model.eval().to(device)

    outliers: list[tuple[Path, str]] = []
    class_stats = []

    for label in sorted(class_images.keys()):
        paths = class_images[label]
        if len(paths) < 3:
            # Too few images to compute meaningful centroid
            class_stats.append((label, len(paths), 1.0, 1.0, "too_few"))
            continue

        # Embed all images for this class
        imgs = []
        valid_paths = []
        for p in paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception:
                outliers.append((p, "cannot_open"))

        if len(imgs) < 3:
            class_stats.append((label, len(imgs), 1.0, 1.0, "too_few_valid"))
            continue

        # Batch embed (in chunks of 8 for VRAM)
        all_vecs = []
        for i in range(0, len(imgs), 8):
            batch = imgs[i:i+8]
            vecs = embed_batch(batch, processor, model, device)
            all_vecs.append(vecs)
        all_vecs = np.vstack(all_vecs)

        # Compute centroid
        centroid = all_vecs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Compute similarities
        sims = all_vecs @ centroid
        mean_sim = float(sims.mean())
        min_sim = float(sims.min())

        # Flag outliers
        class_outlier_count = 0
        for j, (sim, p) in enumerate(zip(sims, valid_paths)):
            if sim < SIMILARITY_THRESHOLD:
                outliers.append((p, f"low_similarity_{sim:.3f}"))
                class_outlier_count += 1

        status = "OK" if class_outlier_count == 0 else f"{class_outlier_count}_outliers"
        class_stats.append((label, len(valid_paths), mean_sim, min_sim, status))

        # Close images
        for img in imgs:
            img.close()

    # Print class report
    print(f"\n{'Class':<30} {'Count':>5} {'Mean Sim':>8} {'Min Sim':>8} {'Status'}")
    print("-" * 75)
    for label, count, mean_s, min_s, status in sorted(class_stats, key=lambda x: x[3]):
        flag = " ***" if "outlier" in status else ""
        print(f"{label:<30} {count:>5} {mean_s:>8.3f} {min_s:>8.3f} {status}{flag}")

    # Move outliers to quarantine
    if outliers:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        for p, reason in outliers:
            if p.exists():
                dest = QUARANTINE_DIR / p.name
                shutil.move(str(p), str(dest))
                print(f"\n  Quarantined: {p.name} ({reason})")

    total_quarantined = len(to_quarantine) + len(outliers)
    final_count = remaining - len(outliers)
    print(f"\n=== Summary ===")
    print(f"  Original: {len(image_paths)}")
    print(f"  Quarantined (size/aspect): {len(to_quarantine)}")
    print(f"  Quarantined (similarity): {len(outliers)}")
    print(f"  Total quarantined: {total_quarantined}")
    print(f"  Clean references remaining: {final_count}")

    # Save report
    report = {
        "original_count": len(image_paths),
        "quarantined_size": len(to_quarantine),
        "quarantined_similarity": len(outliers),
        "final_count": final_count,
        "class_stats": [
            {"label": l, "count": c, "mean_sim": round(ms, 3), "min_sim": round(mis, 3), "status": s}
            for l, c, ms, mis, s in class_stats
        ],
    }
    report_path = PROJECT_ROOT / "backend" / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
