"""
Convert Full angle view HEIC images to cropped JPEG references.

Pipeline:
1. Walk the Full angle view directory structure
2. Convert HEIC -> PIL Image
3. Run RF-DETR detection, take top-1 box
4. Crop to the detected box
5. Save to backend/references_full_angle/ with naming convention matching existing references

Output naming: {brand}_{variant}_{n}.jpg
  e.g., mevius_freezy_dew_1.jpg, winston_night_blue_2.jpg
"""

import re
import sys
from pathlib import Path

import pillow_heif
from PIL import Image

# -- Register HEIC opener with Pillow --
pillow_heif.register_heif_opener()

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parent
FULL_ANGLE_DIR = PROJECT_ROOT / "Full angle view" / "Full angle view"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "references_full_angle"
RUNS_DIR = PROJECT_ROOT / "runs"

# Detection threshold -- using same as pipeline.py
RFDETR_CONF_THRESHOLD = 0.25

IMAGE_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _normalize_folder_to_label(brand_folder: str, variant_folder: str | None) -> str:
    """Convert folder names to the reference naming convention.

    e.g., '1.MEVIUS' + 'MEVIUS FREEZY DEW' -> 'mevius_freezy_dew'
         '5.555' + '555 ORIGINAL' -> '555_original'
    """
    # Strip leading number prefix like "1.", "10.", "19."
    brand = re.sub(r"^\d+\.\s*", "", brand_folder).strip()

    if variant_folder:
        # The variant folder often repeats the brand name, e.g., "MEVIUS OPTION PURPLE"
        variant = variant_folder.strip()
        # Remove the brand prefix from variant if present
        if variant.upper().startswith(brand.upper()):
            variant = variant[len(brand):].strip()
    else:
        variant = ""

    # Build label: brand + variant, lowercase, spaces to underscores
    parts = brand.lower()
    if variant:
        parts += "_" + variant.lower()

    # Clean: replace spaces with underscores, remove non-alphanumeric except underscores
    label = re.sub(r"\s+", "_", parts)
    label = re.sub(r"[^a-z0-9_]", "", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def collect_images() -> list[tuple[str, Path]]:
    """Walk the directory and return (label_base, image_path) pairs."""
    results = []
    if not FULL_ANGLE_DIR.exists():
        print(f"ERROR: {FULL_ANGLE_DIR} not found")
        sys.exit(1)

    for brand_dir in sorted(FULL_ANGLE_DIR.iterdir()):
        if not brand_dir.is_dir():
            continue

        # Check if brand_dir has sub-variant folders or direct images
        has_subdirs = any(p.is_dir() for p in brand_dir.iterdir())

        if has_subdirs:
            for variant_dir in sorted(brand_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                label_base = _normalize_folder_to_label(brand_dir.name, variant_dir.name)
                for img_path in sorted(variant_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        results.append((label_base, img_path))
        else:
            # Images directly in brand folder (no sub-variants)
            label_base = _normalize_folder_to_label(brand_dir.name, None)
            for img_path in sorted(brand_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    results.append((label_base, img_path))

    return results


def load_rfdetr():
    """Load RF-DETR model (same logic as pipeline.py)."""
    from rfdetr import RFDETRMedium

    # Find best checkpoint
    checkpoint = None
    if RUNS_DIR.exists():
        for pattern in ["best*.pth", "*.pth", "best*.pt", "*.pt"]:
            candidates = list(RUNS_DIR.rglob(pattern))
            if candidates:
                checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
                break

    if checkpoint:
        print(f"Loading fine-tuned RF-DETR from {checkpoint}")
        model = RFDETRMedium(pretrain_weights=str(checkpoint))
    else:
        print("No fine-tuned checkpoint, using pre-trained RF-DETR-M")
        model = RFDETRMedium()
    return model


def crop_top1(image: Image.Image, model) -> Image.Image:
    """Run RF-DETR, crop to highest-confidence detection. Falls back to full image."""
    detections = model.predict(image, threshold=RFDETR_CONF_THRESHOLD)

    has_detections = detections is not None and len(detections) > 0
    if not has_detections:
        return image

    # Top-1 by confidence
    confidences = detections.confidence
    best_idx = confidences.argmax()
    box = detections.xyxy[best_idx]

    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = image.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return image

    return image.crop((x1, y1, x2, y2))


def main():
    print("Collecting images from Full angle view...")
    images = collect_images()
    print(f"Found {len(images)} images across {len(set(lb for lb, _ in images))} label groups")

    if not images:
        print("No images found. Exiting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading RF-DETR model...")
    model = load_rfdetr()

    # Track counters per label for sequential numbering
    label_counters: dict[str, int] = {}
    total = len(images)
    saved = 0
    skipped = 0

    for i, (label_base, img_path) in enumerate(images, 1):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [{i}/{total}] SKIP (can't open): {img_path.name} -- {e}")
            skipped += 1
            continue

        cropped = crop_top1(img, model)

        # Sequential number per label
        label_counters[label_base] = label_counters.get(label_base, 0) + 1
        count = label_counters[label_base]
        out_name = f"{label_base}_{count}.jpg"
        out_path = OUTPUT_DIR / out_name

        cropped.save(out_path, "JPEG", quality=95)
        saved += 1

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] Processed -- saved {saved}, skipped {skipped}")

    print(f"\nDone! Saved {saved} cropped references to {OUTPUT_DIR}")
    print(f"Skipped {skipped} images")
    print(f"Unique labels: {len(label_counters)}")

    # Print label summary
    print("\nLabel summary:")
    for label, count in sorted(label_counters.items()):
        print(f"  {label}: {count} images")


if __name__ == "__main__":
    main()
