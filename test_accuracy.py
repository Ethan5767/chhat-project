"""
Test detection pipeline on test_images/ and show per-image results
with annotated images saved showing bounding boxes and brand labels.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from pipeline import (
    get_device,
    load_dino,
    load_rfdetr,
    load_index,
    load_ocr,
    embed_images_batch,
    distance_to_confidence,
    _detect_brands_from_image,
    _build_label_profiles,
    _run_ocr_on_image,
    _ocr_brand_scores_from_items,
    RFDETR_CONF_THRESHOLD,
    FAISS_TOP_K,
    MIN_OUTPUT_CONFIDENCE,
)

TEST_DIR = Path(__file__).resolve().parent / "test_images"
RESULTS_DIR = Path(__file__).resolve().parent / "test_results"


def get_brand_family(label: str) -> str:
    """Extract brand family from a label like 'malboro_red_3' -> 'malboro'."""
    parts = label.split("_")
    return parts[0] if parts else label


def detect_with_boxes(image, rfdetr_model, processor, model, device, index, labels):
    """Run detection and return per-box brand results + overall fused results."""
    from pipeline import (
        _build_label_profiles,
        _run_ocr_on_image,
        _ocr_brand_scores_from_items,
        embed_images_batch,
        distance_to_confidence,
    )

    label_profiles = _build_label_profiles(labels)
    label_profile_map = {p["label"]: p for p in label_profiles}

    detections = rfdetr_model.predict(image, threshold=RFDETR_CONF_THRESHOLD)
    has_det = len(detections) > 0 if detections is not None else False
    num_boxes = len(detections) if has_det else 0

    width, height = image.size
    box_results = []

    if has_det:
        crops = []
        boxes = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(1, min(x2, width))
            y2 = max(1, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            boxes.append((x1, y1, x2, y2))

        # DINO batch embed
        all_vecs = embed_images_batch(crops, processor, model, device)
        k = min(FAISS_TOP_K, len(labels))

        for ci in range(len(crops)):
            vec = all_vecs[ci].reshape(1, -1)
            distances, indices = index.search(vec, k=k)
            top_label = ""
            top_conf = 0.0
            for rank in range(k):
                idx = int(indices[0][rank])
                if 0 <= idx < len(labels):
                    label = labels[idx]
                    conf = distance_to_confidence(float(distances[0][rank]))
                    if conf > top_conf:
                        top_conf = conf
                        top_label = label

            brand = get_brand_family(top_label)
            box_results.append({
                "box": boxes[ci],
                "top_label": top_label,
                "brand": brand,
                "dino_conf": round(top_conf, 3),
            })

    # Get the full fused results
    fused = _detect_brands_from_image(
        image=image,
        rfdetr_model=rfdetr_model,
        processor=processor,
        model=model,
        device=device,
        index=index,
        labels=labels,
    )

    return num_boxes, box_results, fused


def draw_results(image, box_results, fused, output_path):
    """Draw bounding boxes and labels on the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    # Color palette for brand families
    colors = [
        "#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF",
        "#44FFFF", "#FF8844", "#88FF44", "#4488FF", "#FF4488",
        "#88FFFF", "#FFFF88", "#FF88FF", "#88FF88", "#8888FF",
    ]
    brand_colors = {}
    color_idx = 0

    for br in box_results:
        brand = br["brand"]
        if brand not in brand_colors:
            brand_colors[brand] = colors[color_idx % len(colors)]
            color_idx += 1

        x1, y1, x2, y2 = br["box"]
        color = brand_colors[brand]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label_text = f"{brand} ({br['dino_conf']:.2f})"
        # Background for text
        bbox = draw.textbbox((x1, y1 - 16), label_text, font=font_small)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
        draw.text((x1, y1 - 16), label_text, fill="black", font=font_small)

    img.save(output_path, quality=95)
    return output_path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    print("Loading models...")
    processor, model = load_dino(device)
    rfdetr_model = load_rfdetr()
    index, labels = load_index()
    print(f"Index: {len(labels)} reference labels")

    # Get unique brand families
    all_families = sorted(set(get_brand_family(l) for l in labels))
    print(f"Brand families: {len(all_families)} - {', '.join(all_families)}")

    test_images = sorted(
        f for f in TEST_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    print(f"\n{'='*70}")
    print(f"TESTING {len(test_images)} IMAGES")
    print(f"{'='*70}\n")

    all_results = {}

    for img_path in test_images:
        print(f"--- {img_path.name} ---")
        t0 = time.time()

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        num_boxes, box_results, fused = detect_with_boxes(
            image, rfdetr_model, processor, model, device, index, labels
        )
        elapsed = time.time() - t0

        # Filter fused results
        detected = {k: v for k, v in fused.items() if v >= MIN_OUTPUT_CONFIDENCE}
        sorted_detected = sorted(detected.items(), key=lambda x: x[1], reverse=True)

        # Aggregate to brand families
        family_best = {}
        for label, conf in sorted_detected:
            family = get_brand_family(label)
            if conf > family_best.get(family, 0.0):
                family_best[family] = conf
        family_sorted = sorted(family_best.items(), key=lambda x: x[1], reverse=True)

        # Per-box brand distribution
        box_brand_counts = {}
        for br in box_results:
            b = br["brand"]
            box_brand_counts[b] = box_brand_counts.get(b, 0) + 1

        print(f"  Size: {w}x{h} | RF-DETR boxes: {num_boxes} | Time: {elapsed:.1f}s")
        print(f"  Detected brand families (conf >= {MIN_OUTPUT_CONFIDENCE}):")
        if family_sorted:
            for family, conf in family_sorted:
                box_count = box_brand_counts.get(family, 0)
                print(f"    {family}: {conf:.3f} ({box_count} boxes)")
        else:
            print(f"    NO_DETECTION")

        # Draw annotated image
        out_path = RESULTS_DIR / f"annotated_{img_path.name}"
        draw_results(image, box_results, fused, out_path)
        print(f"  Annotated: {out_path.name}")

        all_results[img_path.name] = {
            "size": f"{w}x{h}",
            "rf_detr_boxes": num_boxes,
            "time_seconds": round(elapsed, 1),
            "brand_families": {f: round(c, 3) for f, c in family_sorted},
            "all_detections": {k: round(v, 3) for k, v in sorted_detected},
            "box_brand_distribution": box_brand_counts,
        }
        print()

    # Save JSON results
    results_file = RESULTS_DIR / "accuracy_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Final summary
    print(f"{'='*70}")
    print("ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Image':<45} {'Boxes':>5} {'Families':>8} {'Top Brand':<15} {'Conf':>6} {'Time':>5}")
    print(f"{'-'*45} {'-'*5} {'-'*8} {'-'*15} {'-'*6} {'-'*5}")

    total_families = 0
    total_boxes = 0
    for name, r in all_results.items():
        families = r["brand_families"]
        n_fam = len(families)
        total_families += n_fam
        total_boxes += r["rf_detr_boxes"]
        top = list(families.items())[0] if families else ("NONE", 0)
        print(f"  {name:<43} {r['rf_detr_boxes']:>5} {n_fam:>8} {top[0]:<15} {top[1]:>6.3f} {r['time_seconds']:>4.1f}s")

    print(f"\n  Total boxes detected: {total_boxes}")
    print(f"  Total brand families found: {total_families}")
    print(f"  Unique brand families across all images:")

    all_detected_families = {}
    for r in all_results.values():
        for fam, conf in r["brand_families"].items():
            if conf > all_detected_families.get(fam, 0):
                all_detected_families[fam] = conf
    for fam, conf in sorted(all_detected_families.items(), key=lambda x: -x[1]):
        print(f"    {fam}: best {conf:.3f}")

    print(f"\n  Results saved to: {results_file}")
    print(f"  Annotated images saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
