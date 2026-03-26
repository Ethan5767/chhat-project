"""
Detailed diagnostic test: shows RF-DETR box count, per-crop DINO results,
and OCR results separately to understand the pipeline bottlenecks.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from pipeline import (
    get_device,
    load_dino,
    load_rfdetr,
    load_index,
    load_ocr,
    embed_images_batch,
    distance_to_confidence,
    _run_ocr_on_image,
    _ocr_brand_scores_from_items,
    _build_label_profiles,
    RFDETR_CONF_THRESHOLD,
    FAISS_TOP_K,
    MIN_OUTPUT_CONFIDENCE,
)

TEST_DIR = Path(__file__).resolve().parent / "test_images"


def main():
    device = get_device()
    print(f"Device: {device}")

    processor, model = load_dino(device)
    rfdetr_model = load_rfdetr()
    index, labels = load_index()
    label_profiles = _build_label_profiles(labels)

    test_images = sorted(
        f for f in TEST_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )

    for img_path in test_images:
        print(f"\n{'='*60}")
        print(f"IMAGE: {img_path.name}")
        print(f"{'='*60}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        print(f"Size: {w}x{h}")

        # Step 1: RF-DETR detection
        detections = rfdetr_model.predict(image, threshold=RFDETR_CONF_THRESHOLD)
        has_det = len(detections) > 0 if detections is not None else False
        num_boxes = len(detections) if has_det else 0
        print(f"\nRF-DETR boxes: {num_boxes} (threshold={RFDETR_CONF_THRESHOLD})")

        if has_det:
            for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
                x1, y1, x2, y2 = [int(v) for v in box]
                print(f"  Box {i}: [{x1},{y1},{x2},{y2}] conf={float(conf):.3f} size={x2-x1}x{y2-y1}")

        # Step 2: Create crops
        crops = []
        if has_det:
            for box in detections.xyxy:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crops.append(image.crop((x1, y1, x2, y2)))
        else:
            crops.append(image)

        # Step 3: DINO embedding + FAISS search
        all_vecs = embed_images_batch(crops, processor, model, device)
        k = min(FAISS_TOP_K, len(labels))

        print(f"\nDINO top-{k} per crop:")
        for ci in range(len(crops)):
            vec = all_vecs[ci].reshape(1, -1)
            distances, indices = index.search(vec, k=k)
            print(f"  Crop {ci}:")
            for rank in range(k):
                idx = int(indices[0][rank])
                if idx < 0 or idx >= len(labels):
                    continue
                label = labels[idx]
                dist = float(distances[0][rank])
                conf = distance_to_confidence(dist)
                brand = label.split('_')[0]
                print(f"    [{rank+1}] {label} (brand={brand}) conf={conf:.3f}")

        # Step 4: OCR
        print(f"\nOCR per crop:")
        for ci, crop in enumerate(crops):
            ocr_items = _run_ocr_on_image(crop)
            print(f"  Crop {ci}: {len(ocr_items)} text items")
            for item in ocr_items:
                if len(item) >= 3:
                    text = str(item[1]).strip()
                    conf = float(item[2])
                    if len(text) >= 3:
                        print(f"    '{text}' conf={conf:.3f}")

            # OCR brand scores
            scores = _ocr_brand_scores_from_items(ocr_items, label_profiles)
            top_ocr = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_ocr:
                print(f"    OCR brand matches:")
                for label, score in top_ocr:
                    print(f"      {label}: {score:.3f}")

        print()


if __name__ == "__main__":
    main()
