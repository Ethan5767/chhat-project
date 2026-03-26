"""
Test detection pipeline on test_images/ and output detailed results.
Saves annotated results and a summary report.
"""

import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from pipeline import (
    get_device,
    load_dino,
    load_rfdetr,
    load_index,
    _detect_brands_from_image,
    MIN_OUTPUT_CONFIDENCE,
)

TEST_DIR = Path(__file__).resolve().parent / "test_images"
RESULTS_DIR = Path(__file__).resolve().parent / "test_results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    t0 = time.time()
    processor, model = load_dino(device)
    rfdetr_model = load_rfdetr()
    index, labels = load_index()
    print(f"Models loaded in {time.time() - t0:.1f}s")
    print(f"Index has {len(labels)} reference labels")

    test_images = sorted(
        f for f in TEST_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    print(f"\nFound {len(test_images)} test images\n")

    all_results = {}

    for img_path in test_images:
        print(f"--- {img_path.name} ---")
        t1 = time.time()

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        print(f"  Size: {w}x{h}")

        brand_scores = _detect_brands_from_image(
            image=image,
            rfdetr_model=rfdetr_model,
            processor=processor,
            model=model,
            device=device,
            index=index,
            labels=labels,
        )

        elapsed = time.time() - t1

        # Filter by minimum confidence and sort
        filtered = {k: v for k, v in brand_scores.items() if v >= MIN_OUTPUT_CONFIDENCE}
        sorted_brands = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        # Also show top unfiltered results for analysis
        all_sorted = sorted(brand_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Detected brands (conf >= {MIN_OUTPUT_CONFIDENCE}):")
        if sorted_brands:
            for brand, conf in sorted_brands:
                print(f"    {brand}: {conf:.3f}")
        else:
            print(f"    NO_DETECTION")

        print(f"  Top 10 raw scores (before filtering):")
        for brand, conf in all_sorted:
            marker = " *" if conf >= MIN_OUTPUT_CONFIDENCE else ""
            print(f"    {brand}: {conf:.3f}{marker}")

        all_results[img_path.name] = {
            "detected": {b: round(c, 3) for b, c in sorted_brands},
            "top_raw": {b: round(c, 3) for b, c in all_sorted},
            "time_seconds": round(elapsed, 1),
        }
        print()

    # Save results
    results_file = RESULTS_DIR / "detection_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")

    # Summary
    print("\n=== SUMMARY ===")
    total_detected = 0
    for name, result in all_results.items():
        n = len(result["detected"])
        total_detected += n
        brands_str = ", ".join(f"{b}({c:.2f})" for b, c in result["detected"].items()) or "NONE"
        print(f"  {name}: {n} brands - {brands_str}")
    print(f"\nTotal brands detected across all images: {total_detected}")


if __name__ == "__main__":
    main()
