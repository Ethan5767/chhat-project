"""Test the classifier pipeline on the first 10 rows of the sample Excel.

Compares AI-detected brands against ground truth from the Excel file.
"""
import json
import sys
from pathlib import Path

import openpyxl
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from pipeline import (
    get_device,
    load_dino,
    load_rfdetr,
    load_classifier,
    _detect_brands_from_image,
    _format_brand_scores,
    label_to_product,
    download_image,
    load_index,
    MIN_OUTPUT_CONFIDENCE,
)

PROJECT_ROOT = Path(__file__).resolve().parent
EXCEL_PATH = PROJECT_ROOT / "source_materials" / "CHHAT_TC_Sample output Q12AB.xlsx"


def extract_ground_truth(row_values: list) -> dict:
    """Extract ground truth brands and SKUs from an Excel row."""
    # Row structure from exploration:
    # [0] = Serial, [1] = Q6, [2] = Q12A (brands pipe-separated),
    # [3..31] = individual brand columns (Q12A_1 through Q12A_29)
    # [32] = Q12B (SKUs pipe-separated)
    # [33..] = individual SKU columns

    serial = row_values[0]
    brands_raw = str(row_values[2]) if row_values[2] else ""

    # Parse brands from Q12A
    brands = []
    if brands_raw and brands_raw != "None":
        for part in brands_raw.split("|"):
            # Format: "MEVIUS_..." or just brand name
            brand = part.split("_")[0].strip()
            if brand:
                brands.append(brand.upper())

    # Parse SKUs from Q12B
    skus_raw = str(row_values[32]) if len(row_values) > 32 and row_values[32] else ""
    skus = []
    if skus_raw and skus_raw != "None":
        for part in skus_raw.split("|"):
            sku = part.strip()
            if sku:
                skus.append(sku.upper())

    # Collect photo URLs (Q30_1, Q30_2, Q30_3 = columns near the end)
    urls = []
    # Photo link columns are near the end
    for val in row_values:
        if val and str(val).startswith("http"):
            urls.append(str(val).strip())

    return {
        "serial": serial,
        "brands": brands,
        "skus": skus,
        "urls": urls,
    }


def normalize_brand(name: str) -> str:
    """Normalize a brand/SKU name for comparison."""
    name = name.upper().strip()
    # Map common variations
    mappings = {
        "MALBORO": "MARLBORO",
        "GOLD_SEA": "GOLD SEAL",
        "GOLD_SEAL": "GOLD SEAL",
        "COW_BOY": "COW BOY",
        "COCO_PALM": "COCO PALM",
    }
    for old, new in mappings.items():
        name = name.replace(old, new)
    # Replace underscores with spaces
    name = name.replace("_", " ")
    return name


def brand_from_product(product: str) -> str:
    """Extract parent brand from a product/SKU name.

    'mevius_original' -> 'MEVIUS'
    'esse_change' -> 'ESSE'
    '555_sphere2_velvet' -> '555'
    'cow_boy_bluberry_mint' -> 'COW BOY'
    """
    product = product.upper().replace("_", " ")

    # Multi-word brands
    multi_word = ["COW BOY", "COCO PALM", "GOLD SEAL", "GOLD SEA", "YUN YAN"]
    for mw in multi_word:
        if product.startswith(mw):
            return mw.replace("GOLD SEA", "GOLD SEAL")

    # Single-word brands
    parts = product.split()
    if parts:
        brand = parts[0]
        return brand.replace("MALBORO", "MARLBORO")
    return product


def main():
    print("Loading models...")
    device = get_device()
    load_classifier(device)
    classifier, labels = load_index()
    processor, model = load_dino(device)
    rfdetr_model = load_rfdetr()

    print(f"Device: {device}")
    print(f"Classifier classes: {len(labels)}")

    # Read Excel
    wb = openpyxl.load_workbook(str(EXCEL_PATH), read_only=True)
    ws = wb["Sample format Q12AB"]

    rows = []
    for i, row in enumerate(ws.iter_rows(min_row=3, max_row=12, values_only=True)):
        rows.append(list(row))

    print(f"\nTesting {len(rows)} rows...")
    print("=" * 80)

    total_gt_brands = 0
    total_detected_brands = 0
    correct_brands = 0

    for row_idx, row_values in enumerate(rows):
        gt = extract_ground_truth(row_values)
        print(f"\n--- Row {row_idx + 1} | Serial: {gt['serial']} ---")
        print(f"  Ground truth brands: {gt['brands']}")
        print(f"  Ground truth SKUs: {gt['skus'][:5]}{'...' if len(gt['skus']) > 5 else ''}")
        print(f"  Photo URLs: {len(gt['urls'])}")

        if not gt["urls"]:
            print("  SKIP: No photo URLs")
            continue

        # Run pipeline on each image
        all_detected: dict[str, float] = {}
        for url_idx, url in enumerate(gt["urls"]):
            image = download_image(url)
            if image is None:
                print(f"  Image {url_idx + 1}: DOWNLOAD FAILED")
                continue

            try:
                brand_scores = _detect_brands_from_image(
                    image=image,
                    rfdetr_model=rfdetr_model,
                    processor=processor,
                    model=model,
                    device=device,
                    index=classifier,
                    labels=labels,
                )
                for brand, conf in brand_scores.items():
                    if conf >= MIN_OUTPUT_CONFIDENCE:
                        if conf > all_detected.get(brand, 0.0):
                            all_detected[brand] = conf
            except Exception as exc:
                print(f"  Image {url_idx + 1}: ERROR - {exc}")

        # Compare
        detected_brands_set = set()
        for product, conf in sorted(all_detected.items(), key=lambda x: -x[1]):
            brand = brand_from_product(product)
            detected_brands_set.add(brand)
            print(f"  Detected: {product:30s} -> brand={brand:15s} conf={conf:.3f}")

        if not all_detected:
            print("  NO DETECTIONS")

        gt_brand_set = set(normalize_brand(b) for b in gt["brands"])
        detected_norm = set(normalize_brand(b) for b in detected_brands_set)

        matches = gt_brand_set & detected_norm
        missed = gt_brand_set - detected_norm
        false_pos = detected_norm - gt_brand_set

        total_gt_brands += len(gt_brand_set)
        correct_brands += len(matches)

        print(f"\n  Matches: {matches if matches else 'NONE'}")
        if missed:
            print(f"  Missed:  {missed}")
        if false_pos:
            print(f"  False+:  {false_pos}")
        print(f"  Score: {len(matches)}/{len(gt_brand_set)}")

    print("\n" + "=" * 80)
    print(f"OVERALL BRAND ACCURACY: {correct_brands}/{total_gt_brands} "
          f"= {correct_brands/total_gt_brands*100:.1f}%" if total_gt_brands > 0 else "N/A")


if __name__ == "__main__":
    main()
