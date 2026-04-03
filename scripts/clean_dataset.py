"""Clean a COCO dataset for RF-DETR training.

Operations:
  1. Remove annotations with area < 100 pixels
  2. Remove excess empty images (keep ~10% as hard negatives)
  3. Merge all categories into a single 'cigarette_product' class (id=0)
  4. Re-index annotation IDs to be contiguous (1-based)
  5. Fix bbox string values (Roboflow exports have strings instead of floats)

Usage:
  python scripts/clean_dataset.py path/to/_annotations.coco.json
  python scripts/clean_dataset.py datasets/cigarette_packs/train/_annotations.coco.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

DEFAULT_MIN_AREA = 100
DEFAULT_NEGATIVE_RATIO = 0.10
SEED = 42


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(coco: dict, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def print_stats(coco: dict, label: str, min_area: float = DEFAULT_MIN_AREA):
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_ids_with_anns = {a["image_id"] for a in annotations}
    empty_count = sum(1 for img in images if img["id"] not in image_ids_with_anns)

    print(f"\n--- {label} ---")
    print(f"  Images:      {len(images)} ({empty_count} empty)")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Categories:  {[c['name'] for c in categories]}")

    if annotations:
        areas = [float(a.get("area", 0)) for a in annotations]
        print(f"  Area range:  {min(areas):.1f} - {max(areas):.1f}")
        tiny = sum(1 for a in areas if a < min_area)
        print(f"  Tiny (<{min_area}px): {tiny}")


def clean(coco: dict, min_area: float = DEFAULT_MIN_AREA,
          negative_ratio: float = DEFAULT_NEGATIVE_RATIO) -> dict:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # --- Step 1: Fix bbox/area string values ---
    for ann in annotations:
        ann["bbox"] = [float(v) for v in ann["bbox"]]
        if "area" in ann:
            ann["area"] = float(ann["area"])
        else:
            # Compute area from bbox if missing
            _, _, w, h = ann["bbox"]
            ann["area"] = w * h

    # --- Step 2: Remove tiny annotations ---
    before_tiny = len(annotations)
    annotations = [a for a in annotations if a["area"] >= min_area]
    removed_tiny = before_tiny - len(annotations)
    if removed_tiny:
        print(f"  Removed {removed_tiny} annotations with area < {min_area}")

    # --- Step 3: Merge all categories into single class ---
    old_cat_names = [c["name"] for c in categories]
    new_categories = [{"id": 0, "name": "cigarette_product", "supercategory": "none"}]

    # Map all old category IDs to 0
    for ann in annotations:
        ann["category_id"] = 0

    if len(old_cat_names) > 1:
        print(f"  Merged categories {old_cat_names} -> ['cigarette_product']")
    elif old_cat_names != ["cigarette_product"]:
        print(f"  Renamed category {old_cat_names} -> ['cigarette_product']")

    # --- Step 4: Remove excess empty images ---
    image_ids_with_anns = {a["image_id"] for a in annotations}
    images_with_anns = [img for img in images if img["id"] in image_ids_with_anns]
    images_empty = [img for img in images if img["id"] not in image_ids_with_anns]

    n_annotated = len(images_with_anns)
    target_empty = max(1, int(n_annotated * negative_ratio))
    target_empty = min(target_empty, len(images_empty))  # can't keep more than we have

    if len(images_empty) > target_empty:
        random.seed(SEED)
        kept_empty = random.sample(images_empty, target_empty)
        removed_empty = len(images_empty) - target_empty
        print(f"  Removed {removed_empty} excess empty images "
              f"(kept {target_empty} as hard negatives)")
    else:
        kept_empty = images_empty

    images = images_with_anns + kept_empty

    # --- Step 5: Re-index annotation IDs (1-based contiguous) ---
    for new_id, ann in enumerate(annotations, start=1):
        ann["id"] = new_id

    return {
        **{k: v for k, v in coco.items() if k not in ("images", "annotations", "categories")},
        "images": images,
        "annotations": annotations,
        "categories": new_categories,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean COCO dataset for RF-DETR training")
    parser.add_argument("input", type=str, help="Path to COCO JSON file")
    parser.add_argument("--output", type=str, default="",
                        help="Output path (default: appends _cleaned to filename)")
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA,
                        help=f"Minimum annotation area in pixels (default: {DEFAULT_MIN_AREA})")
    parser.add_argument("--negative-ratio", type=float, default=DEFAULT_NEGATIVE_RATIO,
                        help=f"Ratio of hard negatives to keep (default: {DEFAULT_NEGATIVE_RATIO})")
    args = parser.parse_args()

    min_area = args.min_area
    negative_ratio = args.negative_ratio

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_cleaned")

    coco = load_coco(input_path)
    print_stats(coco, "BEFORE", min_area=min_area)

    cleaned = clean(coco, min_area=min_area, negative_ratio=negative_ratio)
    print_stats(cleaned, "AFTER", min_area=min_area)

    save_coco(cleaned, output_path)
    print(f"\nSaved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
