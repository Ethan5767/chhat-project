"""Merge multiple COCO datasets from source_materials/rf-detr-v11 into one training set.

Merges:
  - rf-detr-v11/train/ (833 images)
  - rf-detr-v11/Cigarette pack brand.coco (6)/train/ (316 images, newer annotations)

Output: datasets/cigarette_packs/train/ with merged _annotations.coco.json
The train.py auto-split will handle creating the valid/ split.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "source_materials" / "rf-detr-v11"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "cigarette_packs"


def load_coco(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def fix_bbox(bbox: list) -> list:
    """Fix Roboflow string bbox values -> float."""
    return [float(v) for v in bbox]


def merge():
    sources = [
        ("root", SOURCE_DIR / "train"),
        ("coco6", SOURCE_DIR / "Cigarette pack brand.coco (6)" / "train"),
    ]

    # Load and validate all sources
    datasets = []
    for name, src_dir in sources:
        ann_path = src_dir / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"  SKIP {name}: {ann_path} not found")
            continue
        coco = load_coco(ann_path)
        print(f"  {name}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
        datasets.append((name, src_dir, coco))

    # Unify categories (all should be the same)
    categories = datasets[0][2]["categories"]
    for name, _, coco in datasets[1:]:
        src_cats = {c["id"]: c["name"] for c in coco["categories"]}
        ref_cats = {c["id"]: c["name"] for c in categories}
        if src_cats != ref_cats:
            print(f"  WARNING: {name} has different categories: {src_cats} vs {ref_cats}")
            # Map by name
            name_to_id = {c["name"]: c["id"] for c in categories}
            src_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
            cat_remap = {}
            for cat_name, src_id in src_name_to_id.items():
                if cat_name in name_to_id:
                    cat_remap[src_id] = name_to_id[cat_name]
                else:
                    print(f"    NEW category from {name}: {cat_name}")
                    new_id = max(c["id"] for c in categories) + 1
                    categories.append({"id": new_id, "name": cat_name, "supercategory": "none"})
                    cat_remap[src_id] = new_id
            # Remap annotations
            for ann in coco["annotations"]:
                ann["category_id"] = cat_remap.get(ann["category_id"], ann["category_id"])

    # Prepare output directory
    train_dir = OUTPUT_DIR / "train"
    # Remove old valid dir so train.py regenerates the split
    valid_dir = OUTPUT_DIR / "valid"
    if valid_dir.exists():
        shutil.rmtree(valid_dir)
        print(f"  Removed old valid/ dir (train.py will re-split)")

    train_dir.mkdir(parents=True, exist_ok=True)

    # Merge images and annotations, tracking by filename to deduplicate
    merged_images = {}  # file_name -> image dict
    merged_annotations = []
    seen_files = set()
    next_image_id = 0
    next_ann_id = 0

    for name, src_dir, coco in datasets:
        old_to_new_img_id = {}
        for img in coco["images"]:
            fname = img["file_name"]
            if fname in seen_files:
                # Already merged from a previous source, skip
                continue
            seen_files.add(fname)

            # Copy image file
            src_file = src_dir / fname
            dst_file = train_dir / fname
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)

            old_to_new_img_id[img["id"]] = next_image_id
            merged_images[fname] = {
                **img,
                "id": next_image_id,
            }
            next_image_id += 1

        for ann in coco["annotations"]:
            if ann["image_id"] not in old_to_new_img_id:
                continue  # Image was deduplicated out
            merged_annotations.append({
                **ann,
                "id": next_ann_id,
                "image_id": old_to_new_img_id[ann["image_id"]],
                "bbox": fix_bbox(ann["bbox"]),
            })
            next_ann_id += 1

        added = len(old_to_new_img_id)
        print(f"  Added {added} images from {name}")

    # Build merged COCO JSON
    merged_coco = {
        "info": datasets[0][2].get("info", {}),
        "licenses": datasets[0][2].get("licenses", []),
        "categories": categories,
        "images": list(merged_images.values()),
        "annotations": merged_annotations,
    }

    ann_out = train_dir / "_annotations.coco.json"
    with open(ann_out, "w", encoding="utf-8") as f:
        json.dump(merged_coco, f, ensure_ascii=False)

    print(f"\nMerged dataset: {len(merged_images)} images, {len(merged_annotations)} annotations")
    print(f"Output: {ann_out}")
    print(f"Categories: {[c['name'] for c in categories]}")
    print(f"\nRun train.py to auto-split into train/valid and start training.")


if __name__ == "__main__":
    print("Merging rf-detr-v11 datasets...")
    merge()
