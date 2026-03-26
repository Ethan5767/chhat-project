"""Split the Roboflow COCO export into train/valid/test for RF-DETR."""
import json
import random
import shutil
from pathlib import Path

SEED = 42
TRAIN_RATIO, VALID_RATIO = 0.80, 0.10  # test gets remainder

SRC_DIR = Path(__file__).resolve().parent.parent / "Cigarette pack brand.coco" / "train"
DST_ROOT = Path(__file__).resolve().parent / "datasets" / "cigarette_packs"


def main():
    ann_path = SRC_DIR / "_annotations.coco.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Fix bbox string values
    for ann in coco["annotations"]:
        ann["bbox"] = [float(v) for v in ann["bbox"]]
        if "area" in ann:
            ann["area"] = float(ann["area"])

    images = coco["images"]
    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    splits = {
        "train": images[:n_train],
        "valid": images[n_train : n_train + n_valid],
        "test": images[n_train + n_valid :],
    }

    # Build image_id -> annotations map
    img_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    for split_name, split_images in splits.items():
        split_dir = DST_ROOT / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_anns = []
        for img in split_images:
            split_anns.extend(img_anns.get(img["id"], []))

        # Copy images
        for img in split_images:
            src = SRC_DIR / img["file_name"]
            dst = split_dir / img["file_name"]
            if src.exists():
                shutil.copy2(src, dst)

        # Write COCO JSON
        split_coco = {
            "images": split_images,
            "annotations": split_anns,
            "categories": coco["categories"],
        }
        if "info" in coco:
            split_coco["info"] = coco["info"]
        if "licenses" in coco:
            split_coco["licenses"] = coco["licenses"]

        out_path = split_dir / "_annotations.coco.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_coco, f, ensure_ascii=False)

        print(f"{split_name}: {len(split_images)} images, {len(split_anns)} annotations -> {out_path}")


if __name__ == "__main__":
    main()
