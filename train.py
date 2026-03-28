"""Fine-tune RF-DETR-M on cigarette pack detection dataset.

Usage:
  python train.py
  python train.py --epochs 50 --batch-size 4 --lr 0.0001
  python train.py --progress-file training_progress.json
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from rfdetr import RFDETRMedium

DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "cigarette_packs"
OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
VALID_SPLIT_RATIO = 0.15


def _auto_split_train_valid(dataset_dir: Path, ratio: float = VALID_SPLIT_RATIO):
    """Split train/ into train/ + valid/ when valid/ is missing.

    Reads the COCO JSON from train/, randomly selects ~ratio of images for
    validation, moves those images + their annotations into valid/, and writes
    separate COCO JSONs for each split.
    """
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    ann_path = train_dir / "_annotations.coco.json"

    if not ann_path.exists():
        return

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if len(images) < 4:
        # Too few images to split -- copy train as valid
        valid_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ann_path, valid_dir / "_annotations.coco.json")
        for img in images:
            src = train_dir / img["file_name"]
            if src.exists():
                shutil.copy2(src, valid_dir / img["file_name"])
        print(f"  Auto-split: too few images ({len(images)}), copied train as valid")
        return

    # Shuffle and split
    random.seed(42)
    shuffled = list(images)
    random.shuffle(shuffled)
    n_valid = max(1, int(len(shuffled) * ratio))
    valid_images = shuffled[:n_valid]
    train_images = shuffled[n_valid:]

    valid_ids = {img["id"] for img in valid_images}
    train_ids = {img["id"] for img in train_images}

    valid_anns = [a for a in annotations if a["image_id"] in valid_ids]
    train_anns = [a for a in annotations if a["image_id"] in train_ids]

    base_meta = {k: v for k, v in coco.items() if k not in ("images", "annotations")}

    # Write valid split
    valid_dir.mkdir(parents=True, exist_ok=True)
    valid_coco = {**base_meta, "images": valid_images, "annotations": valid_anns}
    with (valid_dir / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(valid_coco, f, ensure_ascii=False)

    # Move valid images from train/ to valid/
    for img in valid_images:
        src = train_dir / img["file_name"]
        dst = valid_dir / img["file_name"]
        if src.exists():
            shutil.move(str(src), str(dst))

    # Rewrite train annotations (without the moved images)
    train_coco = {**base_meta, "images": train_images, "annotations": train_anns}
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)

    print(f"  Auto-split: {len(train_images)} train / {len(valid_images)} valid images "
          f"({len(train_anns)} / {len(valid_anns)} annotations)")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on cigarette pack detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--progress-file", type=str, default="")
    args = parser.parse_args()

    # Auto-create valid/ split if only train/ exists
    train_ann = DATASET_DIR / "train" / "_annotations.coco.json"
    valid_ann = DATASET_DIR / "valid" / "_annotations.coco.json"
    if train_ann.exists() and not valid_ann.exists():
        print("No valid/ split found. Auto-splitting from train/...")
        _auto_split_train_valid(DATASET_DIR)

    # Validate dataset exists before starting training
    for split in ("train", "valid"):
        ann_file = DATASET_DIR / split / "_annotations.coco.json"
        if not ann_file.exists():
            msg = (
                f"Missing dataset: {ann_file}\n"
                f"Upload a COCO dataset via /upload-coco or provide a Roboflow URL "
                f"when starting training."
            )
            print(f"ERROR: {msg}", flush=True)
            if args.progress_file:
                Path(args.progress_file).parent.mkdir(parents=True, exist_ok=True)
                Path(args.progress_file).write_text(json.dumps({
                    "epoch": 0, "total_epochs": args.epochs,
                    "status": "failed", "error": msg,
                }, indent=2))
            raise FileNotFoundError(msg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = RFDETRMedium()

    # Write initial progress
    if args.progress_file:
        Path(args.progress_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.progress_file).write_text(json.dumps({
            "epoch": 0, "total_epochs": args.epochs,
            "status": "starting",
        }, indent=2))

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=str(OUTPUT_DIR),
        early_stopping=True,
        early_stopping_patience=args.patience,
    )

    # Write final progress
    if args.progress_file:
        Path(args.progress_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.progress_file).write_text(json.dumps({
            "epoch": args.epochs, "total_epochs": args.epochs,
            "status": "complete",
        }, indent=2))

    print(f"Training complete. Checkpoints saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
