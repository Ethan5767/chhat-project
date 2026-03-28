"""Fine-tune RF-DETR-M on cigarette pack detection dataset.

Usage:
  python train.py
  python train.py --epochs 50 --batch-size 4 --lr 0.0001
  python train.py --progress-file training_progress.json
"""
import argparse
import json
from pathlib import Path

from rfdetr import RFDETRMedium

DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "cigarette_packs"
OUTPUT_DIR = Path(__file__).resolve().parent / "runs"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on cigarette pack detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--progress-file", type=str, default="")
    args = parser.parse_args()

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
                Path(args.progress_file).write_text(json.dumps({
                    "epoch": 0, "total_epochs": args.epochs,
                    "status": "failed", "error": msg,
                }, indent=2))
            raise FileNotFoundError(msg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = RFDETRMedium()

    # Write initial progress
    if args.progress_file:
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
        Path(args.progress_file).write_text(json.dumps({
            "epoch": args.epochs, "total_epochs": args.epochs,
            "status": "complete",
        }, indent=2))

    print(f"Training complete. Checkpoints saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
