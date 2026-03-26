"""Fine-tune RF-DETR-M on cigarette pack detection dataset."""
from pathlib import Path
from rfdetr import RFDETRMedium

DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "cigarette_packs"
OUTPUT_DIR = Path(__file__).resolve().parent / "runs"


def main():
    model = RFDETRMedium()

    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=50,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=str(OUTPUT_DIR),
        early_stopping=True,
        early_stopping_patience=10,
    )

    print(f"Training complete. Checkpoints saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
