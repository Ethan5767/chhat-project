"""Fine-tune DINOv2 backbone + classifier head on reference images.

Unlike brand_classifier.py which freezes DINOv2, this unfreezes the last N
transformer layers so DINOv2 learns cigarette-pack-specific visual features.

Requires: 16GB+ VRAM (RunPod A6000/4090 recommended). Won't fit on 6GB.

Usage:
  python finetune_dinov2.py
  python finetune_dinov2.py --epochs 30 --unfreeze-layers 4 --lr 1e-5
  python finetune_dinov2.py --progress-file training_progress.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent
try:
    from backend.paths import CLASSIFIER_BASE_DIR, REFERENCES_DIR
except ImportError:
    from paths import CLASSIFIER_BASE_DIR, REFERENCES_DIR

OUTPUT_DIR = CLASSIFIER_BASE_DIR
DINO_MODEL_ID = "facebook/dinov2-base"
EMBED_DIM = 1536


def label_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


class ReferenceDataset(Dataset):
    """Dataset that loads and augments reference images on the fly."""

    def __init__(self, image_paths: list[Path], labels: list[int], processor, augment: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ]) if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Pad to square
        w, h = img.size
        if w != h:
            s = max(w, h)
            padded = Image.new("RGB", (s, s), (128, 128, 128))
            padded.paste(img, ((s - w) // 2, (s - h) // 2))
            img = padded

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        if self.augment and self.aug_transform:
            pixel_values = self.aug_transform(pixel_values)

        return pixel_values, self.labels[idx]


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone with classification head. Supports partial unfreezing."""

    def __init__(self, dino_model, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.dino = dino_model
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.dino(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        combined = torch.cat([cls_token, patch_mean], dim=1)
        return self.head(combined)

    def freeze_backbone(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze the last N transformer encoder layers."""
        self.freeze_backbone()

        # Unfreeze layernorm
        if hasattr(self.dino, 'layernorm'):
            for param in self.dino.layernorm.parameters():
                param.requires_grad = True

        # Unfreeze last N encoder layers
        if hasattr(self.dino, 'encoder') and hasattr(self.dino.encoder, 'layer'):
            total_layers = len(self.dino.encoder.layer)
            for i in range(max(0, total_layers - n), total_layers):
                for param in self.dino.encoder.layer[i].parameters():
                    param.requires_grad = True


def write_progress(progress_file: Path, data: dict):
    """Write progress to a JSON file for the frontend to poll."""
    if progress_file:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps(data, indent=2))


def train(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_file = Path(args.progress_file) if args.progress_file else None

    # 1. Collect images and labels from pack/ and box/ subfolders
    image_paths = []
    for pkg_type in ("pack", "box"):
        type_dir = REFERENCES_DIR / pkg_type
        if not type_dir.exists():
            print(f"Skipping missing subfolder: {type_dir}")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_paths.extend(type_dir.glob(ext))
            image_paths.extend(type_dir.glob(ext.upper()))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No reference images found in {REFERENCES_DIR}/pack/ or {REFERENCES_DIR}/box/")
        sys.exit(1)

    labels_str = [label_from_filename(p.name) for p in image_paths]
    unique_labels = sorted(set(labels_str))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_int = [label_to_idx[l] for l in labels_str]
    num_classes = len(unique_labels)

    print(f"Found {len(image_paths)} images across {num_classes} classes")

    # 2. Split
    try:
        train_idx, val_idx = train_test_split(
            range(len(image_paths)), test_size=0.15, random_state=42,
            stratify=labels_int
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            range(len(image_paths)), test_size=0.15, random_state=42
        )

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels_int[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels_int[i] for i in val_idx]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    # 3. Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cpu":
        print("WARNING: Fine-tuning DINOv2 on CPU will be very slow. Use a GPU (RunPod A6000 recommended).")

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_ID)

    model = DINOv2Classifier(dino_model, num_classes, hidden_dim=512).to(device)
    model.unfreeze_last_n_layers(args.unfreeze_layers)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")

    # 4. Data loaders
    train_dataset = ReferenceDataset(train_paths, train_labels, processor, augment=True)
    val_dataset = ReferenceDataset(val_paths, val_labels, processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 5. Optimizer - lower LR for backbone, higher for head
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 6. Train
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs (unfreezing last {args.unfreeze_layers} layers)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (pixel_values, targets) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += len(targets)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for pixel_values, targets in val_loader:
                pixel_values = pixel_values.to(device)
                targets = torch.tensor(targets, dtype=torch.long).to(device)
                logits = model(pixel_values)
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += len(targets)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_loss = train_loss / train_total if train_total > 0 else 0

        progress = {
            "epoch": epoch,
            "total_epochs": args.epochs,
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "train_loss": round(avg_loss, 4),
            "best_val_acc": round(best_val_acc, 4),
            "status": "training",
        }
        write_progress(progress_file, progress)

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best — extract just the head weights (named to avoid confusion with per-type classifiers)
            torch.save(model.head.state_dict(), OUTPUT_DIR / "dinov2_finetuned_head.pth")
            # Also save the full model (backbone + head) for future fine-tuning
            torch.save(model.state_dict(), OUTPUT_DIR / "dinov2_finetuned_full.pth")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 7. Save class mapping
    class_mapping = {
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(v): k for k, v in label_to_idx.items()},
        "num_classes": num_classes,
        "embed_dim": EMBED_DIM,
        "hidden_dim": 512,
        "finetuned": True,
        "unfreeze_layers": args.unfreeze_layers,
        "head_weights": "dinov2_finetuned_head.pth",
        "full_model_weights": "dinov2_finetuned_full.pth",
        "note": "Shared backbone. Retrain per-type classifiers (brand_classifier.py) after fine-tuning.",
    }
    with open(OUTPUT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    final_progress = {
        "epoch": epoch,
        "total_epochs": args.epochs,
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_acc": round(best_val_acc, 4),
        "status": "complete",
    }
    write_progress(progress_file, final_progress)

    print(f"\nTraining complete!")
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Head saved to: {OUTPUT_DIR / 'dinov2_finetuned_head.pth'}")
    print(f"  Full model saved to: {OUTPUT_DIR / 'dinov2_finetuned_full.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 brand classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (8 for 24GB, 4 for 16GB)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for backbone (head gets 10x)")
    parser.add_argument("--unfreeze-layers", type=int, default=4, help="Number of DINOv2 layers to unfreeze")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--progress-file", type=str, default="", help="Path to write progress JSON for UI polling")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
