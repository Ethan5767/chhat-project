"""Train a DINOv2-based brand classifier on reference images.

Approach (optimized for 6GB VRAM):
  1. Scan references/ directory, parse filenames to get class labels
  2. Pre-compute DINOv2 embeddings for all images (frozen backbone, no gradient)
  3. Train a linear classifier head on the embeddings
  4. Save model weights + class mapping

Usage:
  python brand_classifier.py
  python brand_classifier.py --epochs 50 --lr 0.001
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCES_BASE_DIR = PROJECT_ROOT / "backend" / "references"
OUTPUT_BASE_DIR = PROJECT_ROOT / "backend" / "classifier_model"
DINO_MODEL_ID = "facebook/dinov2-base"
EMBED_DIM = 1536  # CLS (768) + mean-pooled patches (768)
PACKAGING_TYPES = ("pack", "box")


def label_from_filename(filename: str) -> str:
    """Extract class label from reference filename.

    'mevius_original_1.jpg'  -> 'mevius_original'
    '555_sphere2_velvet_12.jpg' -> '555_sphere2_velvet'
    'other_4_24.jpg' -> 'other_4'
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    max_side = max(w, h)
    padded = Image.new("RGB", (max_side, max_side), (128, 128, 128))
    padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return padded


def compute_embeddings(image_paths: list[Path], processor, model, device: str, batch_size: int = 16) -> tuple[np.ndarray, list[Path]]:
    """Pre-compute DINOv2 embeddings for all images. No gradient needed.

    Returns a tuple of (embeddings, valid_paths) where valid_paths contains only
    the paths that were successfully loaded. Corrupt or unreadable images are
    skipped so that embeddings and paths stay aligned.
    """
    all_vecs = []
    valid_paths: list[Path] = []
    model.eval()
    model.to(device)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        batch_valid_paths: list[Path] = []
        for p in batch_paths:
            try:
                img = pad_to_square(Image.open(p).convert("RGB"))
                imgs.append(img)
                batch_valid_paths.append(p)
            except Exception as exc:
                print(f"  Warning: skipping corrupt image {p}: {exc}")

        if not imgs:
            continue

        with torch.no_grad():
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            patch_means = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
            combined = torch.cat([cls_tokens, patch_means], dim=1)  # (N, 1536)
            vecs = combined.cpu().numpy().astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms
        all_vecs.append(vecs)
        valid_paths.extend(batch_valid_paths)

        print(f"  Embedded {len(valid_paths)}/{len(image_paths)} images", end="\r")

    print()
    return np.vstack(all_vecs), valid_paths


class BrandClassifier(nn.Module):
    """Simple linear classifier on top of frozen DINOv2 embeddings."""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(args):
    pkg_type = getattr(args, "packaging_type", "pack")
    REFERENCES_DIR = REFERENCES_BASE_DIR / pkg_type
    OUTPUT_DIR = OUTPUT_BASE_DIR / pkg_type
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect reference images and labels
    print(f"Scanning {pkg_type} reference images from {REFERENCES_DIR}...")
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(REFERENCES_DIR.glob(ext))
        image_paths.extend(REFERENCES_DIR.glob(ext.upper()))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No reference images found in {REFERENCES_DIR}")
        sys.exit(1)

    # Parse labels
    labels_str = [label_from_filename(p.name) for p in image_paths]
    unique_labels = sorted(set(labels_str))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_int = [label_to_idx[l] for l in labels_str]

    num_classes = len(unique_labels)
    print(f"Found {len(image_paths)} images across {num_classes} classes (pre-filtering)")

    # Show class distribution (before corrupt-image filtering)
    from collections import Counter
    dist = Counter(labels_str)
    print("\nClass distribution:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    print()

    # 2. Pre-compute DINOv2 embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA not available. Training on CPU is not supported (too slow). Exiting.")
        sys.exit(1)
    print(f"Computing DINOv2 embeddings on {device}...")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModel.from_pretrained(DINO_MODEL_ID)
    dino_model.eval()

    embeddings, valid_paths = compute_embeddings(image_paths, processor, dino_model, device, batch_size=args.embed_batch_size)

    skipped = len(image_paths) - len(valid_paths)
    if skipped > 0:
        print(f"  Skipped {skipped} corrupt/unreadable image(s); training on {len(valid_paths)} images")

    # Re-derive labels from the paths that were actually loaded (parallel with embeddings)
    valid_labels_str = [label_from_filename(p.name) for p in valid_paths]
    unique_labels = sorted(set(valid_labels_str))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_int = [label_to_idx[l] for l in valid_labels_str]
    num_classes = len(unique_labels)

    # Free DINOv2 from GPU
    del dino_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. Train/val split
    X = embeddings
    y = np.array(labels_int)

    # Stratified split - but handle classes with only 1 sample
    if len(set(labels_int)) > 1:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
        except ValueError:
            # Some classes have too few samples for stratified split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42
            )
    else:
        X_train, X_val, y_train, y_val = X, X, y, y

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # 4. Create data loaders (operating on pre-computed embeddings, very lightweight)
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Train classifier head
    classifier = BrandClassifier(EMBED_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nTraining classifier for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)
        scheduler.step()

        # Validate
        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = classifier(batch_x)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_loss = train_loss / train_total if train_total > 0 else 0

        # Write progress for UI polling
        if hasattr(args, 'progress_file') and args.progress_file:
            Path(args.progress_file).parent.mkdir(parents=True, exist_ok=True)
            Path(args.progress_file).write_text(json.dumps({
                "epoch": epoch, "total_epochs": args.epochs,
                "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
                "train_loss": round(avg_loss, 4), "best_val_acc": round(best_val_acc, 4),
                "status": "training",
            }, indent=2))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(classifier.state_dict(), OUTPUT_DIR / "best_classifier.pth")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # 6. Save final model + class mapping
    torch.save(classifier.state_dict(), OUTPUT_DIR / "classifier.pth")

    class_mapping = {
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(v): k for k, v in label_to_idx.items()},
        "num_classes": num_classes,
        "embed_dim": EMBED_DIM,
        "hidden_dim": 512,
        "packaging_type": pkg_type,
    }
    with open(OUTPUT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    print(f"\nTraining complete ({pkg_type} classifier)!")
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print(f"  Classes: {num_classes}")
    print(f"  Best model: {OUTPUT_DIR / 'best_classifier.pth'}")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 brand classifier")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for classifier training")
    parser.add_argument("--embed-batch-size", type=int, default=8, help="Batch size for DINOv2 embedding (lower for low VRAM)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--progress-file", type=str, default="", help="Path to write progress JSON for UI polling")
    parser.add_argument("--packaging-type", type=str, default="pack", choices=["pack", "box"], help="Which packaging type to train classifier for")
    args = parser.parse_args()
    train_classifier(args)


if __name__ == "__main__":
    main()
