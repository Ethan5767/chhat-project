# Pack/Box Packaging Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pack/box packaging type support so RF-DETR detects boxes vs packs separately, reference images are organized by type in subfolders, and each type gets its own classifier -- while final output still maps to the same brand/product.

**Architecture:** Reference images move from flat `backend/references/` to `backend/references/pack/` and `backend/references/box/` subfolders. Two independent brand classifiers are trained (one per packaging type). RF-DETR detection gains a `class_id` distinguishing pack vs box. During inference, pack crops route to the pack classifier, box crops route to the box classifier. The label crop UI adds a pack/box toggle. All downstream output (Q12A/Q12B) remains unchanged.

**Tech Stack:** Python, FastAPI, Streamlit, RF-DETR, DINOv2, PyTorch

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `backend/brand_registry.py` | Update `REFERENCES_DIR`, `audit_references()`, `_label_from_filename()` to handle `pack/` and `box/` subfolders |
| Modify | `backend/pipeline.py` | Dual classifier loading, type-aware classification routing, updated `load_classifier()`, `classify_embeddings()`, `_detect_brands_from_image()` |
| Modify | `backend/main.py` | Update `/add-reference`, `/generate-crops`, `/detect-single`, `/build-index`, `/brand-registry`, `/reference-images/`, `/reference-image/`, `/delete-reference-image/` endpoints for pack/box |
| Modify | `brand_classifier.py` | Accept `--packaging-type` arg, scan correct subfolder, save to type-specific output dir |
| Modify | `frontend/app.py` | Add pack/box radio button to label crop cards, update reference viewer to show type, update API calls |
| Create | `migrate_references.py` | One-time script to move existing flat references into `pack/` subfolder |

---

### Task 1: Migration Script -- Move Existing References to `pack/` Subfolder

**Files:**
- Create: `migrate_references.py`

This must run first so existing references are preserved. All current ~385 images are packs.

- [ ] **Step 1: Write the migration script**

```python
"""One-time migration: move flat backend/references/*.jpg into backend/references/pack/."""
from pathlib import Path
import shutil

REFERENCES_DIR = Path(__file__).resolve().parent / "backend" / "references"
PACK_DIR = REFERENCES_DIR / "pack"
BOX_DIR = REFERENCES_DIR / "box"

def migrate():
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    BOX_DIR.mkdir(parents=True, exist_ok=True)

    moved = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        for img_path in list(REFERENCES_DIR.glob(ext)):
            # Skip if already in a subfolder
            if img_path.parent != REFERENCES_DIR:
                continue
            dest = PACK_DIR / img_path.name
            shutil.move(str(img_path), str(dest))
            moved += 1

    # Also handle uppercase extensions
    for ext in ("*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.BMP"):
        for img_path in list(REFERENCES_DIR.glob(ext)):
            if img_path.parent != REFERENCES_DIR:
                continue
            dest = PACK_DIR / img_path.name
            shutil.move(str(img_path), str(dest))
            moved += 1

    print(f"Migrated {moved} images to {PACK_DIR}")
    print(f"Created empty {BOX_DIR} for future box references")

if __name__ == "__main__":
    migrate()
```

- [ ] **Step 2: Run the migration**

Run: `python migrate_references.py`
Expected: "Migrated ~385 images to backend/references/pack/"

- [ ] **Step 3: Verify the migration**

Run: `ls backend/references/pack/ | head -10 && ls backend/references/box/ && ls backend/references/*.jpg 2>/dev/null | wc -l`
Expected: pack/ has images, box/ is empty, root has 0 images

- [ ] **Step 4: Commit**

```bash
git add migrate_references.py
git commit -m "Add migration script for pack/box reference subfolder structure"
```

---

### Task 2: Update `brand_registry.py` -- Subfolder-Aware Reference Scanning

**Files:**
- Modify: `backend/brand_registry.py:12` (REFERENCES_DIR), `:164-169` (_label_from_filename), `:214-276` (audit_references)

- [ ] **Step 1: Update REFERENCES_DIR and add PACKAGING_TYPES constant**

In `backend/brand_registry.py`, replace line 12:

```python
REFERENCES_DIR = Path(__file__).resolve().parent / "references"
```

with:

```python
REFERENCES_DIR = Path(__file__).resolve().parent / "references"
PACKAGING_TYPES = ("pack", "box")
```

- [ ] **Step 2: Update `audit_references()` to scan subfolders**

Replace the `audit_references` function (lines 214-276) with:

```python
def audit_references() -> dict:
    """Check what's in backend/references/{pack,box}/ vs what the registry expects.

    Returns a dict with:
      - found: {internal_name: {"pack": count, "box": count}} for products with reference images
      - missing: [(brand, product_display, internal_name)] for products with 0 images in either type
      - unregistered: {filename_prefix: count} for images not in registry
      - legacy: {old_name: new_name} for images using old naming
      - per_type: {"pack": total_count, "box": total_count}
    """
    if not REFERENCES_DIR.exists():
        return {"found": {}, "missing": [], "unregistered": {}, "legacy": {}, "per_type": {}}

    all_found: dict[str, dict[str, int]] = {}
    total_images = 0
    all_label_counts = Counter()
    per_type_totals = {}

    for pkg_type in PACKAGING_TYPES:
        type_dir = REFERENCES_DIR / pkg_type
        if not type_dir.exists():
            per_type_totals[pkg_type] = 0
            continue

        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_paths.extend(type_dir.glob(ext))
            image_paths.extend(type_dir.glob(ext.upper()))

        per_type_totals[pkg_type] = len(image_paths)
        total_images += len(image_paths)

        file_labels = [_label_from_filename(p.name) for p in image_paths]
        label_counts = Counter(file_labels)
        all_label_counts += label_counts

        for label, count in label_counts.items():
            if label not in all_found:
                all_found[label] = {t: 0 for t in PACKAGING_TYPES}
            all_found[label][pkg_type] = count

    # Check each registry entry
    found = {}
    missing = []
    for brand, products in BRAND_REGISTRY.items():
        for product_display, internal in products:
            counts = all_found.get(internal, {t: 0 for t in PACKAGING_TYPES})
            # Also check legacy aliases
            for old, new in LEGACY_ALIASES.items():
                if new == internal and old in all_found:
                    for t in PACKAGING_TYPES:
                        counts[t] = counts.get(t, 0) + all_found[old].get(t, 0)
            total = sum(counts.values())
            if total > 0:
                found[internal] = counts
            else:
                missing.append((brand, product_display, internal))

    # Find images not in registry
    all_registered = set()
    for brand, products in BRAND_REGISTRY.items():
        for _, internal in products:
            all_registered.add(internal)
    all_registered.update(LEGACY_ALIASES.keys())

    unregistered = {}
    for label, count in all_label_counts.items():
        if label not in all_registered:
            unregistered[label] = count

    # Find legacy-named files
    legacy_found = {}
    for old, new in LEGACY_ALIASES.items():
        if old != new and all_label_counts.get(old, 0) > 0:
            legacy_found[old] = new

    return {
        "found": found,
        "missing": missing,
        "unregistered": unregistered,
        "legacy": legacy_found,
        "total_images": total_images,
        "total_products_found": len(found),
        "total_products_missing": len(missing),
        "per_type": per_type_totals,
    }
```

- [ ] **Step 3: Update `print_audit()` to show per-type counts**

Replace the `print_audit` function (lines 279-309) with:

```python
def print_audit():
    """Print a human-readable audit report."""
    result = audit_references()

    total_products = sum(len(products) for products in BRAND_REGISTRY.values())
    print(f"=== Brand Registry Audit ===")
    print(f"Registry: {len(BRAND_REGISTRY)} brands, {total_products} products")
    print(f"References: {result['total_images']} images")
    for pkg_type, count in result.get("per_type", {}).items():
        print(f"  {pkg_type}: {count} images")
    print(f"Products with images: {result['total_products_found']}/{total_products}")
    print(f"Products missing: {result['total_products_missing']}/{total_products}")

    if result["found"]:
        print(f"\n--- Products with references ({result['total_products_found']}) ---")
        for internal, counts in sorted(result["found"].items()):
            brand, display = INTERNAL_TO_BRAND_product.get(internal, ("?", internal))
            count_str = ", ".join(f"{t}={c}" for t, c in counts.items() if c > 0)
            total = sum(counts.values())
            print(f"  {display:<40} {total:>3} images  ({count_str})  ({internal})")

    if result["missing"]:
        print(f"\n--- MISSING Products ({result['total_products_missing']}) ---")
        for brand, product_display, internal in result["missing"]:
            print(f"  {brand:<15} {product_display:<40} (need: pack/{internal}_1.jpg)")

    if result["legacy"]:
        print(f"\n--- Legacy-named files (should rename) ---")
        for old, new in sorted(result["legacy"].items()):
            print(f"  {old} -> {new}")

    if result["unregistered"]:
        print(f"\n--- Unregistered (not in registry) ---")
        for label, count in sorted(result["unregistered"].items()):
            print(f"  {label}: {count} images")
```

- [ ] **Step 4: Verify audit still works**

Run: `cd rf-detr-cigarette && python -c "from backend.brand_registry import print_audit; print_audit()"`
Expected: Shows audit with per-type counts (all under "pack")

- [ ] **Step 5: Commit**

```bash
git add backend/brand_registry.py
git commit -m "Update brand_registry to scan pack/box reference subfolders"
```

---

### Task 3: Update `brand_classifier.py` -- Type-Aware Training

**Files:**
- Modify: `brand_classifier.py:28-29` (paths), `:116-281` (train_classifier), `:284-297` (main/args)

- [ ] **Step 1: Update paths and add `--packaging-type` argument**

Replace lines 27-31:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCES_DIR = PROJECT_ROOT / "backend" / "references"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "classifier_model"
DINO_MODEL_ID = "facebook/dinov2-base"
EMBED_DIM = 1536  # CLS (768) + mean-pooled patches (768)
```

with:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCES_BASE_DIR = PROJECT_ROOT / "backend" / "references"
OUTPUT_BASE_DIR = PROJECT_ROOT / "backend" / "classifier_model"
DINO_MODEL_ID = "facebook/dinov2-base"
EMBED_DIM = 1536  # CLS (768) + mean-pooled patches (768)
PACKAGING_TYPES = ("pack", "box")
```

- [ ] **Step 2: Update `train_classifier()` to accept packaging_type**

At the start of `train_classifier` (line 116), change:

```python
def train_classifier(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect reference images and labels
    print("Scanning reference images...")
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_paths.extend(REFERENCES_DIR.glob(ext))
        image_paths.extend(REFERENCES_DIR.glob(ext.upper()))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No reference images found in {REFERENCES_DIR}")
        sys.exit(1)
```

to:

```python
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
```

- [ ] **Step 3: Update the model save path in the training loop**

In the training loop (around line 253), replace:

```python
            torch.save(classifier.state_dict(), OUTPUT_DIR / "best_classifier.pth")
```

This already uses `OUTPUT_DIR` which is now type-specific -- no change needed. But verify the final save block (around line 262-273) also uses `OUTPUT_DIR`:

```python
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
```

Add `"packaging_type": pkg_type` to the `class_mapping` dict.

- [ ] **Step 4: Update argparse to add --packaging-type**

Replace the `main()` function (lines 284-297):

```python
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
```

- [ ] **Step 5: Verify training works for pack type**

Run: `python brand_classifier.py --packaging-type pack --epochs 5`
Expected: Scans `backend/references/pack/`, saves to `backend/classifier_model/pack/`

- [ ] **Step 6: Commit**

```bash
git add brand_classifier.py
git commit -m "Add --packaging-type arg to brand_classifier for pack/box training"
```

---

### Task 4: Update `pipeline.py` -- Dual Classifier Loading and Type-Aware Routing

**Files:**
- Modify: `backend/pipeline.py:21-24` (paths), `:260-291` (load_classifier), `:293-321` (classify_embeddings), `:325-331` (load_index), `:499-615` (_detect_brands_from_image)

This is the largest change. The pipeline must load two classifiers and route crops based on RF-DETR class_id.

- [ ] **Step 1: Update path constants and globals**

Replace lines 21-24 and 49-54:

```python
REFERENCES_DIR = _BACKEND_ROOT / "references"
CLASSIFIER_DIR = _BACKEND_ROOT / "classifier_model"
CLASSIFIER_WEIGHTS = CLASSIFIER_DIR / "best_classifier.pth"
CLASS_MAPPING_JSON = CLASSIFIER_DIR / "class_mapping.json"
```

with:

```python
REFERENCES_DIR = _BACKEND_ROOT / "references"
CLASSIFIER_BASE_DIR = _BACKEND_ROOT / "classifier_model"
PACKAGING_TYPES = ("pack", "box")

# Per-type paths (for backward compat, keep top-level aliases pointing to pack)
CLASSIFIER_DIR = CLASSIFIER_BASE_DIR / "pack"
CLASSIFIER_WEIGHTS = CLASSIFIER_DIR / "best_classifier.pth"
CLASS_MAPPING_JSON = CLASSIFIER_DIR / "class_mapping.json"
```

Replace lines 49-54 (globals):

```python
_dino_processor = None
_dino_model = None
_rfdetr_model = None
_ocr_reader = None
_brand_classifier = None
_class_mapping = None
```

with:

```python
_dino_processor = None
_dino_model = None
_rfdetr_model = None
_ocr_reader = None
# Per-type classifiers: {"pack": (classifier, mapping), "box": (classifier, mapping)}
_classifiers: dict[str, tuple] = {}
# Legacy single-classifier aliases (for backward compat)
_brand_classifier = None
_class_mapping = None
```

- [ ] **Step 2: Update `load_classifier()` to accept packaging_type**

Replace the `load_classifier` function (lines 260-291) with:

```python
def load_classifier(device: str = None, packaging_type: str = "pack"):
    """Load the trained brand classifier and class mapping for a packaging type."""
    global _classifiers, _brand_classifier, _class_mapping

    if packaging_type in _classifiers:
        return _classifiers[packaging_type]

    type_dir = CLASSIFIER_BASE_DIR / packaging_type
    weights_path = type_dir / "best_classifier.pth"
    mapping_path = type_dir / "class_mapping.json"

    if not weights_path.exists() or not mapping_path.exists():
        # Fallback: check legacy flat structure (pre-migration)
        legacy_weights = CLASSIFIER_BASE_DIR / "best_classifier.pth"
        legacy_mapping = CLASSIFIER_BASE_DIR / "class_mapping.json"
        if packaging_type == "pack" and legacy_weights.exists() and legacy_mapping.exists():
            weights_path = legacy_weights
            mapping_path = legacy_mapping
        else:
            raise FileNotFoundError(
                f"Missing {packaging_type} classifier model. Expected {weights_path} and {mapping_path}. "
                f"Run 'python brand_classifier.py --packaging-type {packaging_type}' or /build-index first."
            )

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    num_classes = mapping["num_classes"]
    embed_dim = mapping["embed_dim"]
    hidden_dim = mapping.get("hidden_dim", 512)

    classifier = BrandClassifier(embed_dim, num_classes, hidden_dim)

    if device is None:
        device = get_device()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    classifier.to(device)
    classifier.eval()

    _classifiers[packaging_type] = (classifier, mapping)

    # Keep legacy aliases pointing to pack classifier
    if packaging_type == "pack":
        _brand_classifier = classifier
        _class_mapping = mapping

    logger.info("Loaded %s brand classifier: %d classes", packaging_type, num_classes)
    return classifier, mapping
```

- [ ] **Step 3: Update `classify_embeddings()` to accept packaging_type**

Replace the `classify_embeddings` function (lines 293-321) with:

```python
def classify_embeddings(
    embeddings: np.ndarray,
    device: str,
    top_k: int = CLASSIFIER_TOP_K,
    packaging_type: str = "pack",
) -> list[list[tuple[str, float]]]:
    """Run the brand classifier on pre-computed DINOv2 embeddings.

    Args:
        embeddings: (N, embed_dim) numpy array of L2-normalized embeddings
        device: torch device
        top_k: number of top predictions to return per sample
        packaging_type: "pack" or "box" -- determines which classifier to use

    Returns:
        List of N lists, each containing top_k (label, confidence) tuples.
    """
    classifier, mapping = load_classifier(device, packaging_type=packaging_type)
    idx_to_label = mapping["idx_to_label"]

    with torch.no_grad():
        x = torch.tensor(embeddings, dtype=torch.float32).to(device)
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)

    results = []
    for i in range(len(embeddings)):
        topk_vals, topk_idxs = torch.topk(probs[i], min(top_k, probs.shape[1]))
        sample_results = []
        for val, idx in zip(topk_vals, topk_idxs):
            label = idx_to_label[str(idx.item())]
            sample_results.append((label, float(val.item())))
        results.append(sample_results)

    return results
```

- [ ] **Step 4: Update `load_index()` for backward compatibility**

Replace the `load_index` function (lines 325-331) with:

```python
def load_index(packaging_type: str = "pack"):
    """Load classifier and return (classifier, labels) for backward compatibility."""
    device = get_device()
    classifier, mapping = load_classifier(device, packaging_type=packaging_type)
    labels = list(mapping["label_to_idx"].keys())
    return classifier, labels
```

- [ ] **Step 5: Update `build_index()` to train both classifiers**

Replace the `build_index` function (lines 228-257) with:

```python
def build_index(device: str, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> None:
    """Train brand classifiers for each packaging type that has reference images."""
    import subprocess
    import sys

    train_script = _PROJECT_ROOT / "brand_classifier.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    trained_types = []
    for idx, pkg_type in enumerate(PACKAGING_TYPES):
        type_dir = REFERENCES_DIR / pkg_type
        if not type_dir.exists():
            continue
        # Check if there are any images
        has_images = any(type_dir.glob("*.jpg")) or any(type_dir.glob("*.png"))
        if not has_images:
            if progress_cb:
                progress_cb(
                    int((idx + 1) / len(PACKAGING_TYPES) * 100),
                    100,
                    f"Skipping {pkg_type} classifier (no reference images)",
                )
            continue

        if progress_cb:
            progress_cb(
                int(idx / len(PACKAGING_TYPES) * 50),
                100,
                f"Training {pkg_type} classifier...",
            )

        result = subprocess.run(
            [sys.executable, str(train_script), "--packaging-type", pkg_type,
             "--epochs", "100", "--embed-batch-size", "8"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"{pkg_type} classifier training failed:\n{result.stderr}")

        trained_types.append(pkg_type)

    if progress_cb:
        progress_cb(100, 100, f"Classifiers trained: {', '.join(trained_types)}")

    # Reload classifiers
    global _classifiers
    _classifiers = {}
    for pkg_type in trained_types:
        try:
            load_classifier(device, packaging_type=pkg_type)
        except FileNotFoundError:
            pass
```

- [ ] **Step 6: Update `_detect_brands_from_image()` for type-aware routing**

Replace the `_detect_brands_from_image` function (lines 499-615) with:

```python
def _detect_brands_from_image(
    image: Image.Image,
    rfdetr_model,
    processor,
    model,
    device: str,
    index,
    labels: list[str],
) -> dict[str, float]:
    """Detect and classify cigarette brands in an image.

    Pipeline (classifier-first with OCR fallback):
      1. RF-DETR detects bounding boxes with class_id (0=pack, 1=box)
      2. Crops are grouped by packaging type
      3. Each group is classified by its type-specific DINOv2 classifier
      4. EasyOCR runs only on uncertain crops (low confidence / low margin)
      5. OCR signals boost matching classifier families for uncertain crops
    """
    # Load labels for each available packaging type
    type_labels = {}
    type_label_profiles = {}
    for pkg_type in PACKAGING_TYPES:
        try:
            _, pkg_labels = load_index(packaging_type=pkg_type)
            type_labels[pkg_type] = pkg_labels
            type_label_profiles[pkg_type] = _build_label_profiles(pkg_labels)
        except FileNotFoundError:
            pass

    # Fallback: if no type-specific classifiers, use legacy labels
    if not type_labels:
        type_labels["pack"] = labels
        type_label_profiles["pack"] = _build_label_profiles(labels)

    detections = rfdetr_model.predict(image, threshold=RFDETR_CONF_THRESHOLD)
    crops: list[Image.Image] = []
    crop_types: list[str] = []  # "pack" or "box" per crop

    has_detections = len(detections) > 0 if detections is not None else False

    if has_detections:
        width, height = image.size
        xyxy = detections.xyxy
        # class_id: 0 = pack (or single-class legacy), 1 = box
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))

            # Determine packaging type from class_id
            if class_ids is not None and len(class_ids) > i:
                cid = int(class_ids[i])
                crop_types.append("box" if cid == 1 else "pack")
            else:
                crop_types.append("pack")
    else:
        crops.append(image)
        crop_types.append("pack")

    # Group crops by packaging type for batch classification
    type_crop_indices: dict[str, list[int]] = {}
    for idx, pkg_type in enumerate(crop_types):
        effective_type = pkg_type if pkg_type in type_labels else "pack"
        type_crop_indices.setdefault(effective_type, []).append(idx)

    # Embed all crops at once (DINOv2 is shared)
    all_vecs = embed_images_batch(crops, processor, model, device)

    # Classify per type
    all_cls_results: list[list[tuple[str, float]]] = [[] for _ in crops]
    all_label_profiles = _build_label_profiles(labels)  # combined for OCR

    for pkg_type, indices in type_crop_indices.items():
        if not indices:
            continue
        type_vecs = all_vecs[indices]
        type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type=pkg_type)
        for local_idx, global_idx in enumerate(indices):
            all_cls_results[global_idx] = type_results[local_idx]

    # Build combined label profiles for OCR matching (union of all types)
    combined_labels = set()
    for pkg_labels in type_labels.values():
        combined_labels.update(pkg_labels)
    combined_label_profiles = _build_label_profiles(list(combined_labels))
    label_profile_map = {p["label"]: p for p in combined_label_profiles}

    def _needs_ocr_fallback(crop_results: list[tuple[str, float]]) -> bool:
        if not OCR_ENABLED:
            return False
        if not crop_results:
            return True
        top1 = float(crop_results[0][1])
        top2 = float(crop_results[1][1]) if len(crop_results) > 1 else 0.0
        margin = top1 - top2
        return top1 < OCR_FALLBACK_THRESHOLD or margin < OCR_FALLBACK_MARGIN

    ocr_needed = [_needs_ocr_fallback(r) for r in all_cls_results]

    # OCR fallback only for uncertain crops
    per_crop_ocr_scores: list[dict[str, float]] = [{} for _ in crops]
    ocr_best: dict[str, float] = {}
    for idx, crop in enumerate(crops):
        if not ocr_needed[idx]:
            continue
        ocr_items = _run_ocr_on_image(crop)
        crop_ocr_scores = _ocr_brand_scores_from_items(ocr_items, combined_label_profiles)
        per_crop_ocr_scores[idx] = crop_ocr_scores
        for label, conf in crop_ocr_scores.items():
            if conf > ocr_best.get(label, 0.0):
                ocr_best[label] = conf

    # Optional full-image OCR context
    fullimg_ocr_families: dict[str, float] = {}
    if OCR_FULLIMG_ENABLED and has_detections and any(ocr_needed):
        fullimg_scores = _ocr_brand_scores_from_items(_run_ocr_on_image(image), combined_label_profiles)
        for label, conf in fullimg_scores.items():
            if conf > ocr_best.get(label, 0.0):
                ocr_best[label] = conf
            family = label_profile_map.get(label, {}).get("brand", "")
            if family:
                fullimg_ocr_families[family] = max(fullimg_ocr_families.get(family, 0.0), conf)

    # Fuse classifier with OCR fallback
    fused: dict[str, float] = {}
    for crop_idx, crop_results in enumerate(all_cls_results):
        crop_ocr_scores = per_crop_ocr_scores[crop_idx]
        crop_ocr_families: dict[str, float] = {}
        for label, ocr_conf in crop_ocr_scores.items():
            family = label_profile_map.get(label, {}).get("brand", "")
            if family:
                crop_ocr_families[family] = max(crop_ocr_families.get(family, 0.0), ocr_conf)

        for label, cls_conf in crop_results:
            out_conf = float(cls_conf)
            if ocr_needed[crop_idx]:
                brand_family = label_profile_map.get(label, {}).get("brand", "")
                fam_conf = 0.0
                if brand_family:
                    fam_conf = max(
                        crop_ocr_families.get(brand_family, 0.0),
                        fullimg_ocr_families.get(brand_family, 0.0),
                    )
                if fam_conf >= OCR_STRONG_THRESHOLD:
                    out_conf = min(1.0, out_conf + fam_conf * 0.25)
                elif fam_conf > 0:
                    out_conf = min(1.0, out_conf + fam_conf * 0.10)
            if out_conf > fused.get(label, 0.0):
                fused[label] = out_conf

    # OCR-only brands
    for label, ocr_conf in ocr_best.items():
        if label in fused:
            continue
        if ocr_conf >= OCR_INDEPENDENT_MIN_SCORE:
            fused[label] = min(1.0, ocr_conf * 0.85)

    return _aggregate_to_products(fused)
```

- [ ] **Step 7: Verify pipeline imports still work**

Run: `cd rf-detr-cigarette && python -c "from backend.pipeline import load_classifier, classify_embeddings, _detect_brands_from_image, load_index, build_index; print('OK')"`
Expected: "OK"

- [ ] **Step 8: Commit**

```bash
git add backend/pipeline.py
git commit -m "Add dual pack/box classifier routing to detection pipeline"
```

---

### Task 5: Update `backend/main.py` -- API Endpoints

**Files:**
- Modify: `backend/main.py:718-769` (/add-reference), `:631-715` (/generate-crops), `:772-838` (/brand-registry, reference image endpoints), `:247-417` (/detect-single), `:111-171` (build-index)

- [ ] **Step 1: Update `/add-reference` to accept `packaging_type` parameter**

Replace the `/add-reference` endpoint (lines 718-769) with:

```python
@app.post("/add-reference")
async def add_reference(
    image_file: UploadFile = File(...),
    product_name: str = Form(...),
    packaging_type: str = Form("pack"),
):
    """Add a confirmed crop as a reference image for a specific product and packaging type."""
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name is required.")
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'.")

    try:
        from .brand_registry import BRAND_REGISTRY
    except ImportError:
        from brand_registry import BRAND_REGISTRY

    # Validate product_name exists in registry
    valid_internals = set()
    for brand, products in BRAND_REGISTRY.items():
        for _, internal in products:
            valid_internals.add(internal)

    if product_name not in valid_internals:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown product '{product_name}'. Valid products: {sorted(valid_internals)}",
        )

    data = await image_file.read()
    from PIL import Image
    try:
        Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    TYPE_DIR = _BACKEND_ROOT / "references" / packaging_type
    TYPE_DIR.mkdir(parents=True, exist_ok=True)

    # Find next index
    import re
    existing = list(TYPE_DIR.glob(f"{product_name}_*.*"))
    max_idx = 0
    for p in existing:
        match = re.search(r"_(\d+)$", p.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    next_idx = max_idx + 1

    save_path = TYPE_DIR / f"{product_name}_{next_idx}.jpg"
    save_path.write_bytes(data)

    return {
        "status": "added",
        "product": product_name,
        "packaging_type": packaging_type,
        "filename": save_path.name,
        "total_for_product": next_idx,
    }
```

- [ ] **Step 2: Update `/generate-crops` to return detected packaging type**

In the `/generate-crops` endpoint (lines 631-715), after the RF-DETR detection section (around line 671), add class_id tracking. Replace the crop extraction loop:

```python
    crop_images = []
    crop_meta = []
    if detections is not None and len(detections) > 0:
        width, height = pil_img.size
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.05), int(bh * 0.05)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil_img.crop((x1, y1, x2, y2))
            crop_images.append(crop)
            pkg_type = "pack"
            if class_ids is not None and len(class_ids) > i:
                pkg_type = "box" if int(class_ids[i]) == 1 else "pack"
            crop_meta.append({"index": i, "w": x2 - x1, "h": y2 - y1, "conf": float(conf), "packaging_type": pkg_type})
```

Then in the response building (around line 704), add `packaging_type` to each crop:

```python
        crops.append({
            "index": meta["index"],
            "image_b64": crop_b64,
            "width": meta["w"],
            "height": meta["h"],
            "det_conf": round(meta["conf"], 3),
            "packaging_type": meta["packaging_type"],
            "suggested_brand": suggestion.get("brand", ""),
            "suggested_product": suggestion.get("internal_name", ""),
            "suggested_confidence": suggestion.get("confidence", 0.0),
        })
```

Also update the classification section to use type-aware classification. Replace the classification block (lines 676-695):

```python
    suggested_labels = []
    if crop_images:
        try:
            processor, model = load_dino(device)

            vecs = embed_images_batch(crop_images, processor, model, device)

            # Group by packaging type for classification
            type_indices: dict[str, list[int]] = {}
            for idx, meta in enumerate(crop_meta):
                pkg = meta["packaging_type"]
                type_indices.setdefault(pkg, []).append(idx)

            per_crop_results: list[tuple[str, float]] = [("unknown", 0.0)] * len(crop_images)
            for pkg_type, indices in type_indices.items():
                try:
                    type_vecs = vecs[indices]
                    cls_results = classify_embeddings(type_vecs, device, top_k=3, packaging_type=pkg_type)
                    for local_idx, global_idx in enumerate(indices):
                        if cls_results[local_idx]:
                            per_crop_results[global_idx] = cls_results[local_idx][0]
                except FileNotFoundError:
                    pass  # No classifier for this type yet

            for crop_idx, top_pred in enumerate(per_crop_results):
                internal_name = resolve_internal_name(top_pred[0])
                cls_conf = top_pred[1]
                brand = get_brand(internal_name)
                suggested_labels.append({
                    "internal_name": internal_name,
                    "brand": brand,
                    "confidence": round(cls_conf, 3),
                })
        except Exception:
            suggested_labels = [{"internal_name": "", "brand": "", "confidence": 0.0}] * len(crop_images)
```

- [ ] **Step 3: Update `/brand-registry` to include per-type reference counts**

Replace the `/brand-registry` endpoint (lines 772-800) with:

```python
@app.get("/brand-registry")
def get_brand_registry():
    """Return the full brand->products hierarchy with per-type reference counts."""
    try:
        from .brand_registry import BRAND_REGISTRY, audit_references
    except ImportError:
        from brand_registry import BRAND_REGISTRY, audit_references

    audit = audit_references()

    hierarchy = {}
    for brand, products in BRAND_REGISTRY.items():
        hierarchy[brand] = []
        for display_name, internal_name in products:
            found_entry = audit["found"].get(internal_name, {})
            # found_entry is now {pkg_type: count} dict
            pack_count = found_entry.get("pack", 0) if isinstance(found_entry, dict) else found_entry
            box_count = found_entry.get("box", 0) if isinstance(found_entry, dict) else 0
            hierarchy[brand].append({
                "display_name": display_name,
                "internal_name": internal_name,
                "reference_count": pack_count + box_count,
                "pack_count": pack_count,
                "box_count": box_count,
            })

    return {
        "brands": hierarchy,
        "total_brands": len(BRAND_REGISTRY),
        "total_products": sum(len(p) for p in BRAND_REGISTRY.values()),
        "products_with_refs": audit.get("total_products_found", len(audit.get("found", {}))),
        "products_missing": audit.get("total_products_missing", len(audit.get("missing", []))),
        "total_images": audit.get("total_images", 0),
        "per_type": audit.get("per_type", {}),
    }
```

- [ ] **Step 4: Update reference image serving endpoints for subfolders**

Replace `/reference-image/{filename}` (lines 803-810):

```python
@app.get("/reference-image/{packaging_type}/{filename}")
def get_reference_image(packaging_type: str, filename: str):
    """Serve a reference image by packaging type and filename."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _BACKEND_ROOT / "references" / packaging_type
    path = REFERENCES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/jpeg")
```

Replace `/delete-reference-image/{filename}` (lines 813-824):

```python
@app.delete("/reference-image/{packaging_type}/{filename}")
def delete_reference_image(packaging_type: str, filename: str):
    """Delete a reference image by packaging type and filename."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _BACKEND_ROOT / "references" / packaging_type
    path = REFERENCES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if not path.resolve().parent == REFERENCES_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid path")
    path.unlink()
    return {"status": "deleted", "packaging_type": packaging_type, "filename": filename}
```

Replace `/reference-images/{product_name}` (lines 827-837):

```python
@app.get("/reference-images/{product_name}")
def list_reference_images(product_name: str, packaging_type: str = "pack"):
    """List all reference image filenames for a product in a packaging type subfolder."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _BACKEND_ROOT / "references" / packaging_type
    files = sorted(REFERENCES_DIR.glob(f"{product_name}_*.*")) if REFERENCES_DIR.exists() else []
    return {
        "product": product_name,
        "packaging_type": packaging_type,
        "count": len(files),
        "filenames": [f.name for f in files],
    }
```

- [ ] **Step 5: Update `/detect-single` to pass class_id through**

In the `/detect-single` endpoint (around line 297-320), update the detection loop to extract class_ids:

After line 300 (`has_detections = ...`), the crop extraction block (lines 302-330) should add packaging_type to boxes_data:

```python
    if has_detections:
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img_w, x2 + pad_x)
            y2 = min(img_h, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(pil_img.crop((x1, y1, x2, y2)))
            pkg_type = "pack"
            if class_ids is not None and len(class_ids) > i:
                pkg_type = "box" if int(class_ids[i]) == 1 else "pack"
            boxes_data.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_conf": round(float(conf), 3),
                "packaging_type": pkg_type,
                "brands": [],
                "ocr_texts": [],
                "ocr_brand_scores": [],
            })
```

Then update the classify section (line 333-334) to group by type:

```python
    # Classify crops grouped by packaging type
    all_vecs = embed_images_batch(crops, processor, model, device)
    crop_pkg_types = [b.get("packaging_type", "pack") for b in boxes_data]

    type_indices: dict[str, list[int]] = {}
    for idx, pkg_type in enumerate(crop_pkg_types):
        type_indices.setdefault(pkg_type, []).append(idx)

    all_cls_results: list[list[tuple[str, float]]] = [[] for _ in crops]
    for pkg_type, indices in type_indices.items():
        try:
            type_vecs = all_vecs[indices]
            type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type=pkg_type)
            for local_idx, global_idx in enumerate(indices):
                all_cls_results[global_idx] = type_results[local_idx]
        except FileNotFoundError:
            # Fall back to pack classifier
            type_vecs = all_vecs[indices]
            type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type="pack")
            for local_idx, global_idx in enumerate(indices):
                all_cls_results[global_idx] = type_results[local_idx]
```

- [ ] **Step 6: Update imports at top of main.py**

The existing imports (lines 17-53) reference `CLASSIFIER_WEIGHTS` and `CLASS_MAPPING_JSON`. These still exist as aliases in pipeline.py so no change needed. But also add `PACKAGING_TYPES`:

After line 33 (`RFDETR_CONF_THRESHOLD,`), add inside the import block:

```python
        PACKAGING_TYPES,
```

Do the same in both the try and except import blocks.

- [ ] **Step 7: Commit**

```bash
git add backend/main.py
git commit -m "Update API endpoints for pack/box packaging type support"
```

---

### Task 6: Update `frontend/app.py` -- Label Crop UI with Pack/Box Toggle

**Files:**
- Modify: `frontend/app.py:1189-1279` (label crop cards), `:157-189` (cached fetchers), `:1062-1128` (reference viewer)

- [ ] **Step 1: Add pack/box radio button to `_render_crop_card()`**

Replace the `_render_crop_card` function (lines 1189-1276) with:

```python
        @st.fragment
        def _render_crop_card(crop, brand_names, brand_hierarchy):
            """Render a single crop card as a fragment -- selectbox changes only rerun this card."""
            col_img, col_form = st.columns([1, 2])

            with col_img:
                crop_bytes = b64lib.b64decode(crop["image_b64"])
                st.image(crop_bytes, caption=f"Crop #{crop['index']+1} ({crop['width']}x{crop['height']})", width=200)
                if crop.get("suggested_confidence", 0) > 0:
                    st.caption(f"AI suggests: **{crop.get('suggested_brand', '?')}** / {crop.get('suggested_product', '?').replace('_', ' ')} ({crop['suggested_confidence']:.0%})")
                detected_type = crop.get("packaging_type", "pack")
                if detected_type == "box":
                    st.caption("Detected as: **box**")

            with col_form:
                crop_key = f"crop_{crop['index']}"
                products_for_brand = []
                product_internals = {}
                selected_product = ""

                # Packaging type selector (default from RF-DETR detection)
                detected_type = crop.get("packaging_type", "pack")
                type_options = ["pack", "box"]
                default_type_idx = type_options.index(detected_type) if detected_type in type_options else 0
                selected_type = st.radio(
                    "Type",
                    options=type_options,
                    index=default_type_idx,
                    key=f"{crop_key}_type",
                    horizontal=True,
                )

                # Pre-select brand from AI suggestion
                suggested_brand = crop.get("suggested_brand", "")
                default_brand_idx = 0
                if suggested_brand in brand_names:
                    default_brand_idx = brand_names.index(suggested_brand) + 1

                selected_brand = st.selectbox(
                    "Brand",
                    options=["-- skip --"] + brand_names,
                    index=default_brand_idx,
                    key=f"{crop_key}_brand",
                )

                if selected_brand and selected_brand != "-- skip --":
                    products_for_brand = brand_hierarchy.get(selected_brand, [])
                    product_options = [p["display_name"] for p in products_for_brand]
                    product_internals = {p["display_name"]: p["internal_name"] for p in products_for_brand}

                    # Pre-select product from AI suggestion
                    suggested_product = crop.get("suggested_product", "")
                    default_prod_idx = 0
                    for pidx, p in enumerate(products_for_brand):
                        if p["internal_name"] == suggested_product:
                            default_prod_idx = pidx
                            break

                    selected_product = st.selectbox(
                        "Product",
                        options=product_options,
                        index=default_prod_idx,
                        key=f"{crop_key}_product",
                    )

                    if st.button("Add to references", key=f"{crop_key}_add"):
                        internal_name = product_internals.get(selected_product, "")
                        if internal_name:
                            try:
                                crop_bytes_data = b64lib.b64decode(crop["image_b64"])
                                resp = requests.post(
                                    f"{BACKEND_URL}/add-reference",
                                    files={"image_file": ("crop.jpg", crop_bytes_data, "image/jpeg")},
                                    data={"product_name": internal_name, "packaging_type": selected_type},
                                    timeout=15,
                                )
                                resp.raise_for_status()
                                result = resp.json()
                                st.success(f"Added as {selected_type}/{result['filename']} ({result['total_for_product']} total for {selected_product} [{selected_type}])")
                                _fetch_reference_listing.clear()
                                _fetch_brand_hierarchy.clear()
                            except Exception as exc:
                                st.error(f"Failed: {exc}")

                    # Show reference preview for selected type
                    internal_name = product_internals.get(selected_product, "")
                    if internal_name:
                        ref_data = _fetch_reference_listing(internal_name, selected_type)
                        if ref_data and ref_data.get("filenames"):
                            first_ref = ref_data["filenames"][0]
                            img_bytes = _fetch_reference_image_bytes(selected_type, first_ref)
                            if img_bytes:
                                st.image(
                                    img_bytes,
                                    caption=f"Reference preview: {selected_product} [{selected_type}]",
                                    width=170,
                                )
                            st.caption(f"{ref_data['count']} {selected_type} refs total")
                        elif ref_data:
                            st.caption(f"No {selected_type} reference images yet")
                        else:
                            st.caption("Reference preview unavailable")

            st.markdown("---")
```

- [ ] **Step 2: Update cached fetcher functions for new URL structure**

Replace `_fetch_reference_image_bytes` (lines 157-165):

```python
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_reference_image_bytes(packaging_type: str, filename: str):
    """Fetch reference image from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/reference-image/{packaging_type}/{filename}", timeout=10)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None
```

Replace `_fetch_reference_listing` (lines 180-189):

```python
@st.cache_data(ttl=120, show_spinner=False)
def _fetch_reference_listing(internal_name: str, packaging_type: str = "pack"):
    """Fetch reference image listing for a product, cached."""
    try:
        resp = requests.get(f"{BACKEND_URL}/reference-images/{internal_name}?packaging_type={packaging_type}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None
```

- [ ] **Step 3: Update reference viewer in Rebuild Index tab (lines 1062-1128)**

In the product reference viewer section, add a type selector and update all calls. Replace the viewer block starting at line 1055:

```python
        if picked_brand != "-- Select brand --" and picked_product_display != "-- Select product --":
            picked_internal = ""
            for p in picked_products:
                if p["display_name"] == picked_product_display:
                    picked_internal = p["internal_name"]
                    break
            if picked_internal:
                ref_type = st.radio("Reference type", options=["pack", "box"], horizontal=True, key="ref_viewer_type")
                with st.expander(f"Show {ref_type} references for {picked_product_display}", expanded=True):
                    try:
                        ref_data = _fetch_reference_listing(picked_internal, ref_type)
                        filenames = ref_data.get("filenames", []) if ref_data else []
                        if filenames:
                            cols = st.columns(min(5, len(filenames)))
                            for img_idx, fname in enumerate(filenames):
                                with cols[img_idx % len(cols)]:
                                    img_bytes = _fetch_reference_image_bytes(ref_type, fname)
                                    if img_bytes:
                                        st.image(img_bytes, caption=fname, width=110)
                                    if st.button("Delete", key=f"picker_del_{ref_type}_{fname}", type="secondary"):
                                        try:
                                            del_resp = requests.delete(f"{BACKEND_URL}/reference-image/{ref_type}/{fname}", timeout=5)
                                            del_resp.raise_for_status()
                                            _fetch_reference_listing.clear()
                                            _fetch_reference_image_bytes.clear()
                                            _fetch_brand_hierarchy.clear()
                                            st.rerun()
                                        except Exception as exc:
                                            st.error(f"Delete failed: {exc}")
                        else:
                            st.caption(f"No {ref_type} reference images yet for this product.")
                    except Exception as exc:
                        st.caption(f"Could not load references: {exc}")
```

- [ ] **Step 4: Update brand expander reference viewer (lines 1090-1128)**

Update the per-brand reference image viewer to show per-type counts and use new URL paths. In the expander section, update reference count display:

```python
            with st.expander(f"{brand_idx}. {brand_name} -- {len(products)} products {status}", expanded=False):
                for prod_idx, prod in enumerate(products, 1):
                    count = prod["reference_count"]
                    pack_c = prod.get("pack_count", count)
                    box_c = prod.get("box_count", 0)
                    name = prod["display_name"]
                    internal = prod["internal_name"]
                    if count > 0:
                        type_detail = f"pack={pack_c}" + (f", box={box_c}" if box_c > 0 else "")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{brand_idx}.{prod_idx} **{name}** -- {count} refs ({type_detail})")
                        for ref_type in ("pack", "box"):
                            with st.expander(f"View {name} {ref_type} references", expanded=False):
                                try:
                                    ref_data = _fetch_reference_listing(internal, ref_type)
                                    if ref_data and ref_data.get("filenames"):
                                        fnames = ref_data["filenames"][:20]
                                        cols = st.columns(min(5, len(fnames)))
                                        for img_idx, fname in enumerate(fnames):
                                            with cols[img_idx % len(cols)]:
                                                img_bytes = _fetch_reference_image_bytes(ref_type, fname)
                                                if img_bytes:
                                                    st.image(img_bytes, caption=fname, width=120)
                                                if st.button("Delete", key=f"brand_del_{ref_type}_{fname}", type="secondary"):
                                                    try:
                                                        del_resp = requests.delete(f"{BACKEND_URL}/reference-image/{ref_type}/{fname}", timeout=5)
                                                        del_resp.raise_for_status()
                                                        _fetch_reference_listing.clear()
                                                        _fetch_reference_image_bytes.clear()
                                                        _fetch_brand_hierarchy.clear()
                                                        st.rerun()
                                                    except Exception as exc:
                                                        st.error(f"Delete failed: {exc}")
                                        if len(ref_data["filenames"]) > 20:
                                            st.caption(f"... and {len(ref_data['filenames']) - 20} more")
                                    else:
                                        st.caption(f"No {ref_type} references")
                                except Exception:
                                    st.caption("Could not load images")
                    else:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{brand_idx}.{prod_idx} ~~{name}~~ -- missing `(need: pack/{internal}_1.jpg)`")
```

- [ ] **Step 5: Commit**

```bash
git add frontend/app.py
git commit -m "Add pack/box type selector to label crop UI and reference viewer"
```

---

### Task 7: Compatibility Audit and Smoke Test

**Files:**
- All modified files

This task verifies no regressions across the entire system.

- [ ] **Step 1: Verify all Python imports work**

Run:
```bash
cd rf-detr-cigarette
python -c "from backend.brand_registry import audit_references, PACKAGING_TYPES; print('brand_registry OK')"
python -c "from backend.pipeline import load_classifier, classify_embeddings, build_index, load_index, _detect_brands_from_image, PACKAGING_TYPES; print('pipeline OK')"
python -c "from backend.main import app; print('main OK')"
python -c "from brand_classifier import train_classifier, PACKAGING_TYPES; print('brand_classifier OK')"
```
Expected: All print "OK"

- [ ] **Step 2: Verify reference directory structure**

Run:
```bash
ls backend/references/pack/ | wc -l
ls backend/references/box/ 2>/dev/null | wc -l
ls backend/references/*.jpg 2>/dev/null | wc -l
```
Expected: pack/ has ~385 images, box/ has 0, root has 0 loose images

- [ ] **Step 3: Verify audit_references returns new format**

Run:
```bash
python -c "
from backend.brand_registry import audit_references
r = audit_references()
print('found type:', type(list(r['found'].values())[0]) if r['found'] else 'empty')
print('per_type:', r.get('per_type', {}))
print('total:', r['total_images'])
"
```
Expected: `found type: <class 'dict'>`, `per_type: {'pack': ~385, 'box': 0}`, `total: ~385`

- [ ] **Step 4: Verify classifier model path compatibility**

Run:
```bash
# Check if existing classifier model is found by legacy fallback
python -c "
from backend.pipeline import load_classifier, get_device
try:
    c, m = load_classifier(get_device(), packaging_type='pack')
    print(f'Pack classifier loaded: {m[\"num_classes\"]} classes')
except FileNotFoundError as e:
    print(f'Expected: need to retrain. {e}')
"
```
Expected: Either loads from legacy path or reports need to retrain (both are acceptable -- the fallback path handles pre-migration classifiers)

- [ ] **Step 5: Check for broken cross-references in pipeline**

Run:
```bash
python -c "
# Verify run_pipeline still works (it calls _detect_brands_from_image internally)
from backend.pipeline import run_pipeline
print('run_pipeline importable, signature:', run_pipeline.__code__.co_varnames[:3])

# Verify output_format is unmodified
from backend.output_format import build_q12_row
print('output_format OK')
"
```
Expected: Both imports work, output_format unchanged

- [ ] **Step 6: Verify frontend can parse new API response format**

Check that the frontend correctly handles the new `packaging_type` field in crop responses (it should -- new fields are simply ignored by code that doesn't reference them, and the updated `_render_crop_card` uses `.get("packaging_type", "pack")` with a safe default).

- [ ] **Step 7: Commit final verification**

```bash
git add -A
git commit -m "Complete pack/box packaging type support with compatibility verification"
```

---

## Compatibility Checklist

| Component | Backward Compatible? | Notes |
|-----------|---------------------|-------|
| `brand_registry.py` | Yes | `audit_references()` returns new dict format but callers use `.get()` safely |
| `pipeline.py` | Yes | `load_classifier()` falls back to legacy flat path; `classify_embeddings()` defaults to `packaging_type="pack"` |
| `brand_classifier.py` | Yes | `--packaging-type` defaults to `"pack"`, old behavior preserved |
| `main.py` `/add-reference` | Yes | `packaging_type` defaults to `"pack"` in Form parameter |
| `main.py` `/generate-crops` | Yes | New `packaging_type` field added to response, doesn't break existing consumers |
| `main.py` `/reference-image/` | **Breaking** | URL changes from `/{filename}` to `/{packaging_type}/{filename}` -- but only frontend consumes this |
| `main.py` `/reference-images/` | Yes | `packaging_type` is a query param defaulting to `"pack"` |
| `main.py` `/brand-registry` | Yes | New `pack_count`/`box_count` fields added alongside existing `reference_count` |
| `output_format.py` | Unchanged | No changes needed -- both pack/box map to same brand/product |
| `train.py` | Unchanged | RF-DETR training is separate from classifier; class_id mapping depends on COCO annotation categories |
| `frontend/app.py` | Matched | Updated in sync with API changes |
