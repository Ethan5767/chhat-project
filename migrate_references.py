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
            if img_path.parent != REFERENCES_DIR:
                continue
            dest = PACK_DIR / img_path.name
            shutil.move(str(img_path), str(dest))
            moved += 1

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
