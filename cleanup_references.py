"""
Clean up reference images by removing non-front-face views.

Moves side-view, top-edge, and other bad reference images to a backup folder.
These images pollute the FAISS index because they don't match what appears
on store shelves (front face of cigarette packs).

Criteria for removal:
- Aspect ratio > 1.2 (wide = top/bottom edge views)
- Aspect ratio < 0.5 (very tall = side edge views, barcode sides)
- Known bad images (hand-obscured, back-of-pack, etc.)
"""

import shutil
from pathlib import Path
from PIL import Image

REFERENCES_DIR = Path(__file__).resolve().parent / "backend" / "references"
REMOVED_DIR = Path(__file__).resolve().parent / "backend" / "references_removed"

# Images that are borderline by ratio but confirmed bad via visual inspection
KNOWN_BAD = {
    "luxury_menthol_1.jpg",   # back of pack with QR code, upside down
    "cambo_4.jpg",            # hand covering most of the pack
    "other_4_17.jpg",         # hand covering most of the pack
}


def main():
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        f for f in REFERENCES_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )

    removed = 0
    kept = 0
    reasons = {"wide": 0, "tall": 0, "known_bad": 0}

    for f in images:
        with Image.open(f) as img:
            w, h = img.size
        ratio = w / h
        remove = False
        reason = ""

        if f.name in KNOWN_BAD:
            remove = True
            reason = "known_bad"
        elif ratio > 1.2:
            remove = True
            reason = "wide"
        elif ratio < 0.5:
            remove = True
            reason = "tall"

        if remove:
            shutil.move(str(f), str(REMOVED_DIR / f.name))
            removed += 1
            reasons[reason] += 1
        else:
            kept += 1

    print(f"Kept: {kept} front-face images")
    print(f"Removed: {removed} total")
    print(f"  Wide (top/bottom edge): {reasons['wide']}")
    print(f"  Tall (side edge): {reasons['tall']}")
    print(f"  Known bad: {reasons['known_bad']}")


if __name__ == "__main__":
    main()
