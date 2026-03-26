"""
Fix uncropped reference images where RF-DETR missed the detection.

These are studio photos with white/gray backgrounds, pack centered, tripod visible.
Uses OpenCV to find the non-background object and crops to it.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

REFERENCES_DIR = Path(__file__).resolve().parent / "backend" / "references"
AREA_THRESHOLD = 2_000_000  # Images above this are considered uncropped
PADDING_RATIO = 0.03  # 3% padding around the detected object


def _find_center_contour(mask: np.ndarray, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """From a binary mask, find the contour closest to center and return its padded bbox."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = img_h * img_w
    img_center = np.array([img_w / 2, img_h / 2])
    min_area = img_area * 0.002

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + bw / 2
        cy = y + bh / 2
        dist = np.linalg.norm(np.array([cx, cy]) - img_center)
        candidates.append((dist, area, x, y, bw, bh))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    _, _, x, y, bw, bh = candidates[0]

    pad_x = int(bw * PADDING_RATIO)
    pad_y = int(bh * PADDING_RATIO)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + bw + pad_x)
    y2 = min(img_h, y + bh + pad_y)

    crop_area = (x2 - x1) * (y2 - y1)
    if crop_area > img_area * 0.5:
        return None

    return (x1, y1, x2, y2)


def find_object_bbox(img_cv: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the cigarette pack (object near center) using contours.

    Tries two approaches:
    1. Threshold-based (works for dark/colored packs on white background)
    2. Canny edge-based (works for white/light packs where threshold fails)
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = img_cv.shape[:2]

    # -- Approach 1: Threshold (dark objects on white bg) --
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    result = _find_center_contour(thresh, w, h)
    if result is not None:
        # Sanity: crop should be at least 2% of image area (reject tiny barcode-only crops)
        rx1, ry1, rx2, ry2 = result
        if (rx2 - rx1) * (ry2 - ry1) >= (h * w) * 0.02:
            return result

    # -- Approach 2: Canny edges, merge all center-region contours --
    # For white packs, gather all edge contours in the center 60% of the image
    # and merge their bounding boxes into one crop.
    blurred2 = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred2, 20, 80)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    edges = cv2.dilate(edges, kernel2, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Define center region (middle 60% of image)
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    img_area = h * w
    min_area = img_area * 0.0005

    center_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + bw / 2
        cy = y + bh / 2
        # Keep contours whose center is in the middle 60%
        if margin_x < cx < (w - margin_x) and margin_y < cy < (h - margin_y):
            center_contours.append(c)

    if not center_contours:
        return None

    # Merge all center contours into one bounding box
    all_points = np.vstack(center_contours)
    x, y, bw, bh = cv2.boundingRect(all_points)

    pad_x = int(bw * PADDING_RATIO)
    pad_y = int(bh * PADDING_RATIO)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    crop_area = (x2 - x1) * (y2 - y1)
    if crop_area > img_area * 0.5:
        return None

    return (x1, y1, x2, y2)


def main():
    images = sorted(
        f for f in REFERENCES_DIR.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )

    uncropped = []
    for f in images:
        img = Image.open(f)
        if img.size[0] * img.size[1] > AREA_THRESHOLD:
            uncropped.append(f)

    print(f"Found {len(uncropped)} uncropped images to fix")

    fixed = 0
    failed = 0

    for i, f in enumerate(uncropped, 1):
        img_cv = cv2.imread(str(f))
        if img_cv is None:
            print(f"  [{i}/{len(uncropped)}] SKIP (can't read): {f.name}")
            failed += 1
            continue

        bbox = find_object_bbox(img_cv)
        if bbox is None:
            print(f"  [{i}/{len(uncropped)}] FAILED (no object found): {f.name}")
            failed += 1
            continue

        x1, y1, x2, y2 = bbox
        cropped = img_cv[y1:y2, x1:x2]

        # Save back (overwrite)
        cv2.imwrite(str(f), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        fixed += 1

        if i % 20 == 0 or i == len(uncropped):
            print(f"  [{i}/{len(uncropped)}] fixed={fixed}, failed={failed}")

    print(f"\nDone! Fixed {fixed}, failed {failed} out of {len(uncropped)}")


if __name__ == "__main__":
    main()
