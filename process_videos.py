"""Extract frames from cigarette product videos, detect packs with RF-DETR, and save crops.

Pipeline:
  1. Walk VIDEO1/VIDEO2/VIDEO3 directories, derive product name from folder structure
  2. Extract frames at configurable FPS interval (default 1 fps)
  3. Skip near-duplicate frames via structural similarity (SSIM-lite)
  4. Run RF-DETR detection on each unique frame
  5. Crop detected packs with 5% padding
  6. Quality filter: min size, aspect ratio
  7. Save to video_data_set/{internal_name}_{index}.jpg

Folder structure assumptions:
  VIDEO1/VIDEO1/{N.BRAND}/{PRODUCT_NAME}/*.MOV   -- product-level folders
  VIDEO2/{N.BRAND}/{PRODUCT_NAME}/*.MOV           -- some have product folders
  VIDEO2/{N.BRAND}/*.MOV                          -- some have videos directly in brand
  VIDEO3/{N.BRAND}/*.MOV                          -- brand-level only
"""
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "source_materials"
OUTPUT_DIR = PROJECT_ROOT / "video_data_set"

# ─── Configuration ───
EXTRACT_FPS = 1.0          # Extract 1 frame per second
DETECTION_CONF = 0.25      # RF-DETR confidence threshold (fine-tuned model is calibrated)
RFDETR_CHECKPOINT = PROJECT_ROOT / "runs" / "checkpoint_best_ema.pth"
MIN_CROP_W = 50            # Minimum crop width
MIN_CROP_H = 60            # Minimum crop height
MIN_ASPECT = 0.25          # width/height minimum
MAX_ASPECT = 3.0           # width/height maximum
CROP_PAD = 0.05            # 5% bounding box padding
JPEG_QUALITY = 92
FRAME_DIFF_THRESHOLD = 0.92  # Skip frame if >92% similar to previous (via histogram correlation)
MAX_CROPS_PER_VIDEO = 50   # Safety cap per video file


# ─── Folder name to internal name mapping ───
# Maps folder product names to brand_registry internal_filename_prefix
# Handles spelling variations in folder names vs registry

FOLDER_TO_INTERNAL = {
    # VIDEO1 products
    "MEVIUS ORIGINAL": "mevius_original",
    "MEVIUS SKY BLUE": "mevius_sky_blue",
    "MEVIUS OPTION PURPLE": "mevius_option_purple",
    "MEVIUS FREEZY DEW": "mevius_freezy_dew",
    "MEVIUS KIMAVI": "mevius_kimavi",
    "WINSTON NIGHT BLUE": "winston_night_blue",
    "WINSTON OPTION PURPLE": "winston_option_purple",
    "WINSTON OPTION BLUE": "winston_option_blue",
    "ESSE CHANGE": "esse_change",
    "ESSE DOUBLE CHANGE": "esse_other",
    "ESSE LIGHT": "esse_light",
    "ESSE MENTHOL": "esse_menthol",
    "FINE RED HARD PACK": "fine_red_hard_pack",
    "FINE OTHER": "fine_other",
    "555 ORIGINAL": "555_original",
    "555 OTHER": "555_other",
    "555 SPHERE2 VELVET": "555_sphere2_velvet",
    "ARA RED": "ara_red",
    "ARA GOLD": "ara_gold",
    "ARA MENTHOL": "ara_menthol",
    "ARA OTHER": "ara_other",
    "LUXURY MENTHOL": "luxury_menthol",
    "LUXURY OTHER": "luxury_other",
    # VIDEO2 products
    "MALBORO RED": "marlboro_red",
    "MALBORO GOLD": "marlboro_gold",
    "MALBORO OTHER": "marlboro_other",
    "IZA MENTHOL": "iza_menthol",
    "IZA OTHER": "iza_other",
    "COW BOY BLUBERRY MINT": "cow_boy_blueberry_mint",
    "COW BOY MENTHOL": "cow_boy_menthol",
    "COW BOY OTHER": "cow_boy_other",
    "ORIS PULSE BLUE": "oris_pulse_blue",
    "ORIS ICE PLUS": "oris_ice_plus",
    "ORIS SLIVER": "oris_silver",
    # VIDEO2 brand-level (no product subfolder)
    "CAMBO": "cambo_classical",
    "GOLD SEA": "gold_seal_menthol_compact",
    "HERO": "hero",
    # VIDEO3 brand-level
    "MODERN": "modern",
    "MOND": "mond",
    "CHUNGHWA": "chunghwa",
    "GALAXY": "other",
    "OTHER (4)": "other",
}

# Fallback: strip number prefix and try direct brand mapping
BRAND_FALLBACK = {
    "MEVIUS": "mevius_original",
    "WINSTON": "winston_night_blue",
    "ESSE": "esse_change",
    "FINE": "fine_red_hard_pack",
    "555": "555_original",
    "ARA": "ara_red",
    "LUXURY": "luxury_full_flavour",
    "MARLBORO": "marlboro_red",
    "MALBORO": "marlboro_red",
    "CAMBO": "cambo_classical",
    "IZA": "iza_ff",
    "HERO": "hero",
    "COW BOY": "cow_boy_hard_pack",
    "ORIS": "oris_pulse_blue",
    "GOLD SEA": "gold_seal_menthol_compact",
    "GOLD SEAL": "gold_seal_menthol_compact",
    "MODERN": "modern",
    "MOND": "mond",
    "CHUNGHWA": "chunghwa",
    "GALAXY": "other",
    "OTHERS": "other",
}


def strip_number_prefix(name: str) -> str:
    """Strip leading number+dot prefix: '1.MEVIUS' -> 'MEVIUS'."""
    return re.sub(r"^\d+\.\s*", "", name).strip()


def resolve_internal_name(video_path: Path) -> str:
    """Derive internal product name from the video file's folder hierarchy."""
    rel = video_path.relative_to(SOURCE_DIR)
    parts = list(rel.parts)

    # Walk up from the video file, collecting meaningful folder names
    # Skip the VIDEO1/VIDEO2/VIDEO3 top-level and any nested VIDEO1 duplicate
    meaningful = []
    for p in parts[:-1]:  # exclude filename
        cleaned = strip_number_prefix(p)
        if cleaned.upper().startswith("VIDEO"):
            continue
        meaningful.append(cleaned.upper())

    if not meaningful:
        return "unknown"

    # Try product-level match first (last meaningful folder)
    product_folder = meaningful[-1]
    if product_folder in FOLDER_TO_INTERNAL:
        return FOLDER_TO_INTERNAL[product_folder]

    # Try brand-level match (first meaningful folder after VIDEO*)
    brand_folder = meaningful[0]
    if brand_folder in FOLDER_TO_INTERNAL:
        return FOLDER_TO_INTERNAL[brand_folder]

    # Fallback brand mapping
    if brand_folder in BRAND_FALLBACK:
        return BRAND_FALLBACK[brand_folder]

    # Last resort: normalize the deepest folder name
    name = product_folder.lower().replace(" ", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name or "unknown"


def frames_are_similar(frame1, frame2) -> bool:
    """Quick histogram-based similarity check to skip near-duplicate frames."""
    if frame1 is None or frame2 is None:
        return False
    # Convert to grayscale and resize for fast comparison
    g1 = cv2.cvtColor(cv2.resize(frame1, (128, 128)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(frame2, (128, 128)), cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [64], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return score > FRAME_DIFF_THRESHOLD


def collect_video_files() -> list[tuple[Path, str]]:
    """Find all MOV/mp4 video files and resolve their internal names."""
    video_dirs = [
        SOURCE_DIR / "VIDEO1",
        SOURCE_DIR / "VIDEO2",
        SOURCE_DIR / "VIDEO3",
    ]
    results = []
    for vdir in video_dirs:
        if not vdir.exists():
            print(f"  Warning: {vdir} not found, skipping")
            continue
        for ext in ("*.MOV", "*.mov", "*.mp4", "*.MP4", "*.avi", "*.AVI"):
            for vfile in sorted(vdir.rglob(ext)):
                internal = resolve_internal_name(vfile)
                results.append((vfile, internal))
    return results


def process_single_video(
    video_path: Path,
    internal_name: str,
    rfdetr_model,
    counters: dict[str, int],
    global_idx: dict[str, int],
) -> int:
    """Extract frames from one video, detect and crop packs. Returns count saved."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    SKIP: Cannot open {video_path.name}")
        counters["skipped_open"] += 1
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    frame_interval = max(1, int(fps / EXTRACT_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved = 0
    frame_num = 0
    prev_frame = None
    out_dir = OUTPUT_DIR / internal_name
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Only process at target FPS interval
        if (frame_num - 1) % frame_interval != 0:
            continue

        # Skip near-duplicate frames
        if frames_are_similar(frame, prev_frame):
            counters["skipped_duplicate"] += 1
            continue
        prev_frame = frame.copy()

        # Safety cap
        if saved >= MAX_CROPS_PER_VIDEO:
            break

        # Convert to PIL for RF-DETR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Detect
        try:
            detections = rfdetr_model.predict(pil_img, threshold=DETECTION_CONF)
        except Exception as e:
            counters["skipped_error"] += 1
            continue

        has_dets = detections is not None and len(detections) > 0
        if not has_dets:
            counters["skipped_no_det"] += 1
            continue

        # Process ALL detections in the frame (video frames may show single pack)
        xyxy = detections.xyxy
        confs = detections.confidence
        img_w, img_h = pil_img.size

        for det_idx in range(len(confs)):
            if saved >= MAX_CROPS_PER_VIDEO:
                break

            box = xyxy[det_idx]
            conf = float(confs[det_idx])
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Add padding
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * CROP_PAD), int(bh * CROP_PAD)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img_w, x2 + pad_x)
            y2 = min(img_h, y2 + pad_y)

            crop = pil_img.crop((x1, y1, x2, y2))
            cw, ch = crop.size

            # Size filter
            if cw < MIN_CROP_W or ch < MIN_CROP_H:
                counters["skipped_size"] += 1
                continue

            # Aspect ratio filter
            aspect = cw / ch
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                counters["skipped_aspect"] += 1
                continue

            # Save
            idx = global_idx.get(internal_name, 0) + 1
            global_idx[internal_name] = idx
            out_name = f"{internal_name}_{idx}.jpg"

            # Resize if too large (keep crops manageable)
            if max(cw, ch) > 800:
                scale = 800 / max(cw, ch)
                crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)

            crop.save(out_dir / out_name, "JPEG", quality=JPEG_QUALITY)
            saved += 1

    cap.release()
    return saved


def main():
    print("=" * 60)
    print("  VIDEO FRAME EXTRACTION & CROP PIPELINE")
    print("=" * 60)

    # Collect all video files
    print("\n[1/3] Scanning video files...")
    videos = collect_video_files()
    if not videos:
        print("No video files found in source_materials/VIDEO1, VIDEO2, VIDEO3")
        sys.exit(1)

    # Print summary of what we found
    from collections import Counter
    name_counts = Counter(name for _, name in videos)
    print(f"  Found {len(videos)} video files across {len(name_counts)} products:")
    for name, count in sorted(name_counts.items()):
        print(f"    {name}: {count} videos")

    # Prepare output directory (local OUTPUT_DIR only — not the whole project)
    if OUTPUT_DIR.exists():
        print(f"\n  Output dir exists, clearing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load RF-DETR
    print("\n[2/3] Loading fine-tuned RF-DETR model...")
    from rfdetr import RFDETRMedium
    if RFDETR_CHECKPOINT.exists():
        print(f"  Using fine-tuned checkpoint: {RFDETR_CHECKPOINT.name}")
        rfdetr = RFDETRMedium(pretrain_weights=str(RFDETR_CHECKPOINT))
    else:
        print("  WARNING: No fine-tuned checkpoint found, falling back to pretrained")
        rfdetr = RFDETRMedium()
    try:
        rfdetr.optimize_for_inference()
        print("  RF-DETR optimized for inference")
    except Exception:
        pass
    print("  RF-DETR loaded.")

    # Process all videos
    print("\n[3/3] Processing videos...")
    counters = {
        "skipped_open": 0,
        "skipped_duplicate": 0,
        "skipped_no_det": 0,
        "skipped_size": 0,
        "skipped_aspect": 0,
        "skipped_error": 0,
    }
    global_idx: dict[str, int] = {}
    total_saved = 0

    for i, (vpath, internal_name) in enumerate(videos, 1):
        print(f"\n  [{i}/{len(videos)}] {vpath.name} -> {internal_name}")
        saved = process_single_video(vpath, internal_name, rfdetr, counters, global_idx)
        total_saved += saved
        print(f"    Saved {saved} crops (total: {total_saved})")

    # Print final summary
    print("\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total crops saved:      {total_saved}")
    print(f"  Output directory:       {OUTPUT_DIR}")
    print(f"\n  Per-product breakdown:")
    for name, count in sorted(global_idx.items()):
        print(f"    {name}: {count} crops")
    print(f"\n  Skipped frames:")
    for key, val in sorted(counters.items()):
        print(f"    {key}: {val}")

    # Cleanup model
    import torch
    del rfdetr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone. Upload {OUTPUT_DIR.name}/ to the server.")


if __name__ == "__main__":
    main()
