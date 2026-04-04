"""Visualize Co-DETR detections with numbered bounding boxes and a legend panel.

Each detection gets a unique number + color on the image. A legend panel at the
bottom maps each number to its classified brand, confidence, and detection score.

Usage (on RunPod pod):
    cd /workspace/chhat-project
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_detections.py \
        --input backend/uploads/50_missed_pipeline_input.csv \
        --results backend/uploads/50_missed_pipeline_input_results.csv \
        --output backend/uploads/50_missed_annotated_results.csv
"""

import argparse
import csv
import os
import sys
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline import (
    load_codetr, detect_objects, load_dino, load_index,
    embed_images_batch, classify_embeddings, get_url_columns,
    CODETR_CONF_THRESHOLD, CLASSIFIER_TOP_K, get_device,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Distinct, high-contrast colors for up to 20 detections
COLORS = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (0, 130, 200),    # blue
    (255, 165, 0),    # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 180, 40),   # olive/yellow
    (0, 128, 128),    # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),   # brown
    (255, 250, 200),  # beige
    (128, 0, 0),      # maroon
    (170, 255, 195),  # mint
    (0, 0, 128),      # navy
    (255, 215, 180),  # coral
    (128, 128, 0),    # olive
    (255, 56, 56),    # light red
    (0, 255, 127),    # spring green
    (100, 149, 237),  # cornflower
]


def _load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def download_image(url: str, cache_dir: Path = None) -> Image.Image | None:
    if cache_dir:
        import hashlib
        fname = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        cached = cache_dir / fname
        if cached.exists():
            try:
                return Image.open(cached).convert("RGB")
            except Exception:
                pass
        for pattern in ["ID=", "id="]:
            if pattern in url:
                img_id = url.split(pattern)[1].split("&")[0]
                for f in cache_dir.glob(f"*{img_id}*"):
                    try:
                        return Image.open(f).convert("RGB")
                    except Exception:
                        continue
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to download {url[:80]}: {e}")
        return None


def annotate_image(image: Image.Image, detections: list, crop_labels: list,
                   crop_confidences: list) -> Image.Image:
    """Draw numbered bounding boxes on image + legend panel at bottom."""
    w, h = image.size
    n = len(detections)
    if n == 0:
        return image.copy()

    # Font sizes scale with image
    box_font_size = max(16, min(32, w // 30))
    legend_font_size = max(14, min(24, w // 40))
    box_font = _load_font(box_font_size)
    legend_font = _load_font(legend_font_size)

    # Build legend entries
    entries = []
    for i, (det, label, cls_conf) in enumerate(zip(detections, crop_labels, crop_confidences)):
        det_conf = det["confidence"]
        entries.append(f"#{i+1}  {label}  (cls: {cls_conf:.0%}, det: {det_conf:.0%})")

    # Calculate legend panel height
    line_height = legend_font_size + 8
    legend_padding = 12
    cols = 2 if n > 6 else 1
    rows_per_col = (n + cols - 1) // cols
    legend_h = legend_padding * 2 + rows_per_col * line_height + 10  # +10 for title

    # Create new image with legend panel at bottom
    new_h = h + legend_h
    canvas = Image.new("RGB", (w, new_h), (30, 30, 30))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw bounding boxes with numbers
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        color = COLORS[i % len(COLORS)]

        # Box outline (thick)
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        # Number badge (circle with number)
        num_text = str(i + 1)
        badge_size = box_font_size + 10
        badge_x = x1
        badge_y = max(0, y1 - badge_size - 4)

        # Circle background
        draw.ellipse(
            [badge_x, badge_y, badge_x + badge_size, badge_y + badge_size],
            fill=color, outline="white", width=2,
        )
        # Number centered in circle
        num_bbox = draw.textbbox((0, 0), num_text, font=box_font)
        num_w = num_bbox[2] - num_bbox[0]
        num_h = num_bbox[3] - num_bbox[1]
        draw.text(
            (badge_x + (badge_size - num_w) // 2, badge_y + (badge_size - num_h) // 2 - 2),
            num_text, fill="white", font=box_font,
        )

    # Draw legend panel
    legend_y_start = h + legend_padding
    col_width = (w - legend_padding * 2) // cols

    for i, entry in enumerate(entries):
        col = i // rows_per_col
        row = i % rows_per_col
        color = COLORS[i % len(COLORS)]

        x = legend_padding + col * col_width
        y = legend_y_start + row * line_height

        # Color swatch
        swatch_size = legend_font_size
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=color, outline="white")

        # Text
        draw.text((x + swatch_size + 8, y), entry, fill="white", font=legend_font)

    return canvas


def upload_to_spaces(img: Image.Image, key: str, s3_client, bucket: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    s3_client.put_object(
        Bucket=bucket, Key=key, Body=buf.getvalue(),
        ContentType="image/jpeg", ACL="public-read",
    )
    region = os.environ.get("DO_SPACES_REGION", "sgp1")
    return f"https://{bucket}.{region}.digitaloceanspaces.com/{key}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original pipeline input CSV")
    parser.add_argument("--results", required=True, help="Pipeline results CSV")
    parser.add_argument("--output", required=True, help="Output CSV with annotation URLs")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    s3 = boto3.client("s3",
        endpoint_url=os.environ["DO_SPACES_ENDPOINT"],
        aws_access_key_id=os.environ["DO_SPACES_KEY"],
        aws_secret_access_key=os.environ["DO_SPACES_SECRET"],
        region_name=os.environ.get("DO_SPACES_REGION", "sgp1"))
    bucket = os.environ["DO_SPACES_BUCKET"]

    device = get_device()
    logger.info(f"Device: {device}")
    load_codetr()
    processor, dino_model = load_dino(device)
    classifier, class_labels = load_index()

    import pandas as pd
    input_df = pd.read_csv(args.input, encoding="utf-8-sig")
    url_columns = get_url_columns(input_df)
    logger.info(f"URL columns: {url_columns}")

    cache_dir = Path(args.input).parent / "image_cache"
    if not cache_dir.is_dir():
        cache_dir = None
    else:
        logger.info(f"Image cache: {cache_dir}")

    results_rows = []
    with open(args.results, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        results_fieldnames = reader.fieldnames
        results_rows = list(reader)

    annotation_urls = []
    for row_idx, (_, input_row) in enumerate(input_df.iterrows()):
        serial = str(input_row.iloc[0])
        logger.info(f"[{row_idx+1}/{len(input_df)}] Serial {serial}")

        row_images = []
        for col in url_columns:
            url = str(input_row.get(col, "")).strip()
            if url and url.startswith("http"):
                img = download_image(url, cache_dir)
                if img:
                    row_images.append((col, img))

        if not row_images:
            logger.warning(f"  No images for {serial}")
            annotation_urls.append("")
            continue

        row_annotated_urls = []
        for img_idx, (col_name, image) in enumerate(row_images):
            dets = detect_objects(image, backend="codetr", threshold=CODETR_CONF_THRESHOLD)

            crop_labels = []
            crop_confidences = []
            width, height = image.size
            for det in dets:
                x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(width, x2 + pad_x)
                cy2 = min(height, y2 + pad_y)
                crop = image.crop((cx1, cy1, cx2, cy2))

                vecs = embed_images_batch([crop], processor, dino_model, device)
                cls_results = classify_embeddings(vecs, device, top_k=1, packaging_type="pack")
                if cls_results and cls_results[0]:
                    label = cls_results[0][0][0]
                    cls_conf = cls_results[0][0][1]
                    label = label.replace("_", " ").upper()
                else:
                    label = "UNKNOWN"
                    cls_conf = 0.0
                crop_labels.append(label)
                crop_confidences.append(cls_conf)

            annotated = annotate_image(image, dets, crop_labels, crop_confidences)

            s3_key = f"annotations/50missed/{serial}_{col_name}.jpg"
            url = upload_to_spaces(annotated, s3_key, s3, bucket)
            row_annotated_urls.append(url)
            logger.info(f"  {col_name}: {len(dets)} detections -> {url}")

        annotation_urls.append(" | ".join(row_annotated_urls))

    new_fieldnames = results_fieldnames + ["annotated_images"]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for i, row in enumerate(results_rows):
            row["annotated_images"] = annotation_urls[i] if i < len(annotation_urls) else ""
            writer.writerow(row)

    logger.info(f"\nDone! Output: {args.output}")
    logger.info(f"Total rows with annotations: {sum(1 for u in annotation_urls if u)}")


if __name__ == "__main__":
    main()
