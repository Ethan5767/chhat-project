"""Visualize Co-DETR detections with bounding boxes and brand labels.

Reads a pipeline results CSV + original input CSV, re-runs detection on each image,
draws annotated bounding boxes, uploads to DO Spaces, and adds URLs to the CSV.

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline import (
    load_codetr, detect_objects, load_dino, load_index,
    embed_images_batch, classify_embeddings, get_url_columns,
    CODETR_CONF_THRESHOLD, CLASSIFIER_TOP_K, get_device,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Colors for bounding boxes (BGR-ish for variety)
COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0),
    (128, 0, 128), (0, 200, 200), (255, 20, 147), (0, 128, 0),
    (70, 130, 180), (220, 20, 60), (255, 215, 0), (138, 43, 226),
]


def download_image(url: str, cache_dir: Path = None) -> Image.Image | None:
    """Download image from URL, with optional local cache."""
    if cache_dir:
        # Check cache by URL hash
        import hashlib
        fname = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        cached = cache_dir / fname
        if cached.exists():
            try:
                return Image.open(cached).convert("RGB")
            except Exception:
                pass
        # Also check by image ID in URL
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


def annotate_image(image: Image.Image, detections: list, crop_labels: list) -> Image.Image:
    """Draw bounding boxes and labels on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

    for i, (det, label) in enumerate(zip(detections, crop_labels)):
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        conf = det["confidence"]
        color = COLORS[i % len(COLORS)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background + text
        text = f"{label} ({conf:.2f})"
        bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1 - 20), text, fill="white", font=font)

    return img


def upload_to_spaces(img: Image.Image, key: str, s3_client, bucket: str) -> str:
    """Upload PIL image to DO Spaces, return public URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    s3_client.put_object(
        Bucket=bucket, Key=key, Body=buf.getvalue(),
        ContentType="image/jpeg", ACL="public-read",
    )
    endpoint = os.environ.get("DO_SPACES_ENDPOINT", "")
    region = os.environ.get("DO_SPACES_REGION", "sgp1")
    # Construct public URL
    url = f"https://{bucket}.{region}.digitaloceanspaces.com/{key}"
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original pipeline input CSV")
    parser.add_argument("--results", required=True, help="Pipeline results CSV")
    parser.add_argument("--output", required=True, help="Output CSV with annotation URLs")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    # Init S3
    s3 = boto3.client("s3",
        endpoint_url=os.environ["DO_SPACES_ENDPOINT"],
        aws_access_key_id=os.environ["DO_SPACES_KEY"],
        aws_secret_access_key=os.environ["DO_SPACES_SECRET"],
        region_name=os.environ.get("DO_SPACES_REGION", "sgp1"))
    bucket = os.environ["DO_SPACES_BUCKET"]

    # Load models
    device = get_device()
    logger.info(f"Device: {device}")
    load_codetr()
    processor, dino_model = load_dino(device)
    classifier, class_labels = load_index()

    # Read input CSV for image URLs
    import pandas as pd
    input_df = pd.read_csv(args.input, encoding="utf-8-sig")
    url_columns = get_url_columns(input_df)
    logger.info(f"URL columns: {url_columns}")

    # Check for image cache
    cache_dir = Path(args.input).parent / "image_cache"
    if not cache_dir.is_dir():
        cache_dir = None
    else:
        logger.info(f"Image cache: {cache_dir}")

    # Read results CSV
    results_rows = []
    with open(args.results, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        results_fieldnames = reader.fieldnames
        results_rows = list(reader)

    # Process each row
    annotation_urls = []
    for row_idx, (_, input_row) in enumerate(input_df.iterrows()):
        serial = str(input_row.iloc[0])
        logger.info(f"[{row_idx+1}/{len(input_df)}] Serial {serial}")

        # Collect all images for this row
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

        # Process each image: detect + classify + annotate
        row_annotated_urls = []
        for img_idx, (col_name, image) in enumerate(row_images):
            # Detect
            dets = detect_objects(image, backend="codetr", threshold=CODETR_CONF_THRESHOLD)

            # Classify each detection
            crop_labels = []
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

                # Embed and classify
                vecs = embed_images_batch([crop], processor, dino_model, device)
                cls_results = classify_embeddings(vecs, device, top_k=1, packaging_type="pack")
                if cls_results and cls_results[0]:
                    label = cls_results[0][0][0]  # top-1 label
                    # Convert internal label to display name
                    label = label.replace("_", " ").upper()
                else:
                    label = "UNKNOWN"
                crop_labels.append(label)

            # Draw annotations
            annotated = annotate_image(image, dets, crop_labels)

            # Upload to DO Spaces
            s3_key = f"annotations/50missed/{serial}_{col_name}.jpg"
            url = upload_to_spaces(annotated, s3_key, s3, bucket)
            row_annotated_urls.append(url)
            logger.info(f"  {col_name}: {len(dets)} detections -> {url}")

        annotation_urls.append(" | ".join(row_annotated_urls))

    # Add annotation URLs to results CSV
    new_fieldnames = results_fieldnames + ["annotated_images"]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for i, row in enumerate(results_rows):
            row["annotated_images"] = annotation_urls[i] if i < len(annotation_urls) else ""
            writer.writerow(row)

    logger.info(f"\nDone! Output: {args.output}")
    logger.info(f"Total annotated images uploaded: {sum(1 for u in annotation_urls if u)}")


if __name__ == "__main__":
    main()
