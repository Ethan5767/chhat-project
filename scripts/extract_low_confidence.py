"""
Extract low-confidence rows from 12K batch results for annotation.
Downloads images and creates a summary CSV.
"""

import csv
import os
import re
import urllib.request
import urllib.error
import ssl
import random
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\kimto\OneDrive\Desktop\RE-AI\rf-detr-cigarette")
INPUT_CSV = PROJECT_ROOT / "source_materials" / "12000_batch_final_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "source material" / "low_confidence_for_annotation"
SUMMARY_CSV = PROJECT_ROOT / "source material" / "low_confidence_summary.csv"

URL_COLUMNS = ["Q30_1", "Q30_2", "Q30_3", "Q33_1", "Q33_2", "Q33_3"]

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SSL context to handle certificate issues
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def extract_image_id(url):
    """Extract ID=XXXXX from URL."""
    match = re.search(r'ID=(\d+)', url)
    return match.group(1) if match else None


def download_image(url, filepath):
    """Download image, return True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = resp.read()
            if len(data) < 1000:  # Too small, probably an error page
                return False
            with open(filepath, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        print(f"  Failed {filepath.name}: {e}")
        return False


# Read CSV and categorize rows
print("Reading CSV...")
uncertain_rows = []  # 0.50 <= conf <= 0.80
guessing_rows = []   # conf < 0.50

with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        conf_str = row.get("overall_confidence", "").strip()
        if not conf_str:
            continue
        try:
            conf = float(conf_str)
        except ValueError:
            continue

        if conf < 0.50:
            guessing_rows.append(row)
        elif conf <= 0.80:
            uncertain_rows.append(row)

print(f"Uncertain (0.50-0.80): {len(uncertain_rows)} rows")
print(f"Guessing (<0.50): {len(guessing_rows)} rows")

# Collect image URLs from selected rows
def collect_urls(rows):
    """Collect unique (image_id, url) pairs from rows."""
    urls = {}  # image_id -> url
    for row in rows:
        for col in URL_COLUMNS:
            url = row.get(col, "").strip()
            if url and url.startswith("http"):
                img_id = extract_image_id(url)
                if img_id and img_id not in urls:
                    urls[img_id] = url
    return urls


uncertain_urls = collect_urls(uncertain_rows)
guessing_urls = collect_urls(guessing_rows)

print(f"Unique images - uncertain: {len(uncertain_urls)}, guessing: {len(guessing_urls)}")

# Sample if needed: ~300 from uncertain, ~100 from guessing
if len(uncertain_urls) > 300:
    sampled_keys = random.sample(list(uncertain_urls.keys()), 300)
    uncertain_urls = {k: uncertain_urls[k] for k in sampled_keys}

if len(guessing_urls) > 100:
    sampled_keys = random.sample(list(guessing_urls.keys()), 100)
    guessing_urls = {k: guessing_urls[k] for k in sampled_keys}

print(f"After sampling - uncertain: {len(uncertain_urls)}, guessing: {len(guessing_urls)}")

# Merge for download
all_urls = {}
all_urls.update(guessing_urls)
all_urls.update(uncertain_urls)

print(f"Total images to download: {len(all_urls)}")

# Download images
downloaded = 0
failed = 0
for i, (img_id, url) in enumerate(all_urls.items()):
    filepath = OUTPUT_DIR / f"{img_id}.jpg"
    if filepath.exists():
        downloaded += 1
        continue

    if (i + 1) % 20 == 0:
        print(f"  Downloading {i+1}/{len(all_urls)} (ok={downloaded}, fail={failed})...")

    if download_image(url, filepath):
        downloaded += 1
    else:
        failed += 1

print(f"\nDownload complete: {downloaded} ok, {failed} failed")

# Write summary CSV
print("Writing summary CSV...")
all_low_conf_rows = uncertain_rows + guessing_rows
# Sort by confidence ascending (worst first)
all_low_conf_rows.sort(key=lambda r: float(r.get("overall_confidence", "0")))

with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Respondent.Serial", "overall_confidence", "image_urls"])

    for row in all_low_conf_rows:
        serial = row.get("Respondent.Serial", "")
        conf = row.get("overall_confidence", "")
        urls = []
        for col in URL_COLUMNS:
            url = row.get(col, "").strip()
            if url and url.startswith("http"):
                urls.append(url)
        writer.writerow([serial, conf, ";".join(urls)])

print(f"Summary CSV written: {len(all_low_conf_rows)} rows -> {SUMMARY_CSV}")
print("Done.")
