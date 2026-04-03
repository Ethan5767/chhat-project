"""Download all images from the 12k batch CSV into source_materials/12k_images/.

Usage:
    python scripts/download_12k_images.py
    python scripts/download_12k_images.py --workers 16
"""
import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CSV_PATH = Path(__file__).resolve().parent.parent / "source_materials" / "12000_batch.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "source_materials" / "12k_images"
URL_COLUMNS = ["Q30_1", "Q30_2", "Q30_3", "Q33_1", "Q33_2", "Q33_3"]
TIMEOUT = 30
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _is_url(val: str) -> bool:
    return bool(val and _URL_RE.match(val.strip()))


def _safe_filename(row_idx: int, col: str, url: str) -> str:
    """Generate a unique filename from row index, column name, and URL."""
    # Extract image ID from URL if possible
    match = re.search(r"ID=(\d+)", url)
    img_id = match.group(1) if match else str(hash(url) % 10**8)
    return f"row{row_idx:05d}_{col}_{img_id}.jpg"


def download_one(row_idx: int, col: str, url: str, out_dir: Path) -> tuple[bool, str]:
    fname = _safe_filename(row_idx, col, url)
    fpath = out_dir / fname
    if fpath.exists() and fpath.stat().st_size > 1000:
        return True, f"[skip] {fname} (exists)"
    try:
        resp = requests.get(url.strip(), timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        content = resp.content
        if len(content) < 500:
            return False, f"[fail] {fname} -- too small ({len(content)} bytes)"
        fpath.write_bytes(content)
        return True, f"[ok] {fname} ({len(content) // 1024}KB)"
    except Exception as exc:
        return False, f"[fail] {fname} -- {exc}"


def main():
    parser = argparse.ArgumentParser(description="Download 12k batch images")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all URLs
    tasks = []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            for col in URL_COLUMNS:
                val = row.get(col, "")
                if _is_url(val):
                    tasks.append((row_idx, col, val.strip()))

    print(f"Found {len(tasks)} image URLs across {URL_COLUMNS}")
    print(f"Downloading to {OUTPUT_DIR} with {args.workers} workers...")

    ok_count = 0
    fail_count = 0
    skip_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, row_idx, col, url, OUTPUT_DIR): (row_idx, col)
            for row_idx, col, url in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            success, msg = future.result()
            if "skip" in msg:
                skip_count += 1
            elif success:
                ok_count += 1
            else:
                fail_count += 1
            if i % 200 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] ok={ok_count} skip={skip_count} fail={fail_count}")

    print(f"\nDone. ok={ok_count} skip={skip_count} fail={fail_count}")
    print(f"Images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
