"""Pre-download all images from a batch CSV to a local cache directory.

Usage:
    python predownload_images.py <csv_path> [cache_dir]

Downloads images with 30 concurrent threads, saves as {image_id}.jpg.
The pipeline can then load from cache instead of hitting the network.
Progress is printed every 100 images.
"""
import csv
import os
import re
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 20
MAX_WORKERS = 30
MAX_RETRIES = 2


def extract_image_id(url: str) -> str | None:
    """Extract the ID parameter from an Ipsos image URL.
    Uses regex to match pipeline.py's _extract_image_id for consistency."""
    m = re.search(r'[?&]ID=(\d+)', url, re.IGNORECASE)
    return m.group(1) if m else None


def extract_all_urls(csv_path: str) -> list[tuple[str, str]]:
    """Extract all (image_id, url) pairs from a CSV."""
    url_pattern = re.compile(r"https?://", re.IGNORECASE)
    pairs = []
    seen_ids = set()

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if url_pattern.match(cell):
                    img_id = extract_image_id(cell)
                    if img_id and img_id not in seen_ids:
                        seen_ids.add(img_id)
                        pairs.append((img_id, cell))

    return pairs


def download_one(img_id: str, url: str, cache_dir: Path) -> tuple[str, bool, str]:
    """Download a single image. Returns (img_id, success, message)."""
    out_path = cache_dir / f"{img_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 0:
        return img_id, True, "cached"

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            if len(resp.content) < 100:
                return img_id, False, "too small (%d bytes)" % len(resp.content)
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return img_id, True, "downloaded"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1 * (attempt + 1))
                continue
            return img_id, False, str(e)[:80]

    return img_id, False, "max retries"


def get_cache_dir(csv_path: str | Path) -> Path:
    """Canonical image cache directory for a given CSV. Always next to the CSV file."""
    return Path(csv_path).parent / "image_cache"


def predownload(csv_path: str, cache_dir: str | None = None):
    csv_path = Path(csv_path)
    if cache_dir is None:
        cache_dir = get_cache_dir(csv_path)
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting URLs from %s", csv_path)
    pairs = extract_all_urls(str(csv_path))
    logger.info("Found %d unique images to download", len(pairs))

    # Check how many are already cached
    already = sum(1 for img_id, _ in pairs if (cache_dir / f"{img_id}.jpg").exists())
    logger.info("Already cached: %d, remaining: %d", already, len(pairs) - already)

    succeeded = already
    failed = 0
    total = len(pairs)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_one, img_id, url, cache_dir): img_id
            for img_id, url in pairs
        }
        for i, future in enumerate(as_completed(futures), 1):
            img_id, ok, msg = future.result()
            if ok:
                if msg != "cached":
                    succeeded += 1
            else:
                failed += 1

            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %.1f img/s | failed: %d | elapsed: %.0fs",
                    i, total, 100 * i / total, rate, failed, elapsed,
                )

    # Final count
    cached_count = sum(1 for img_id, _ in pairs if (cache_dir / f"{img_id}.jpg").exists())
    elapsed = time.time() - t0
    logger.info("Done. %d/%d cached, %d failed, %.0fs elapsed", cached_count, total, total - cached_count, elapsed)
    logger.info("Cache dir: %s", cache_dir)
    return cache_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predownload_images.py <csv_path> [cache_dir]")
        sys.exit(1)
    csv_path = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
    predownload(csv_path, cache_dir)
