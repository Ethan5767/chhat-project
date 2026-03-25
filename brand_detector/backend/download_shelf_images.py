import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

URL_COLUMN_PREFIXES = ("Q32 Photo Link", "Q35 Photo Link")
TIMEOUT = 10


def _valid_url(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    if text.lower() == "nan":
        return False
    return text.startswith("http://") or text.startswith("https://")


def _extension_from_content_type(content_type: str) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "bmp" in ct:
        return ".bmp"
    return ".jpg"


def main():
    parser = argparse.ArgumentParser(description="Download shelf images from Excel URL columns.")
    parser.add_argument("--excel", required=True, help="Path to the .xlsx file")
    parser.add_argument("--sheet", default="raw data", help="Sheet name containing URL columns")
    parser.add_argument(
        "--output",
        default="../datasets/raw_shelf_images",
        help="Output folder (default: datasets/raw_shelf_images when run from backend/)",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=1,
        help="0-based row index to use as column header inside the sheet.",
    )
    parser.add_argument("--workers", type=int, default=24, help="Concurrent download workers.")
    parser.add_argument("--max-images", type=int, default=1200, help="Max unique URLs to download.")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=args.sheet, header=args.header_row)
    url_cols = [c for c in df.columns if any(str(c).startswith(p) for p in URL_COLUMN_PREFIXES)]
    if not url_cols:
        raise ValueError(f"No URL columns found with prefixes: {URL_COLUMN_PREFIXES}")

    seen = set()
    urls = []
    for _, row in df.iterrows():
        for col in url_cols:
            value = row[col]
            if not _valid_url(value):
                continue
            url = str(value).strip()
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)

    urls = urls[: max(1, int(args.max_images))]

    def _download_one(item):
        i, url = item
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            ext = _extension_from_content_type(resp.headers.get("Content-Type", ""))
            h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
            filename = f"shelf_{i:05d}_{h}{ext}"
            (out_dir / filename).write_bytes(resp.content)
            return True
        except Exception:
            return False

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(_download_one, (i, url)) for i, url in enumerate(urls, start=1)]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), start=1):
            ok = fut.result()
            if ok:
                downloaded += 1
            else:
                failed += 1
            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] downloaded={downloaded} failed={failed}")

    print(f"Done. Unique URLs={len(urls)} downloaded={downloaded} failed={failed}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
