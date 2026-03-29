"""Download and extract a Roboflow dataset ZIP from a raw share URL.

Example:
  python download_roboflow_dataset.py --url "https://app.roboflow.com/ds/XXXXX?key=YYYYY"
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "cigarette_packs"


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract ZIP safely and block path traversal."""
    output_dir = output_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (output_dir / member.filename).resolve()
            if not str(member_path).startswith(str(output_dir)):
                raise RuntimeError(f"Unsafe zip member path detected: {member.filename}")
        zf.extractall(output_dir)


def _download_zip(url: str, out_zip: Path, timeout: int) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with out_zip.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _print_dataset_summary(output_dir: Path) -> None:
    split_names = ("train", "valid", "test")
    print("\nDataset summary:")
    for split in split_names:
        split_dir = output_dir / split
        ann_file = split_dir / "_annotations.coco.json"
        if not split_dir.exists():
            print(f"  - {split}: missing")
            continue
        image_count = 0
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_count += len(list(split_dir.glob(ext)))
            image_count += len(list(split_dir.glob(ext.upper())))
        print(
            f"  - {split}: {image_count} images | "
            f"annotations: {'yes' if ann_file.exists() else 'no'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Roboflow dataset ZIP from raw URL")
    parser.add_argument("--url", required=True, help="Roboflow raw dataset URL")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to extract dataset into",
    )
    parser.add_argument(
        "--zip-name",
        default="roboflow_dataset.zip",
        help="Temporary ZIP filename",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout (seconds)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before extraction",
    )
    args = parser.parse_args()

    parsed = urlparse(args.url)
    if "roboflow.com" not in parsed.netloc:
        raise ValueError("Expected a roboflow.com URL")
    qs = parse_qs(parsed.query)
    if "key" not in qs:
        raise ValueError("Roboflow URL must include ?key=...")

    output_dir = Path(args.output_dir).resolve()
    zip_path = output_dir.parent / args.zip_name

    if args.clean and output_dir.exists():
        # --clean: deletes output_dir on this machine before re-download (explicit CLI opt-in).
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from Roboflow URL: {args.url}")
    _download_zip(args.url, zip_path, timeout=args.timeout)
    print(f"Saved ZIP to: {zip_path}")

    print(f"Extracting ZIP to: {output_dir}")
    _safe_extract_zip(zip_path, output_dir)

    try:
        zip_path.unlink(missing_ok=True)
    except Exception:
        pass

    print("Extraction complete.")
    _print_dataset_summary(output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
