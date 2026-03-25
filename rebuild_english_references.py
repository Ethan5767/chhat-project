"""
Extract images from the Word brand book and name them using English text from the
document (the part after '|' in each caption cell; if missing, Latin text only).
Writes to project references/ and brand_detector/backend/references/.
"""
from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "KH Census_Brand Book - Updated31122025.docx"
MANIFEST = ROOT / "references" / "image_brand_manifest.csv"
OUT_DIRS = [
    ROOT / "references",
    ROOT / "brand_detector" / "backend" / "references",
]
KHMER_RE = re.compile(r"[\u1780-\u17FF]+")


def english_from_brand_text(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "unknown"
    if "|" in raw:
        return raw.split("|", 1)[1].strip()
    no_khmer = KHMER_RE.sub(" ", raw)
    no_khmer = re.sub(r"\s+", " ", no_khmer).strip()
    if no_khmer:
        return no_khmer
    return "unknown"


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def clear_images(folder: Path) -> None:
    if not folder.is_dir():
        return
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            p.unlink()


def main() -> None:
    rows = []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            rows.append(rec)

    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        clear_images(out)

    raw_dir = ROOT / "references" / "_extract_tmp"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DOCX, "r") as zf:
        for info in zf.infolist():
            if info.filename.startswith("word/media/") and not info.is_dir():
                (raw_dir / Path(info.filename).name).write_bytes(zf.read(info.filename))

    slug_counts: dict[str, int] = {}
    rename_rows: list[tuple[str, str, str, str]] = []
    copied = 0

    for rec in rows:
        media = (rec.get("media_file") or "").strip()
        txt = (rec.get("brand_candidate_text") or "").strip()
        if not media:
            continue
        src = raw_dir / media
        if not src.is_file():
            continue

        english = english_from_brand_text(txt)
        base_slug = slugify(english)
        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        n = slug_counts[base_slug]
        stem = base_slug if n == 1 else f"{base_slug}_{n}"

        ext = src.suffix.lower() or ".jpg"
        dst_name = f"{stem}{ext}"

        for out in OUT_DIRS:
            shutil.copy2(src, out / dst_name)
        copied += 1
        rename_rows.append((media, txt, english, dst_name))

    shutil.rmtree(raw_dir, ignore_errors=True)

    map_path = ROOT / "references" / "english_rename_map.csv"
    with map_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["media_file", "brand_candidate_text", "english_name_used", "filename"])
        w.writerows(rename_rows)

    shutil.copy2(map_path, ROOT / "brand_detector" / "backend" / "references" / "english_rename_map.csv")

    print(f"copied_images={copied}")
    print(f"map={map_path}")


if __name__ == "__main__":
    main()
