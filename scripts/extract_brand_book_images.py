"""Extract product images from the KH Census Brand Book docx and save as brand_book_samples.

Reads the docx, maps each embedded image to its product name,
converts the display name to the internal name via brand_registry,
and saves images as {internal_name}_{N}.{ext} in backend/brand_book_samples/.
"""
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path so we can import brand_registry
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from backend.brand_registry import BRAND_REGISTRY


def build_display_to_internal() -> dict[str, str]:
    """Build mapping from uppercase display name -> internal name."""
    mapping = {}
    for brand, products in BRAND_REGISTRY.items():
        for display_name, internal_name in products:
            mapping[display_name.upper().strip()] = internal_name
    # Manual overrides for docx naming quirks
    mapping["ESSE IT'S BUBBLE PURPLE"] = "esse_its_bubble_purple"
    mapping["ESSE IT'S DEEP BROWN"] = "esse_its_deep_brown"
    mapping["GOLD SEAL MENTHOL KING SIZE"] = "gold_seal_menthol_kingsize"
    mapping["UANGHELOU"] = "huanghelou"
    mapping["L.A"] = "la"
    mapping["ESS  E RED"] = "esse_red"
    mapping["ESSE CHANGE DOUBLE (APPLEMINT/ORANGE)"] = "esse_change_double_applemint_orange"
    mapping["ESSE CHANGE DOUBLE (APPLEMINT/WINE)"] = "esse_change_double_applemint_wine"
    mapping["ESSE CHANGE DOUBLE CAFE APPLEMINT & ORANGE"] = "esse_change_double_cafe_applemint_orange"
    return mapping


def normalize_product_name(raw: str) -> str:
    """Clean up raw product name from docx cell text."""
    # Take first line (before Khmer text)
    name = raw.split("\n")[0].strip()
    # Remove Khmer/non-ASCII suffix after underscore
    if "_" in name:
        parts = name.split("_")
        # Keep the English part (check if second part is mostly ASCII)
        ascii_part = parts[0].strip()
        for p in parts[1:]:
            if p.strip() and all(ord(c) < 256 for c in p.strip()):
                ascii_part += " " + p.strip()
        name = ascii_part
    return name.upper().strip()


def extract_images(docx_path: str, output_dir: Path):
    doc = Document(docx_path)
    display_to_internal = build_display_to_internal()

    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    # Collect unique (rId, product_name) pairs
    seen_rids = {}  # rId -> product_name (first occurrence wins)
    image_entries = []  # [(rId, product_display_name)]

    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                drawings = cell._element.findall(".//w:drawing", ns)
                if not drawings:
                    continue

                # Find product name from surrounding cells
                product_name = ""
                cell_text = cell.text.strip()

                # Strategy: check this cell first (many tables have text+image in same cell)
                if cell_text:
                    product_name = cell_text.split("\n")[0].strip()

                # Check cell below (image rows often have text in next row)
                if not product_name and ri + 1 < len(table.rows):
                    next_cell = table.rows[ri + 1].cells[ci]
                    next_text = next_cell.text.strip()
                    if next_text:
                        product_name = next_text.split("\n")[0].strip()

                # Check cell above
                if not product_name and ri > 0:
                    prev_cell = table.rows[ri - 1].cells[ci]
                    prev_text = prev_cell.text.strip()
                    if prev_text:
                        product_name = prev_text.split("\n")[0].strip()

                for drawing in drawings:
                    blips = drawing.findall(".//a:blip", ns)
                    for blip in blips:
                        rId = blip.get(
                            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                        )
                        if rId and rId not in seen_rids:
                            seen_rids[rId] = product_name
                            image_entries.append((rId, product_name))

    # Now extract and save images
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = defaultdict(int)  # internal_name -> count
    skipped = []
    saved = []

    # Get the document's relationship parts
    part = doc.part

    for rId, raw_product in image_entries:
        normalized = normalize_product_name(raw_product)
        internal = display_to_internal.get(normalized)

        if not internal:
            # Try fuzzy matching - remove common suffixes
            for key, val in display_to_internal.items():
                if normalized.startswith(key) or key.startswith(normalized):
                    internal = val
                    break

        if not internal:
            # Try matching by removing "E-" vs "E_" differences
            alt = normalized.replace("-", " ").replace("_", " ")
            for key, val in display_to_internal.items():
                if key.replace("-", " ").replace("_", " ") == alt:
                    internal = val
                    break

        if not internal:
            skipped.append((rId, raw_product, normalized))
            continue

        # Get the image blob
        try:
            rel = part.rels[rId]
            image_blob = rel.target_part.blob
            content_type = rel.target_part.content_type
        except (KeyError, AttributeError) as e:
            skipped.append((rId, raw_product, f"blob error: {e}"))
            continue

        # Determine extension
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/webp": ".webp",
            "image/tiff": ".tiff",
            "image/x-emf": ".emf",
            "image/x-wmf": ".wmf",
        }
        ext = ext_map.get(content_type, ".png")

        # Skip EMF/WMF (vector formats, not useful as reference images)
        if ext in (".emf", ".wmf"):
            skipped.append((rId, raw_product, f"vector format: {content_type}"))
            continue

        counters[internal] += 1
        filename = f"{internal}_{counters[internal]}{ext}"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            f.write(image_blob)
        saved.append((filename, raw_product))

    return saved, skipped, counters


def main():
    docx_path = PROJECT_ROOT / "source_materials" / "KH Census_Brand Book - Updated31122025.docx"
    output_dir = PROJECT_ROOT / "backend" / "brand_book_samples"

    print(f"Extracting from: {docx_path}")
    print(f"Output to: {output_dir}")
    print()

    saved, skipped, counters = extract_images(str(docx_path), output_dir)

    print(f"Saved {len(saved)} images for {len(counters)} products:")
    for internal, count in sorted(counters.items()):
        print(f"  {internal}: {count}")

    if skipped:
        print(f"\nSkipped {len(skipped)} images:")
        for rId, raw, reason in skipped:
            print(f"  {rId}: '{raw}' -> {reason}")


if __name__ == "__main__":
    main()
