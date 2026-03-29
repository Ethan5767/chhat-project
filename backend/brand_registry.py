"""Complete brand and product registry for CHHAT cigarette detection.

Single source of truth for:
  - All 29 brands and 67 products from the client Excel
  - Internal reference filename conventions
  - Mapping between internal names and display names
  - What exists vs what's missing in reference images
"""
from pathlib import Path
from collections import Counter

try:
    from .paths import REFERENCES_DIR
except ImportError:
    from paths import REFERENCES_DIR
PACKAGING_TYPES = ("pack", "box")

# ─── Complete brand/product registry ───
# Structure: brand_name -> list of (product_display_name, internal_filename_prefix)
# internal_filename_prefix is what reference images are named: {prefix}_1.jpg, {prefix}_2.jpg, etc.

BRAND_REGISTRY = {
    "MEVIUS": [
        ("MEVIUS ORIGINAL", "mevius_original"),
        ("MEVIUS SKY BLUE", "mevius_sky_blue"),
        ("MEVIUS OPTION PURPLE", "mevius_option_purple"),
        ("MEVIUS FREEZY DEW", "mevius_freezy_dew"),
        ("MEVIUS OPTION PURPLE SUPER SLIMS", "mevius_option_purple_super_slims"),
        ("MEVIUS KIWAMI", "mevius_kimavi"),
        ("MEVIUS E-SERIES BLUE", "mevius_e_series_blue"),
        ("MEVIUS MINT FLOW", "mevius_mint_flow"),
    ],
    "WINSTON": [
        ("WINSTON NIGHT BLUE", "winston_night_blue"),
        ("WINSTON OPTION PURPLE", "winston_option_purple"),
        ("WINSTON OPTION BLUE", "winston_option_blue"),
    ],
    "ESSE": [
        ("ESSE CHANGE", "esse_change"),
        ("ESSE LIGHTS", "esse_light"),
        ("ESSE MENTHOL", "esse_menthol"),
        ("ESSE GOLD", "esse_gold"),
        ("ESSE OTHERS", "esse_other"),
    ],
    "FINE": [
        ("FINE RED HARD PACK", "fine_red_hard_pack"),
        ("FINE OTHERS", "fine_other"),
    ],
    "555": [
        ("555 SPHERE2 VELVETY", "555_sphere2_velvet"),
        ("555 ORIGINAL", "555_original"),
        ("555 GOLD", "555_gold"),
        ("555 OTHERS", "555_other"),
    ],
    "ARA": [
        ("ARA RED", "ara_red"),
        ("ARA GOLD", "ara_gold"),
        ("ARA MENTHOL", "ara_menthol"),
        ("ARA OTHERS", "ara_other"),
    ],
    "LUXURY": [
        ("LUXURY FULL FLAVOUR", "luxury_full_flavour"),
        ("LUXURY MENTHOL", "luxury_menthol"),
        ("LUXURY OTHERS", "luxury_other"),
    ],
    "GOLD SEAL": [
        ("GOLD SEAL MENTHOL COMPACT", "gold_seal_menthol_compact"),
        ("GOLD SEAL MENTHOL KINGSIZE", "gold_seal_menthol_kingsize"),
        ("GOLD SEAL OTHERS", "gold_seal_other"),
    ],
    "MARLBORO": [
        ("MARLBORO RED", "marlboro_red"),
        ("MARLBORO GOLD", "marlboro_gold"),
        ("MARLBORO OTHERS", "marlboro_other"),
    ],
    "CAMBO": [
        ("CAMBO CLASSICAL", "cambo_classical"),
        ("CAMBO FF", "cambo_ff"),
        ("CAMBO MENTHOL", "cambo_menthol"),
    ],
    "IZA": [
        ("IZA FF", "iza_ff"),
        ("IZA MENTHOL", "iza_menthol"),
        ("IZA OTHERS", "iza_other"),
    ],
    "HERO": [
        ("HERO HARD PACK", "hero"),
    ],
    "COW BOY": [
        ("COW BOY BLUEBERRY MINT", "cow_boy_blueberry_mint"),
        ("COW BOY HARD PACK", "cow_boy_hard_pack"),
        ("COW BOY MENTHOL", "cow_boy_menthol"),
        ("COW BOY OTHERS", "cow_boy_other"),
    ],
    "COCO PALM": [
        ("COCO PALM HARD PACK", "coco_palm_hard_pack"),
        ("COCO PALM MENTHOL", "coco_palm_menthol"),
        ("COCO PALM OTHERS", "coco_palm_other"),
    ],
    "CROWN": [
        ("CROWN", "crown"),
    ],
    "LAPIN": [
        ("LAPIN FF", "lapin_ff"),
        ("LAPIN MENTHOL", "lapin_menthol"),
    ],
    "ORIS": [
        ("ORIS PULSE BLUE", "oris_pulse_blue"),
        ("ORIS ICE PLUS", "oris_ice_plus"),
        ("ORIS SILVER", "oris_silver"),
        ("ORIS OTHERS", "oris_other"),
    ],
    "JET": [
        ("JET", "jet"),
    ],
    "L&M": [
        ("L&M", "l_and_m"),
    ],
    "DJARUM": [
        ("DJARUM", "djarum"),
    ],
    "LIBERATION": [
        ("LIBERATION", "liberation"),
    ],
    "MODERN": [
        ("MODERN", "modern"),
    ],
    "MOND": [
        ("MOND", "mond"),
    ],
    "NATIONAL": [
        ("NATIONAL", "national"),
    ],
    "CHUNGHWA": [
        ("CHUNGHWA", "chunghwa"),
    ],
    "SHUANGXI": [
        ("SHUANGXI", "shuangxi"),
    ],
    "YUN YAN": [
        ("YUN YAN", "yun_yan"),
    ],
    "CHINESE BRAND": [
        ("CHINESE BRANDS", "chinese_brand"),
    ],
    "OTHERS": [
        ("OTHERS", "other"),
    ],
}

# ─── Legacy name aliases ───
# Maps old/misspelled reference filenames to the correct registry name
LEGACY_ALIASES = {
    # Old misspellings from original dataset (all renamed already, kept for safety)
    "malboro_red": "marlboro_red",
    "malboro_gold": "marlboro_gold",
    "malboro_other": "marlboro_other",
    "gold_sea": "gold_seal_menthol_compact",
    "oris_sliver": "oris_silver",
    "cow_boy_bluberry_mint": "cow_boy_blueberry_mint",
    "esse_double_change": "esse_other",
    "mevius_kimavi": "mevius_kimavi",  # KIWAMI in Excel, KIMAVI in references — same product
    "other_4": "other",
    "galaxy": "other",
}


def _label_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


# ─── Derived lookups ───

# internal_name -> (brand, product_display_name)
INTERNAL_TO_BRAND_product: dict[str, tuple[str, str]] = {}
for brand, products in BRAND_REGISTRY.items():
    for product_display, internal in products:
        INTERNAL_TO_BRAND_product[internal] = (brand, product_display)

# product_display_name -> internal_name
DISPLAY_TO_INTERNAL: dict[str, str] = {}
for brand, products in BRAND_REGISTRY.items():
    for product_display, internal in products:
        DISPLAY_TO_INTERNAL[product_display] = internal

# internal_name -> product_display_name
INTERNAL_TO_DISPLAY: dict[str, str] = {}
for brand, products in BRAND_REGISTRY.items():
    for product_display, internal in products:
        INTERNAL_TO_DISPLAY[internal] = product_display


def resolve_internal_name(name: str) -> str:
    """Resolve a legacy/aliased name to the canonical internal name."""
    return LEGACY_ALIASES.get(name, name)


def get_brand(internal_name: str) -> str:
    """Get the parent brand for an internal product name."""
    resolved = resolve_internal_name(internal_name)
    if resolved in INTERNAL_TO_BRAND_product:
        return INTERNAL_TO_BRAND_product[resolved][0]
    return internal_name.upper().split("_")[0]


def get_display_name(internal_name: str) -> str:
    """Get the display product name for an internal name."""
    resolved = resolve_internal_name(internal_name)
    if resolved in INTERNAL_TO_DISPLAY:
        return INTERNAL_TO_DISPLAY[resolved]
    return internal_name.upper().replace("_", " ")


def audit_references() -> dict:
    """Check what's in backend/references/{pack,box}/ vs what the registry expects.

    Returns a dict with:
      - found: {internal_name: {"pack": count, "box": count}} for products with reference images
      - missing: [(brand, product_display, internal_name)] for products with 0 images in either type
      - unregistered: {filename_prefix: count} for images not in registry
      - legacy: {old_name: new_name} for images using old naming
      - per_type: {"pack": total_count, "box": total_count}
    """
    if not REFERENCES_DIR.exists():
        return {"found": {}, "missing": [], "unregistered": {}, "legacy": {}, "per_type": {}}

    all_found: dict[str, dict[str, int]] = {}
    total_images = 0
    all_label_counts = Counter()
    per_type_totals = {}

    for pkg_type in PACKAGING_TYPES:
        type_dir = REFERENCES_DIR / pkg_type
        if not type_dir.exists():
            per_type_totals[pkg_type] = 0
            continue

        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_paths.extend(type_dir.glob(ext))
            image_paths.extend(type_dir.glob(ext.upper()))

        per_type_totals[pkg_type] = len(image_paths)
        total_images += len(image_paths)

        file_labels = [_label_from_filename(p.name) for p in image_paths]
        label_counts = Counter(file_labels)
        all_label_counts += label_counts

        for label, count in label_counts.items():
            if label not in all_found:
                all_found[label] = {t: 0 for t in PACKAGING_TYPES}
            all_found[label][pkg_type] = count

    # Check each registry entry
    found = {}
    missing = []
    for brand, products in BRAND_REGISTRY.items():
        for product_display, internal in products:
            counts = all_found.get(internal, {t: 0 for t in PACKAGING_TYPES})
            # Also check legacy aliases
            for old, new in LEGACY_ALIASES.items():
                if new == internal and old in all_found:
                    for t in PACKAGING_TYPES:
                        counts[t] = counts.get(t, 0) + all_found[old].get(t, 0)
            total = sum(counts.values())
            if total > 0:
                found[internal] = counts
            else:
                missing.append((brand, product_display, internal))

    # Find images not in registry
    all_registered = set()
    for brand, products in BRAND_REGISTRY.items():
        for _, internal in products:
            all_registered.add(internal)
    all_registered.update(LEGACY_ALIASES.keys())

    unregistered = {}
    for label, count in all_label_counts.items():
        if label not in all_registered:
            unregistered[label] = count

    # Find legacy-named files
    legacy_found = {}
    for old, new in LEGACY_ALIASES.items():
        if old != new and all_label_counts.get(old, 0) > 0:
            legacy_found[old] = new

    return {
        "found": found,
        "missing": missing,
        "unregistered": unregistered,
        "legacy": legacy_found,
        "total_images": total_images,
        "total_products_found": len(found),
        "total_products_missing": len(missing),
        "per_type": per_type_totals,
    }


def print_audit():
    """Print a human-readable audit report."""
    result = audit_references()

    total_products = sum(len(products) for products in BRAND_REGISTRY.values())
    print(f"=== Brand Registry Audit ===")
    print(f"Registry: {len(BRAND_REGISTRY)} brands, {total_products} products")
    print(f"References: {result['total_images']} images")
    for pkg_type, count in result.get("per_type", {}).items():
        print(f"  {pkg_type}: {count} images")
    print(f"Products with images: {result['total_products_found']}/{total_products}")
    print(f"Products missing: {result['total_products_missing']}/{total_products}")

    if result["found"]:
        print(f"\n--- Products with references ({result['total_products_found']}) ---")
        for internal, counts in sorted(result["found"].items()):
            brand, display = INTERNAL_TO_BRAND_product.get(internal, ("?", internal))
            count_str = ", ".join(f"{t}={c}" for t, c in counts.items() if c > 0)
            total = sum(counts.values())
            print(f"  {display:<40} {total:>3} images  ({count_str})  ({internal})")

    if result["missing"]:
        print(f"\n--- MISSING Products ({result['total_products_missing']}) ---")
        for brand, product_display, internal in result["missing"]:
            print(f"  {brand:<15} {product_display:<40} (need: pack/{internal}_1.jpg)")

    if result["legacy"]:
        print(f"\n--- Legacy-named files (should rename) ---")
        for old, new in sorted(result["legacy"].items()):
            print(f"  {old} -> {new}")

    if result["unregistered"]:
        print(f"\n--- Unregistered (not in registry) ---")
        for label, count in sorted(result["unregistered"].items()):
            print(f"  {label}: {count} images")


if __name__ == "__main__":
    print_audit()
