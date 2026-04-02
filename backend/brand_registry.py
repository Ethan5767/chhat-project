"""Complete brand and product registry for CHHAT cigarette detection.

Single source of truth for:
  - All 29 brands and 67 products from the client Excel
  - Internal reference filename conventions
  - Mapping between internal names and display names
  - What exists vs what's missing in reference images
"""
import os
from pathlib import Path
from collections import Counter

_BACKEND_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = Path(os.environ.get("CHHAT_DATA_ROOT", str(_BACKEND_ROOT)))
REFERENCES_DIR = _DATA_ROOT / "references"
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
        # Main products (have their own Q12B columns)
        ("ESSE CHANGE", "esse_change"),
        ("ESSE LIGHTS", "esse_light"),
        ("ESSE MENTHOL", "esse_menthol"),
        ("ESSE GOLD", "esse_gold"),
        # Granular ESSE OTHERS (each trained separately, output -> ESSE OTHERS)
        ("ESSE CHANGE CAFE", "esse_change_cafe"),
        ("ESSE BLACK", "esse_black"),
        ("ESSE RED", "esse_red"),
        ("ESSE CHANGE COOLIPS SWEET APPLE", "esse_change_coolips_sweet_apple"),
        ("ESSE CHANGE DOUBLE APPLEMINT ORANGE", "esse_change_double_applemint_orange"),
        ("ESSE CHANGE DOUBLE APPLEMINT WINE", "esse_change_double_applemint_wine"),
        ("ESSE CHANGE DOUBLE CAFE APPLEMINT ORANGE", "esse_change_double_cafe_applemint_orange"),
        ("ESSE CHANGE DOUBLE SHOT", "esse_change_double_shot"),
        ("ESSE CHANGE FROZEN PEACH MOJITO", "esse_change_frozen_peach_mojito"),
        ("ESSE CHANGE HIMALAYA", "esse_change_himalaya"),
        ("ESSE CHANGE MANGO", "esse_change_mango"),
        ("ESSE CHANGE SHOOTING RED COLA", "esse_change_shooting_red_cola"),
        ("ESSE CHANGE STRAWBERRY", "esse_change_strawberry"),
        ("ESSE GREEN", "esse_green"),
        ("ESSE ITS BUBBLE PURPLE", "esse_its_bubble_purple"),
        ("ESSE ITS DEEP BROWN", "esse_its_deep_brown"),
    ],
    "FINE": [
        ("FINE RED HARD PACK", "fine_red_hard_pack"),
        # Granular FINE OTHERS (output -> FINE OTHERS)
        ("FINE GOLD", "fine_gold"),
        ("FINE MENTHOL", "fine_menthol"),
        ("FINE CHARCOAL FILTER", "fine_charcoal_filter"),
    ],
    "555": [
        # Main products
        ("555 SPHERE2 VELVETY", "555_sphere2_velvet"),
        ("555 ORIGINAL", "555_original"),
        ("555 GOLD", "555_gold"),
        # Granular 555 OTHERS (output -> 555 OTHERS)
        ("555 REFINED CHARCOAL FILTER", "555_refined_charcoal_filter"),
        ("555 SWITCH", "555_switch"),
        ("555 BERRY BOOST", "555_berry_boost"),
        ("555 PRESTIGE", "555_prestige"),
        ("555 SLIM", "555_slim"),
        ("555 SPHERE2 SPARKY", "555_sphere2_sparky"),
    ],
    "ARA": [
        # Main products
        ("ARA RED", "ara_red"),
        ("ARA GOLD", "ara_gold"),
        ("ARA MENTHOL", "ara_menthol"),
        # Granular ARA OTHERS (output -> ARA OTHERS)
        ("ARA COOL", "ara_cool"),
        ("ARA TROPICAL", "ara_tropical"),
        ("ARA NEXT", "ara_next"),
        ("ARA ORIGINAL", "ara_original"),
        ("ARA PREMIER", "ara_premier"),
    ],
    "LUXURY": [
        # Main products
        ("LUXURY FULL FLAVOUR", "luxury_full_flavour"),
        ("LUXURY MENTHOL", "luxury_menthol"),
        # Granular LUXURY OTHERS (output -> LUXURY OTHERS)
        ("LUXURY SS DOUBLE CAPSULE", "luxury_ss_double_capsule"),
        ("LUXURY BLUEBERRY MINT OPTION", "luxury_blueberry_mint_option"),
        ("LUXURY ORANGE OPTION", "luxury_orange_option"),
        ("LUXURY BLUEBERRY OPTION", "luxury_blueberry_option"),
        ("LUXURY LIGHTS", "luxury_lights"),
        ("LUXURY MENTHOL OPTION", "luxury_menthol_option"),
        ("LUXURY PREMIUM OPTION BLUEBERRY", "luxury_premium_option_blueberry"),
        ("LUXURY SPECIAL BLEND", "luxury_special_blend"),
    ],
    "GOLD SEAL": [
        # Main products
        ("GOLD SEAL MENTHOL COMPACT", "gold_seal_menthol_compact"),
        ("GOLD SEAL MENTHOL KINGSIZE", "gold_seal_menthol_kingsize"),
        # Granular GOLD SEAL OTHERS (output -> GOLD SEAL OTHERS)
        ("GOLD SEAL FULL FLAVOR", "gold_seal_full_flavor"),
        ("GOLD SEAL GOLD MIDI SLIMS", "gold_seal_gold_midi_slims"),
        ("GOLD SEAL CLASSIC RED", "gold_seal_classic_red"),
        ("GOLD SEAL SPECIAL GOLD", "gold_seal_special_gold"),
    ],
    "MARLBORO": [
        # Main products
        ("MARLBORO RED", "marlboro_red"),
        ("MARLBORO GOLD", "marlboro_gold"),
        # Granular MARLBORO OTHERS (output -> MARLBORO OTHERS)
        ("MARLBORO MENTHOL", "marlboro_menthol"),
        ("MARLBORO ICE BLAST", "marlboro_ice_blast"),
        ("MARLBORO GOLD ADVANCE", "marlboro_gold_advance"),
        ("MARLBORO SPLASH", "marlboro_splash"),
        ("MARLBORO VISTA FOREST", "marlboro_vista_forest"),
        ("MARLBORO VISTA ICE BLAST", "marlboro_vista_ice_blast"),
    ],
    "CAMBO": [
        ("CAMBO CLASSICAL", "cambo_classical"),
        ("CAMBO FF", "cambo_ff"),
        ("CAMBO MENTHOL", "cambo_menthol"),
    ],
    "IZA": [
        # Main products
        ("IZA FF", "iza_ff"),
        ("IZA MENTHOL", "iza_menthol"),
        # Granular IZA OTHERS (output -> IZA OTHERS)
        ("IZA LIGHTS", "iza_lights"),
        ("IZA DOUBLE BURST", "iza_double_burst"),
    ],
    "HERO": [
        ("HERO HARD PACK", "hero"),
    ],
    "COW BOY": [
        # Main products
        ("COW BOY BLUEBERRY MINT", "cow_boy_blueberry_mint"),
        ("COW BOY HARD PACK", "cow_boy_hard_pack"),
        ("COW BOY MENTHOL", "cow_boy_menthol"),
        # Granular COW BOY OTHERS (output -> COW BOY OTHERS)
        ("COW BOY LIGHTS", "cow_boy_lights"),
        ("COW BOY SUPER SLIMS", "cow_boy_super_slims"),
    ],
    "COCO PALM": [
        # Main products
        ("COCO PALM HARD PACK", "coco_palm_hard_pack"),
        ("COCO PALM MENTHOL", "coco_palm_menthol"),
        # Granular COCO PALM OTHERS (output -> COCO PALM OTHERS)
        ("COCO PALM GOLD", "coco_palm_gold"),
    ],
    "CROWN": [
        ("CROWN", "crown"),
    ],
    "LAPIN": [
        ("LAPIN FF", "lapin_ff"),
        ("LAPIN MENTHOL", "lapin_menthol"),
    ],
    "ORIS": [
        # Main products
        ("ORIS PULSE BLUE", "oris_pulse_blue"),
        ("ORIS ICE PLUS", "oris_ice_plus"),
        ("ORIS SILVER", "oris_silver"),
        # Granular ORIS OTHERS (output -> ORIS OTHERS)
        ("ORIS PULSE", "oris_pulse"),
        ("ORIS MENTHOL", "oris_menthol"),
        ("ORIS AZURE BLUE", "oris_azure_blue"),
        ("ORIS BLACK", "oris_black"),
        ("ORIS FINE RED", "oris_fine_red"),
        ("ORIS INTENSE BLACK CURRANT", "oris_intense_black_currant"),
        ("ORIS INTENSE DEEP MIX", "oris_intense_deep_mix"),
        ("ORIS INTENSE GUAVA", "oris_intense_guava"),
        ("ORIS INTENSE PURPLE FIZZ", "oris_intense_purple_fizz"),
        ("ORIS INTENSE SUMMER FIZZ", "oris_intense_summer_fizz"),
        ("ORIS INTENSE TROPICAL DEW", "oris_intense_tropical_dew"),
        ("ORIS PULSE APPLEMINT ORANGE", "oris_pulse_applemint_orange"),
        ("ORIS PULSE MENTHOL ORANGE", "oris_pulse_menthol_orange"),
        ("ORIS PULSE SUPER SLIMS STRAWBERRY FUSION", "oris_pulse_super_slims_strawberry_fusion"),
        ("ORIS RED", "oris_red"),
        ("ORIS SLIMS CHOCOLATE", "oris_slims_chocolate"),
        ("ORIS SLIMS GOLD", "oris_slims_gold"),
        ("ORIS SLIMS STRAWBERRY", "oris_slims_strawberry"),
        ("ORIS TWIN SENSE BERRY MIX", "oris_twin_sense_berry_mix"),
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
        # Granular Chinese brands (output -> CHINESE BRANDS)
        ("DIAMOND HEHUA", "diamond_hehua"),
        ("DOUBLE HAPPINESS", "double_happiness"),
        ("HARMONIZATION", "harmonization"),
        ("HOMATA TOBACCO GROUP", "homata_tobacco_group"),
        ("HUANGHELOU", "huanghelou"),
        ("LIGUN", "ligun"),
    ],
    "OTHERS": [
        # Granular OTHERS brands from brand book (output -> OTHERS)
        ("ANGKOR MEAS", "angkor_meas"),
        ("ASIA STAR", "asia_star"),
        ("BLACK DEVIL", "black_devil"),
        ("CORSET", "corset"),
        ("DAVIDOFF", "davidoff"),
        ("DEMOCRAT", "democrat"),
        ("DOLCE", "dolce"),
        ("DUNHILL", "dunhill"),
        ("ENGLISHMAN", "englishman"),
        ("GALAXY", "galaxy"),
        ("GD", "gd"),
        ("GUDANG GARAM", "gudang_garam"),
        ("JUNE", "june"),
        ("KENTUCKY SELECT", "kentucky_select"),
        ("KODE", "kode"),
        ("LA", "la"),
        ("LUCKY STRIKE", "lucky_strike"),
        ("MARINER", "mariner"),
        ("MILANO", "milano"),
        ("OKI", "oki"),
        ("PHILIP MORRIS", "philip_morris"),
        ("PHNOM MEAS", "phnom_meas"),
        ("PINE", "pine"),
        ("RICHMAN", "richman"),
        ("RICHSONS", "richsons"),
        ("SAIGON", "saigon"),
        ("TETON", "teton"),
        ("TEXAS 5", "texas_5"),
        ("THANG LONG", "thang_long"),
        ("VIBE", "vibe"),
        ("WEST", "west"),
        ("WONDER", "wonder"),
        ("YELLOW ELEPHANT", "yellow_elephant"),
        ("ZEST YOGO", "zest_yogo"),
        ("ZOUK", "zouk"),
    ],
}

# ─── Output product mapping ───
# Maps granular internal names to the Q12B output product name.
# Products NOT listed here use their display name directly (they have their own Q12B column).
# Products listed here are "sub-products" that roll up to the brand's OTHERS column in output.
OUTPUT_PRODUCT_MAP = {
    # ESSE sub-products -> ESSE OTHERS
    "esse_change_cafe": "ESSE OTHERS",
    "esse_black": "ESSE OTHERS",
    "esse_red": "ESSE OTHERS",
    "esse_change_coolips_sweet_apple": "ESSE OTHERS",
    "esse_change_double_applemint_orange": "ESSE OTHERS",
    "esse_change_double_applemint_wine": "ESSE OTHERS",
    "esse_change_double_cafe_applemint_orange": "ESSE OTHERS",
    "esse_change_double_shot": "ESSE OTHERS",
    "esse_change_frozen_peach_mojito": "ESSE OTHERS",
    "esse_change_himalaya": "ESSE OTHERS",
    "esse_change_mango": "ESSE OTHERS",
    "esse_change_shooting_red_cola": "ESSE OTHERS",
    "esse_change_strawberry": "ESSE OTHERS",
    "esse_green": "ESSE OTHERS",
    "esse_its_bubble_purple": "ESSE OTHERS",
    "esse_its_deep_brown": "ESSE OTHERS",
    # FINE sub-products -> FINE OTHERS
    "fine_gold": "FINE OTHERS",
    "fine_menthol": "FINE OTHERS",
    "fine_charcoal_filter": "FINE OTHERS",
    # 555 sub-products -> 555 OTHERS
    "555_refined_charcoal_filter": "555 OTHERS",
    "555_switch": "555 OTHERS",
    "555_berry_boost": "555 OTHERS",
    "555_prestige": "555 OTHERS",
    "555_slim": "555 OTHERS",
    "555_sphere2_sparky": "555 OTHERS",
    # ARA sub-products -> ARA OTHERS
    "ara_cool": "ARA OTHERS",
    "ara_tropical": "ARA OTHERS",
    "ara_next": "ARA OTHERS",
    "ara_original": "ARA OTHERS",
    "ara_premier": "ARA OTHERS",
    # LUXURY sub-products -> LUXURY OTHERS
    "luxury_ss_double_capsule": "LUXURY OTHERS",
    "luxury_blueberry_mint_option": "LUXURY OTHERS",
    "luxury_orange_option": "LUXURY OTHERS",
    "luxury_blueberry_option": "LUXURY OTHERS",
    "luxury_lights": "LUXURY OTHERS",
    "luxury_menthol_option": "LUXURY OTHERS",
    "luxury_premium_option_blueberry": "LUXURY OTHERS",
    "luxury_special_blend": "LUXURY OTHERS",
    # GOLD SEAL sub-products -> GOLD SEAL OTHERS
    "gold_seal_full_flavor": "GOLD SEAL OTHERS",
    "gold_seal_gold_midi_slims": "GOLD SEAL OTHERS",
    "gold_seal_classic_red": "GOLD SEAL OTHERS",
    "gold_seal_special_gold": "GOLD SEAL OTHERS",
    # MARLBORO sub-products -> MARLBORO OTHERS
    "marlboro_menthol": "MARLBORO OTHERS",
    "marlboro_ice_blast": "MARLBORO OTHERS",
    "marlboro_gold_advance": "MARLBORO OTHERS",
    "marlboro_splash": "MARLBORO OTHERS",
    "marlboro_vista_forest": "MARLBORO OTHERS",
    "marlboro_vista_ice_blast": "MARLBORO OTHERS",
    # IZA sub-products -> IZA OTHERS
    "iza_lights": "IZA OTHERS",
    "iza_double_burst": "IZA OTHERS",
    # COW BOY sub-products -> COW BOY OTHERS
    "cow_boy_lights": "COW BOY OTHERS",
    "cow_boy_super_slims": "COW BOY OTHERS",
    # COCO PALM sub-products -> COCO PALM OTHERS
    "coco_palm_gold": "COCO PALM OTHERS",
    # ORIS sub-products -> ORIS OTHERS
    "oris_pulse": "ORIS OTHERS",
    "oris_menthol": "ORIS OTHERS",
    "oris_azure_blue": "ORIS OTHERS",
    "oris_black": "ORIS OTHERS",
    "oris_fine_red": "ORIS OTHERS",
    "oris_intense_black_currant": "ORIS OTHERS",
    "oris_intense_deep_mix": "ORIS OTHERS",
    "oris_intense_guava": "ORIS OTHERS",
    "oris_intense_purple_fizz": "ORIS OTHERS",
    "oris_intense_summer_fizz": "ORIS OTHERS",
    "oris_intense_tropical_dew": "ORIS OTHERS",
    "oris_pulse_applemint_orange": "ORIS OTHERS",
    "oris_pulse_menthol_orange": "ORIS OTHERS",
    "oris_pulse_super_slims_strawberry_fusion": "ORIS OTHERS",
    "oris_red": "ORIS OTHERS",
    "oris_slims_chocolate": "ORIS OTHERS",
    "oris_slims_gold": "ORIS OTHERS",
    "oris_slims_strawberry": "ORIS OTHERS",
    "oris_twin_sense_berry_mix": "ORIS OTHERS",
    # CHINESE BRAND sub-brands -> CHINESE BRANDS
    "diamond_hehua": "CHINESE BRANDS",
    "double_happiness": "CHINESE BRANDS",
    "harmonization": "CHINESE BRANDS",
    "homata_tobacco_group": "CHINESE BRANDS",
    "huanghelou": "CHINESE BRANDS",
    "ligun": "CHINESE BRANDS",
    # OTHERS sub-brands -> OTHERS
    "angkor_meas": "OTHERS",
    "asia_star": "OTHERS",
    "black_devil": "OTHERS",
    "corset": "OTHERS",
    "davidoff": "OTHERS",
    "democrat": "OTHERS",
    "dolce": "OTHERS",
    "dunhill": "OTHERS",
    "englishman": "OTHERS",
    "galaxy": "OTHERS",
    "gd": "OTHERS",
    "gudang_garam": "OTHERS",
    "june": "OTHERS",
    "kentucky_select": "OTHERS",
    "kode": "OTHERS",
    "la": "OTHERS",
    "lucky_strike": "OTHERS",
    "mariner": "OTHERS",
    "milano": "OTHERS",
    "oki": "OTHERS",
    "philip_morris": "OTHERS",
    "phnom_meas": "OTHERS",
    "pine": "OTHERS",
    "richman": "OTHERS",
    "richsons": "OTHERS",
    "saigon": "OTHERS",
    "teton": "OTHERS",
    "texas_5": "OTHERS",
    "thang_long": "OTHERS",
    "vibe": "OTHERS",
    "west": "OTHERS",
    "wonder": "OTHERS",
    "yellow_elephant": "OTHERS",
    "zest_yogo": "OTHERS",
    "zouk": "OTHERS",
}

# ─── Legacy name aliases ───
# Maps old/misspelled reference filenames to the correct registry name
LEGACY_ALIASES = {
    # Old misspellings from original dataset
    "malboro_red": "marlboro_red",
    "malboro_gold": "marlboro_gold",
    "malboro_other": "marlboro_menthol",
    "gold_sea": "gold_seal_menthol_compact",
    "oris_sliver": "oris_silver",
    "cow_boy_bluberry_mint": "cow_boy_blueberry_mint",
    "esse_double_change": "esse_change",
    # Old catch-all names -> keep working (map to first sub-product or generic)
    "esse_other": "esse_change_cafe",
    "fine_other": "fine_gold",
    "555_other": "555_refined_charcoal_filter",
    "ara_other": "ara_cool",
    "luxury_other": "luxury_lights",
    "gold_seal_other": "gold_seal_full_flavor",
    "marlboro_other": "marlboro_menthol",
    "iza_other": "iza_lights",
    "cow_boy_other": "cow_boy_lights",
    "coco_palm_other": "coco_palm_gold",
    "oris_other": "oris_pulse",
    "chinese_brand": "diamond_hehua",
    "other": "galaxy",
    "other_4": "galaxy",
    "cambo": "cambo_ff",
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
