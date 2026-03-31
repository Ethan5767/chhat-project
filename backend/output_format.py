"""Q12A/Q12B output format mapping for CHHAT cigarette detection.

Maps detected brand/product names from the pipeline to the exact column structure
required by the client Excel format.
"""

# Q12A: Brand detection columns
# Maps Q12A column code -> (English brand name, Khmer name)
Q12A_BRANDS = {
    "Q12A_1":  ("MEVIUS", "ម៉ៃសេវែន / មេវៀស"),
    "Q12A_2":  ("WINSTON", "វីនស្តុន"),
    "Q12A_3":  ("ESSE", "អេសសេ"),
    "Q12A_4":  ("FINE", "ហ្វីន"),
    "Q12A_5":  ("555", "បារី\u200b\u200b 555"),
    "Q12A_6":  ("ARA", "សេក"),
    "Q12A_7":  ("LUXURY", "លុចសារី"),
    "Q12A_8":  ("GOLD SEAL", "ហ្គោលសៀល"),
    "Q12A_9":  ("MARLBORO", "ម៉ាបូរ៉ូ"),
    "Q12A_10": ("CAMBO", "ខេមបូ"),
    "Q12A_11": ("IZA", "អ៊ីសា"),
    "Q12A_12": ("HERO", "ហេរ៉ូ"),
    "Q12A_13": ("COW BOY", "ខោប៊យ"),
    "Q12A_14": ("COCO PALM", "ដើមដូង"),
    "Q12A_15": ("CROWN", "ក្រោន"),
    "Q12A_16": ("LAPIN", "ឡាពីន / ទន្សាយ"),
    "Q12A_17": ("ORIS", "អូរីស"),
    "Q12A_18": ("JET", "ជែត"),
    "Q12A_19": ("L&M", "អិល អ៊ែន អឹម"),
    "Q12A_20": ("DJARUM", "ចារ៉ាម"),
    "Q12A_21": ("LIBERATION", "រំដោះ"),
    "Q12A_22": ("MODERN", "ម៉ូឌឺន"),
    "Q12A_23": ("MOND", "ម៉ន់"),
    "Q12A_24": ("NATIONAL", "ជាតិ"),
    "Q12A_25": ("CHUNGHWA", "ឆុងវ៉ា"),
    "Q12A_26": ("SHUANGXI", "ស៊ុងស៊ីង"),
    "Q12A_27": ("YUN YAN", "យ័នយែន"),
    "Q12A_28": ("CHINESE BRAND", "ម៉ាកចិន"),
    "Q12A_29": ("OTHERS", "ម៉ាកផ្សេងៗ"),
}

# Reverse lookup: English brand name -> Q12A code
BRAND_TO_Q12A = {v[0]: k for k, v in Q12A_BRANDS.items()}

# Q12B: product detection columns
# Maps Q12B column code -> product display name
Q12B_productS = {
    "Q12B_L[{_1}]._scale":  "MEVIUS ORIGINAL",
    "Q12B_L[{_2}]._scale":  "MEVIUS SKY BLUE",
    "Q12B_L[{_3}]._scale":  "MEVIUS OPTION PURPLE",
    "Q12B_L[{_4}]._scale":  "MEVIUS FREEZY DEW",
    "Q12B_L[{_5}]._scale":  "MEVIUS OPTION PURPLE SUPER SLIMS",
    "Q12B_L[{_6}]._scale":  "MEVIUS KIWAMI",
    "Q12B_L[{_7a}]._scale": "MEVIUS E-SERIES BLUE",
    "Q12B_L[{_7b}]._scale": "MEVIUS MINT FLOW",
    "Q12B_L[{_8}]._scale":  "WINSTON NIGHT BLUE",
    "Q12B_L[{_9}]._scale":  "WINSTON OPTION PURPLE",
    "Q12B_L[{_10}]._scale": "WINSTON OPTION BLUE",
    "Q12B_L[{_11}]._scale": "ESSE CHANGE",
    "Q12B_L[{_12}]._scale": "ESSE LIGHTS",
    "Q12B_L[{_13}]._scale": "ESSE MENTHOL",
    "Q12B_L[{_14}]._scale": "ESSE GOLD",
    "Q12B_L[{_15}]._scale": "ESSE OTHERS",
    "Q12B_L[{_16}]._scale": "FINE RED HARD PACK",
    "Q12B_L[{_17}]._scale": "FINE OTHERS",
    "Q12B_L[{_18}]._scale": "555 SPHERE2 VELVETY",
    "Q12B_L[{_19}]._scale": "555 ORIGINAL",
    "Q12B_L[{_20}]._scale": "555 GOLD",
    "Q12B_L[{_21}]._scale": "555 OTHERS",
    "Q12B_L[{_22}]._scale": "ARA RED",
    "Q12B_L[{_23}]._scale": "ARA GOLD",
    "Q12B_L[{_24}]._scale": "ARA MENTHOL",
    "Q12B_L[{_25}]._scale": "ARA OTHERS",
    "Q12B_L[{_26}]._scale": "LUXURY FULL FLAVOUR",
    "Q12B_L[{_27}]._scale": "LUXURY MENTHOL",
    "Q12B_L[{_28}]._scale": "LUXURY OTHERS",
    "Q12B_L[{_29}]._scale": "GOLD SEAL MENTHOL COMPACT",
    "Q12B_L[{_30}]._scale": "GOLD SEAL MENTHOL KINGSIZE",
    "Q12B_L[{_31}]._scale": "GOLD SEAL OTHERS",
    "Q12B_L[{_32}]._scale": "MARLBORO RED",
    "Q12B_L[{_33}]._scale": "MARLBORO GOLD",
    "Q12B_L[{_34}]._scale": "MARLBORO OTHERS",
    "Q12B_L[{_35}]._scale": "CAMBO CLASSICAL",
    "Q12B_L[{_36}]._scale": "CAMBO FF",
    "Q12B_L[{_37}]._scale": "CAMBO MENTHOL",
    "Q12B_L[{_38}]._scale": "IZA FF",
    "Q12B_L[{_39}]._scale": "IZA MENTHOL",
    "Q12B_L[{_40}]._scale": "IZA OTHERS",
    "Q12B_L[{_41}]._scale": "HERO HARD PACK",
    "Q12B_L[{_42}]._scale": "COW BOY BLUEBERRY MINT",
    "Q12B_L[{_43}]._scale": "COW BOY HARD PACK",
    "Q12B_L[{_44}]._scale": "COW BOY MENTHOL",
    "Q12B_L[{_45}]._scale": "COW BOY OTHERS",
    "Q12B_L[{_46}]._scale": "COCO PALM HARD PACK",
    "Q12B_L[{_47}]._scale": "COCO PALM MENTHOL",
    "Q12B_L[{_48}]._scale": "COCO PALM OTHERS",
    "Q12B_L[{_49}]._scale": "CROWN",
    "Q12B_L[{_50}]._scale": "LAPIN FF",
    "Q12B_L[{_51}]._scale": "LAPIN MENTHOL",
    "Q12B_L[{_52}]._scale": "ORIS PULSE BLUE",
    "Q12B_L[{_53}]._scale": "ORIS ICE PLUS",
    "Q12B_L[{_54}]._scale": "ORIS SILVER",
    "Q12B_L[{_55}]._scale": "ORIS OTHERS",
    "Q12B_L[{_56}]._scale": "JET",
    "Q12B_L[{_57}]._scale": "L&M",
    "Q12B_L[{_60}]._scale": "DJARUM",
    "Q12B_L[{_61}]._scale": "LIBERATION",
    "Q12B_L[{_62}]._scale": "MODERN",
    "Q12B_L[{_63}]._scale": "MOND",
    "Q12B_L[{_64}]._scale": "NATIONAL",
    "Q12B_L[{_59}]._scale": "CHUNGHWA",
    "Q12B_L[{_65}]._scale": "SHUANGXI",
    "Q12B_L[{_66}]._scale": "YUN YAN",
    "Q12B_L[{_58}]._scale": "CHINESE BRANDS",
    "Q12B_L[{_67}]._scale": "OTHERS",
}

# Reverse lookup: product name -> Q12B code
product_TO_Q12B = {v: k for k, v in Q12B_productS.items()}

# Map internal pipeline names to display product names
# Pipeline uses: mevius_original, esse_change, 555_sphere2_velvet, etc.
# Display uses: MEVIUS ORIGINAL, ESSE CHANGE, 555 SPHERE2 VELVETY, etc.
INTERNAL_TO_product = {
    # MEVIUS
    "mevius_original":                  "MEVIUS ORIGINAL",
    "mevius_sky_blue":                  "MEVIUS SKY BLUE",
    "mevius_option_purple":             "MEVIUS OPTION PURPLE",
    "mevius_freezy_dew":                "MEVIUS FREEZY DEW",
    "mevius_option_purple_super_slims": "MEVIUS OPTION PURPLE SUPER SLIMS",
    "mevius_kimavi":                    "MEVIUS KIWAMI",
    "mevius_e_series_blue":             "MEVIUS E-SERIES BLUE",
    "mevius_mint_flow":                 "MEVIUS MINT FLOW",
    # WINSTON
    "winston_night_blue":               "WINSTON NIGHT BLUE",
    "winston_option_purple":            "WINSTON OPTION PURPLE",
    "winston_option_blue":              "WINSTON OPTION BLUE",
    # ESSE
    "esse_change":                      "ESSE CHANGE",
    "esse_light":                       "ESSE LIGHTS",
    "esse_menthol":                     "ESSE MENTHOL",
    "esse_gold":                        "ESSE GOLD",
    "esse_other":                       "ESSE OTHERS",
    # FINE
    "fine_red_hard_pack":               "FINE RED HARD PACK",
    "fine_other":                       "FINE OTHERS",
    # 555
    "555_sphere2_velvet":               "555 SPHERE2 VELVETY",
    "555_original":                     "555 ORIGINAL",
    "555_gold":                         "555 GOLD",
    "555_other":                        "555 OTHERS",
    # ARA
    "ara_red":                          "ARA RED",
    "ara_gold":                         "ARA GOLD",
    "ara_menthol":                      "ARA MENTHOL",
    "ara_other":                        "ARA OTHERS",
    # LUXURY
    "luxury_full_flavour":              "LUXURY FULL FLAVOUR",
    "luxury_menthol":                   "LUXURY MENTHOL",
    "luxury_other":                     "LUXURY OTHERS",
    # GOLD SEAL
    "gold_seal_menthol_compact":        "GOLD SEAL MENTHOL COMPACT",
    "gold_seal_menthol_kingsize":       "GOLD SEAL MENTHOL KINGSIZE",
    "gold_seal_other":                  "GOLD SEAL OTHERS",
    # MARLBORO
    "marlboro_red":                     "MARLBORO RED",
    "marlboro_gold":                    "MARLBORO GOLD",
    "marlboro_other":                   "MARLBORO OTHERS",
    # CAMBO
    "cambo_classical":                  "CAMBO CLASSICAL",
    "cambo_ff":                         "CAMBO FF",
    "cambo_menthol":                    "CAMBO MENTHOL",
    # IZA
    "iza_ff":                           "IZA FF",
    "iza_menthol":                      "IZA MENTHOL",
    "iza_other":                        "IZA OTHERS",
    # HERO
    "hero":                             "HERO HARD PACK",
    # COW BOY
    "cow_boy_blueberry_mint":           "COW BOY BLUEBERRY MINT",
    "cow_boy_hard_pack":                "COW BOY HARD PACK",
    "cow_boy_menthol":                  "COW BOY MENTHOL",
    "cow_boy_other":                    "COW BOY OTHERS",
    # COCO PALM
    "coco_palm_hard_pack":              "COCO PALM HARD PACK",
    "coco_palm_menthol":                "COCO PALM MENTHOL",
    "coco_palm_other":                  "COCO PALM OTHERS",
    # CROWN
    "crown":                            "CROWN",
    # LAPIN
    "lapin_ff":                         "LAPIN FF",
    "lapin_menthol":                    "LAPIN MENTHOL",
    # ORIS
    "oris_pulse_blue":                  "ORIS PULSE BLUE",
    "oris_ice_plus":                    "ORIS ICE PLUS",
    "oris_silver":                      "ORIS SILVER",
    "oris_other":                       "ORIS OTHERS",
    # JET
    "jet":                              "JET",
    # L&M
    "l_and_m":                          "L&M",
    # DJARUM
    "djarum":                           "DJARUM",
    # LIBERATION
    "liberation":                       "LIBERATION",
    # MODERN
    "modern":                           "MODERN",
    # MOND
    "mond":                             "MOND",
    # NATIONAL
    "national":                         "NATIONAL",
    # CHUNGHWA
    "chunghwa":                         "CHUNGHWA",
    # SHUANGXI
    "shuangxi":                         "SHUANGXI",
    # YUN YAN
    "yun_yan":                          "YUN YAN",
    # CHINESE BRAND
    "chinese_brand":                    "CHINESE BRANDS",
    # OTHERS
    "other":                            "OTHERS",
    # Legacy aliases (old/misspelled names — kept for backward compatibility)
    "malboro_red":                      "MARLBORO RED",
    "malboro_gold":                     "MARLBORO GOLD",
    "malboro_other":                    "MARLBORO OTHERS",
    "gold_sea":                         "GOLD SEAL MENTHOL COMPACT",
    "oris_sliver":                      "ORIS SILVER",
    "cow_boy_bluberry_mint":            "COW BOY BLUEBERRY MINT",
    "esse_double_change":               "ESSE OTHERS",
    "other_4":                          "OTHERS",
    "galaxy":                           "OTHERS",
    "cambo":                            "CAMBO FF",
}

# Map product name to parent brand
product_TO_BRAND = {}
for product_name in Q12B_productS.values():
    for brand_code, (brand_eng, _) in Q12A_BRANDS.items():
        if product_name.startswith(brand_eng):
            product_TO_BRAND[product_name] = brand_eng
            break
    else:
        # Single-word brands that are also product names
        if product_name in [b[0] for b in Q12A_BRANDS.values()]:
            product_TO_BRAND[product_name] = product_name
        elif product_name == "CHINESE BRANDS":
            product_TO_BRAND[product_name] = "CHINESE BRAND"
        elif product_name == "OTHERS":
            product_TO_BRAND[product_name] = "OTHERS"


def internal_to_display(internal_name: str) -> str:
    """Convert pipeline internal name to display product name."""
    return INTERNAL_TO_product.get(internal_name, internal_name.upper().replace("_", " "))


def build_q12_row(detected_products: list[str]) -> dict:
    """Build Q12A and Q12B columns from a list of detected product names.

    Args:
        detected_products: list of internal pipeline names like
            ['mevius_original', 'esse_change', '555_sphere2_velvet']

    Returns:
        dict with all Q12A and Q12B column values.
    """
    # Convert internal names to display product names
    display_products = []
    for p in detected_products:
        product = internal_to_display(p)
        if product and product not in display_products:
            display_products.append(product)

    # Find parent brands for each product
    detected_brands = []
    for product in display_products:
        brand = product_TO_BRAND.get(product, "")
        if brand and brand not in detected_brands:
            detected_brands.append(brand)

    row = {}

    # Q12A: combined brand column (pipe-separated with Khmer)
    q12a_parts = []
    for brand in detected_brands:
        q12a_code = BRAND_TO_Q12A.get(brand, "")
        if q12a_code:
            _, khmer = Q12A_BRANDS[q12a_code]
            q12a_parts.append(f"{brand}_{khmer}")
    row["Q12A"] = "|".join(q12a_parts) if q12a_parts else ""

    # Q12A individual columns
    for q12a_code, (brand_eng, khmer) in Q12A_BRANDS.items():
        if brand_eng in detected_brands:
            row[q12a_code] = f"{brand_eng}_{khmer}"
        else:
            row[q12a_code] = ""

    # Q12B: combined product column (pipe-separated)
    row["Q12B"] = "|".join(display_products) if display_products else ""

    # Q12B individual columns
    for q12b_code, product_name in Q12B_productS.items():
        if product_name in display_products:
            row[q12b_code] = product_name
        else:
            row[q12b_code] = ""

    return row


def get_output_columns() -> list[str]:
    """Return the ordered list of output column headers matching the Excel format."""
    cols = ["Respondent.Serial", "Q6", "Q12A"]
    # Q12A individual brand columns
    for i in range(1, 30):
        cols.append(f"Q12A_{i}")
    # Q12B combined
    cols.append("Q12B")
    # Q12B individual product columns (in the exact order from the Excel)
    q12b_order = [
        "_1", "_2", "_3", "_4", "_5", "_6", "_7a", "_7b",
        "_8", "_9", "_10", "_11", "_12", "_13", "_14", "_15",
        "_16", "_17", "_18", "_19", "_20", "_21", "_22", "_23",
        "_24", "_25", "_26", "_27", "_28", "_29",
        "_31", "_32", "_33", "_34", "_35", "_36", "_37",
        "_38", "_39", "_40", "_41", "_42", "_43", "_44", "_45",
        "_46", "_47", "_48", "_49", "_50", "_51", "_52", "_53",
        "_54", "_55", "_56", "_57",
        "_60", "_61", "_62", "_63", "_64", "_59", "_65", "_66", "_58", "_67",
    ]
    for code in q12b_order:
        cols.append(f"Q12B_L[{{{{code}}}}]._scale".replace("{{{{code}}}}", f"{{{code}}}"))

    # Photo link columns
    cols.extend(["Q30_1", "Q30_2", "Q30_3", "Q33_1", "Q33_2", "Q33_3"])
    return cols
