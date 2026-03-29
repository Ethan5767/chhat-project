# Codebase Bugfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all identified bugs across output_format.py, pipeline.py, main.py, brand_classifier.py, train.py, process_glass_view.py, and frontend/app.py.

**Architecture:** Fixes are grouped by file and priority. Critical mapping errors in output_format.py are fixed first (they corrupt CSV output). Then pipeline robustness, API safety, frontend resilience, and training quality fixes.

**Tech Stack:** Python 3.12, FastAPI, Streamlit, PyTorch, RF-DETR, DINOv2

---

### Task 1: Fix output_format.py -- Wrong and missing product mappings

**Files:**
- Modify: `backend/output_format.py:123-168`

This is the highest priority fix. The INTERNAL_TO_product dictionary has wrong mappings and is missing 19+ products. The single source of truth is `backend/brand_registry.py` BRAND_REGISTRY.

- [ ] **Step 1: Replace the entire INTERNAL_TO_product dictionary**

Replace lines 123-168 with a complete dictionary auto-derived from brand_registry, plus legacy aliases:

```python
INTERNAL_TO_product = {
    # MEVIUS
    "mevius_original": "MEVIUS ORIGINAL",
    "mevius_sky_blue": "MEVIUS SKY BLUE",
    "mevius_option_purple": "MEVIUS OPTION PURPLE",
    "mevius_freezy_dew": "MEVIUS FREEZY DEW",
    "mevius_option_purple_super_slims": "MEVIUS OPTION PURPLE SUPER SLIMS",
    "mevius_kimavi": "MEVIUS KIWAMI",
    "mevius_e_series_blue": "MEVIUS E-SERIES BLUE",
    "mevius_mint_flow": "MEVIUS MINT FLOW",
    # WINSTON
    "winston_night_blue": "WINSTON NIGHT BLUE",
    "winston_option_purple": "WINSTON OPTION PURPLE",
    "winston_option_blue": "WINSTON OPTION BLUE",
    # ESSE
    "esse_change": "ESSE CHANGE",
    "esse_light": "ESSE LIGHTS",
    "esse_menthol": "ESSE MENTHOL",
    "esse_gold": "ESSE GOLD",
    "esse_other": "ESSE OTHERS",
    # FINE
    "fine_red_hard_pack": "FINE RED HARD PACK",
    "fine_other": "FINE OTHERS",
    # 555
    "555_sphere2_velvet": "555 SPHERE2 VELVETY",
    "555_original": "555 ORIGINAL",
    "555_gold": "555 GOLD",
    "555_other": "555 OTHERS",
    # ARA
    "ara_red": "ARA RED",
    "ara_gold": "ARA GOLD",
    "ara_menthol": "ARA MENTHOL",
    "ara_other": "ARA OTHERS",
    # LUXURY
    "luxury_full_flavour": "LUXURY FULL FLAVOUR",
    "luxury_menthol": "LUXURY MENTHOL",
    "luxury_other": "LUXURY OTHERS",
    # GOLD SEAL
    "gold_seal_menthol_compact": "GOLD SEAL MENTHOL COMPACT",
    "gold_seal_menthol_kingsize": "GOLD SEAL MENTHOL KINGSIZE",
    "gold_seal_other": "GOLD SEAL OTHERS",
    # MARLBORO
    "marlboro_red": "MARLBORO RED",
    "marlboro_gold": "MARLBORO GOLD",
    "marlboro_other": "MARLBORO OTHERS",
    # CAMBO
    "cambo_classical": "CAMBO CLASSICAL",
    "cambo_ff": "CAMBO FF",
    "cambo_menthol": "CAMBO MENTHOL",
    # IZA
    "iza_ff": "IZA FF",
    "iza_menthol": "IZA MENTHOL",
    "iza_other": "IZA OTHERS",
    # HERO
    "hero": "HERO HARD PACK",
    # COW BOY
    "cow_boy_blueberry_mint": "COW BOY BLUEBERRY MINT",
    "cow_boy_hard_pack": "COW BOY HARD PACK",
    "cow_boy_menthol": "COW BOY MENTHOL",
    "cow_boy_other": "COW BOY OTHERS",
    # COCO PALM
    "coco_palm_hard_pack": "COCO PALM HARD PACK",
    "coco_palm_menthol": "COCO PALM MENTHOL",
    "coco_palm_other": "COCO PALM OTHERS",
    # CROWN
    "crown": "CROWN",
    # LAPIN
    "lapin_ff": "LAPIN FF",
    "lapin_menthol": "LAPIN MENTHOL",
    # ORIS
    "oris_pulse_blue": "ORIS PULSE BLUE",
    "oris_ice_plus": "ORIS ICE PLUS",
    "oris_silver": "ORIS SILVER",
    "oris_other": "ORIS OTHERS",
    # JET
    "jet": "JET",
    # L&M
    "l_and_m": "L&M",
    # DJARUM
    "djarum": "DJARUM",
    # LIBERATION
    "liberation": "LIBERATION",
    # MODERN
    "modern": "MODERN",
    # MOND
    "mond": "MOND",
    # NATIONAL
    "national": "NATIONAL",
    # CHUNGHWA
    "chunghwa": "CHUNGHWA",
    # SHUANGXI
    "shuangxi": "SHUANGXI",
    # YUN YAN
    "yun_yan": "YUN YAN",
    # CHINESE BRAND
    "chinese_brand": "CHINESE BRANDS",
    # OTHERS
    "other": "OTHERS",
    # Legacy aliases (old filenames still in some references)
    "esse_double_change": "ESSE OTHERS",
    "malboro_red": "MARLBORO RED",
    "malboro_gold": "MARLBORO GOLD",
    "malboro_other": "MARLBORO OTHERS",
    "gold_sea": "GOLD SEAL MENTHOL COMPACT",
    "oris_sliver": "ORIS SILVER",
    "cow_boy_bluberry_mint": "COW BOY BLUEBERRY MINT",
    "cambo": "CAMBO FF",
    "galaxy": "OTHERS",
    "other_4": "OTHERS",
}
```

**What was wrong:**
- `luxury_other` mapped to "LUXURY FULL FLAVOUR" instead of "LUXURY OTHERS"
- `iza_other` mapped to "IZA FF" instead of "IZA OTHERS"
- `cow_boy_other` mapped to "COW BOY HARD PACK" instead of "COW BOY OTHERS"
- `cow_boy_bluberry_mint` typo preserved as legacy alias
- 19+ products completely missing (all CAMBO, COCO PALM, CROWN, LAPIN, etc.)

- [ ] **Step 2: Verify the fix**

Run: `cd backend && python -c "from output_format import INTERNAL_TO_product; from brand_registry import BRAND_REGISTRY; missing = [i for b,ps in BRAND_REGISTRY.items() for _,i in ps if i not in INTERNAL_TO_product]; print('Missing:', missing if missing else 'NONE')"`

Expected: `Missing: NONE`

---

### Task 2: Fix pipeline.py -- OCR parsing and classifier robustness

**Files:**
- Modify: `backend/pipeline.py:325,449,564-568`

- [ ] **Step 1: Add case-insensitive glob in build_index (line 325)**

Replace:
```python
has_images = any(type_dir.glob("*.jpg")) or any(type_dir.glob("*.png"))
```

With:
```python
has_images = any(type_dir.glob("*.[jJ][pP][gG]")) or any(type_dir.glob("*.[jJ][pP][eE][gG]")) or any(type_dir.glob("*.[pP][nN][gG]"))
```

- [ ] **Step 2: Add KeyError guard in classify_embeddings (line 449)**

Replace:
```python
label = idx_to_label[str(idx.item())]
```

With:
```python
label_key = str(idx.item())
label = idx_to_label.get(label_key, f"unknown_{label_key}")
```

- [ ] **Step 3: Add error guard around OCR item parsing (lines 564-568)**

Replace:
```python
    for item in ocr_items:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        text_conf = float(item[2])
```

With:
```python
    for item in ocr_items:
        if len(item) < 3:
            continue
        try:
            text = str(item[1]).strip()
            text_conf = float(item[2])
        except (IndexError, TypeError, ValueError):
            continue
```

---

### Task 3: Fix main.py -- Race conditions and input validation

**Files:**
- Modify: `backend/main.py:119-124,139-144,1471,1480`

- [ ] **Step 1: Fix race condition in update_progress (lines 139-144)**

Replace:
```python
def update_progress(job_id: str, current: int, total: int, message: str):
    pct = int((current / total) * 100) if total > 0 else 0
    with jobs_lock:
        job = jobs.get(job_id)
    if job:
        job["queue"].put((pct, message))
```

With:
```python
def update_progress(job_id: str, current: int, total: int, message: str):
    pct = int((current / total) * 100) if total > 0 else 0
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job["queue"].put((pct, message))
```

Move the `job["queue"].put()` call INSIDE the lock so the job cannot be deleted between the get and the put.

- [ ] **Step 2: Add file locking to batch_history updates (lines 119-124)**

Replace:
```python
def _update_batch_history(job_id: str, updates: dict) -> None:
    rows = _load_batch_history()
    for row in rows:
        if row.get("job_id") == job_id:
            row.update(updates)
    _BATCH_HISTORY_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
```

With:
```python
_batch_history_lock = threading.Lock()

def _update_batch_history(job_id: str, updates: dict) -> None:
    with _batch_history_lock:
        rows = _load_batch_history()
        for row in rows:
            if row.get("job_id") == job_id:
                row.update(updates)
        _BATCH_HISTORY_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
```

Also wrap `_append_batch_history` with the same lock.

- [ ] **Step 3: Add file size limit and path sanitization to /upload-coco (lines 1471, 1480)**

After `data = await coco_file.read()` at line 1471, add:
```python
    MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1 GB
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 1 GB)")
```

At line 1480, replace:
```python
        save_path = DATASETS_DIR / filename
```

With:
```python
        save_path = DATASETS_DIR / Path(filename).name
```

Apply the same `Path(filename).name` sanitization to ALL places where `filename` is used as a path component in this endpoint.

---

### Task 4: Fix brand_classifier.py -- Skip bad images instead of placeholder

**Files:**
- Modify: `brand_classifier.py:60-84`

- [ ] **Step 1: Skip bad images instead of using blank placeholder (lines 68-75)**

Replace:
```python
        for p in batch_paths:
            try:
                img = pad_to_square(Image.open(p).convert("RGB"))
                imgs.append(img)
            except Exception as exc:
                print(f"  Warning: could not open {p}: {exc}")
                # Use a blank image as placeholder
                imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
```

With:
```python
        valid_paths = []
        for p in batch_paths:
            try:
                img = pad_to_square(Image.open(p).convert("RGB"))
                imgs.append(img)
                valid_paths.append(p)
            except Exception as exc:
                print(f"  Warning: skipping corrupt image {p}: {exc}")
```

Then update the corresponding label/path tracking that follows this loop to use `valid_paths` instead of `batch_paths`, so labels stay aligned with actual images.

NOTE: This requires checking how `image_paths` and `labels` are used after embedding. The labels list must exclude skipped images.

---

### Task 5: Fix process_glass_view.py -- Error categorization and resource leak

**Files:**
- Modify: `process_glass_view.py:102-108,171-175`

- [ ] **Step 1: Fix resource leak in centroid computation (lines 102-108)**

Replace:
```python
            try:
                img = Image.open(p).convert("RGB")
                vec = embed_single(img, processor, model, device)
                vecs.append(vec)
                img.close()
            except Exception:
                continue
```

With:
```python
            try:
                with Image.open(p) as img:
                    img_rgb = img.convert("RGB")
                    vec = embed_single(img_rgb, processor, model, device)
                    vecs.append(vec)
            except Exception:
                continue
```

- [ ] **Step 2: Fix wrong error category for file open failures (lines 171-175)**

Replace:
```python
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped_no_det += 1
            continue
```

With:
```python
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped_error += 1
            continue
```

Then add `skipped_error = 0` to the counter initialization block around line 158, and add it to the summary print at the end.

---

### Task 6: Fix frontend/app.py -- Error handling and cache clearing

**Files:**
- Modify: `frontend/app.py:1,29-32,520-528,1352-1353`

- [ ] **Step 1: Remove unused import (line 1)**

Remove:
```python
import base64
```

- [ ] **Step 2: Add error handling to stream_progress (lines 29-32)**

Replace:
```python
def stream_progress(job_id: str) -> Generator[tuple[int, str], None, None]:
    url = f"{BACKEND_URL}/progress/{job_id}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
```

With:
```python
def stream_progress(job_id: str) -> Generator[tuple[int, str], None, None]:
    url = f"{BACKEND_URL}/progress/{job_id}"
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        yield 0, f"Connection error: {exc}"
        return
    with response:
```

- [ ] **Step 3: Fix file pointer not reset on CSV parse failure (lines 520-528)**

Replace:
```python
    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
            with st.expander(f"Preview ({len(preview_df)} rows)", expanded=True):
                st.dataframe(preview_df.head(10), width="stretch")
            st.caption(f"Batch mode: **{BATCH_MODE_LABEL}**")
            uploaded.seek(0)
        except Exception as exc:
            st.error(f"Could not preview CSV: {exc}")
```

With:
```python
    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
            with st.expander(f"Preview ({len(preview_df)} rows)", expanded=True):
                st.dataframe(preview_df.head(10), width="stretch")
            st.caption(f"Batch mode: **{BATCH_MODE_LABEL}**")
        except Exception as exc:
            st.error(f"Could not preview CSV: {exc}")
        finally:
            uploaded.seek(0)
```

- [ ] **Step 4: Add missing cache clear after adding reference (lines 1352-1353)**

Replace:
```python
                                _fetch_reference_listing.clear()
                                _fetch_brand_hierarchy.clear()
```

With:
```python
                                _fetch_reference_listing.clear()
                                _fetch_brand_hierarchy.clear()
                                _fetch_reference_image_bytes.clear()
```

---

### Task 7: Fix brand_registry.py -- Self-referential alias

**Files:**
- Modify: `backend/brand_registry.py:159`

- [ ] **Step 1: Remove the no-op alias (line 159)**

Replace:
```python
    "mevius_kimavi": "mevius_kimavi",  # KIWAMI in Excel, KIMAVI in references - same product
```

With a comment only (no mapping needed since it's the canonical name):
```python
    # mevius_kimavi: canonical name (KIWAMI in Excel, KIMAVI in reference filenames)
```
