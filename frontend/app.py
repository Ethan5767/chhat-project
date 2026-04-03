import json
import os
import time
from io import BytesIO
from typing import Generator

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
BATCH_MODE_LABEL = "ALL ROWS"

# Match wide layout (~.block-container max-width 1360px minus padding). The old 650px estimate
# was far too small, so the iframe clipped the bottom of landscape images.
_SINGLE_IFRAME_REF_WIDTH = 1280


def _single_detection_iframe_height(img_w: int, img_h: int) -> int:
    """Height for st.components.html so img { width:100%; height:auto } fits without clipping."""
    iw = max(1, int(img_w))
    ih = max(1, int(img_h))
    scaled = int(_SINGLE_IFRAME_REF_WIDTH * ih / iw)
    return max(400, scaled) + 160


def stream_progress(job_id: str) -> Generator[tuple[int, str], None, None]:
    url = f"{BACKEND_URL}/progress/{job_id}"
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        yield 0, f"Connection error: {exc}"
        return
    with response:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.startswith("DONE|"):
                yield 100, "Done"
                return
            if payload.startswith("ERROR|"):
                message = payload.split("|", 1)[1] if "|" in payload else "Unknown error"
                raise RuntimeError(message)
            pct_str, message = payload.split("|", 1) if "|" in payload else ("0", payload)
            try:
                pct = int(float(pct_str))
            except ValueError:
                pct = 0
            yield max(0, min(100, pct)), message


def render_working_animation(target, text: str = "Model is working...", pct: int = 0):
    """Render an animated detective character working on the project."""
    # Pick character frame based on progress
    if pct < 20:
        char_frame = "detective-look"
        detail = "Scanning shelves..."
    elif pct < 40:
        char_frame = "detective-detect"
        detail = "Detecting cigarette packs..."
    elif pct < 60:
        char_frame = "detective-match"
        detail = "Matching brands..."
    elif pct < 80:
        char_frame = "detective-read"
        detail = "Reading pack labels..."
    else:
        char_frame = "detective-write"
        detail = "Compiling results..."

    target.markdown(
        f"""
        <div class="worker-card">
            <div class="worker-scene">
                <div class="detective" id="{char_frame}">
                    <div class="detective-hat"></div>
                    <div class="detective-head">
                        <div class="detective-eye left-eye"></div>
                        <div class="detective-eye right-eye"></div>
                    </div>
                    <div class="detective-body"></div>
                    <div class="detective-magnifier">
                        <div class="magnifier-glass"></div>
                        <div class="magnifier-handle"></div>
                    </div>
                </div>
                <div class="smoke-particles">
                    <div class="particle p1"></div>
                    <div class="particle p2"></div>
                    <div class="particle p3"></div>
                </div>
            </div>
            <div class="worker-text">
                <div class="worker-title">{text}</div>
                <div class="worker-detail">{detail}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _final_brand_row_has_detection(val) -> bool:
    s = str(val).strip()
    if s in ("ERROR", "NOT_PROCESSED"):
        return False
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if not arr:
                return False
            if arr[0] in ("ERROR", "NOT_PROCESSED"):
                return False
            return arr[0] != "NO_DETECTION"
        except json.JSONDecodeError:
            return False
    return s not in ("NO_DETECTION", "ERROR", "NOT_PROCESSED")


def _final_brand_row_is_error(val) -> bool:
    s = str(val).strip()
    if s == "ERROR":
        return True
    if s.startswith("["):
        try:
            arr = json.loads(s)
            return bool(arr) and arr[0] == "ERROR"
        except json.JSONDecodeError:
            return False
    return False


def _show_result_preview(result_df: pd.DataFrame):
    if "final_brand" in result_df.columns:
        base_cols = ["final_brand", "overall_confidence"]
        extra_cols = [
            c for c in result_df.columns
            if (c.endswith("_detected_brands") or c.endswith("_confidence"))
            and c not in base_cols
        ]
        display_cols = [c for c in base_cols + extra_cols if c in result_df.columns]
        st.dataframe(result_df[display_cols].head(20), width="stretch")

        rows_total = len(result_df)
        rows_detected = int(result_df["final_brand"].map(_final_brand_row_has_detection).sum())
        rows_errors = int(result_df["final_brand"].map(_final_brand_row_is_error).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows processed", rows_total)
        m2.metric("Brands detected", rows_detected)
        m3.metric("Errors", rows_errors)
    else:
        st.dataframe(result_df.head(20), width="stretch")


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_reference_image_bytes(packaging_type: str, filename: str):
    """Fetch reference image from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/reference-image/{packaging_type}/{filename}", timeout=10)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_brand_hierarchy():
    """Fetch brand registry once and cache."""
    try:
        resp = requests.get(f"{BACKEND_URL}/brand-registry", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("brands", {}), data
    except Exception:
        return {}, {}


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_reference_listing(internal_name: str, packaging_type: str = "pack"):
    """Fetch reference image listing for a product, cached."""
    try:
        resp = requests.get(f"{BACKEND_URL}/reference-images/{internal_name}?packaging_type={packaging_type}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_brand_book_sample(internal_name: str):
    """Fetch brand book sample image, cached for 1 hour."""
    try:
        resp = requests.get(f"{BACKEND_URL}/brand-book-sample/{internal_name}", timeout=5)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_index_status():
    """Fetch index status, cached 30s."""
    try:
        resp = requests.get(f"{BACKEND_URL}/index-status", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"exists": False, "brand_count": 0, "brands": []}


# -- Page config --
st.set_page_config(page_title="RF-DETR Brand Detector", layout="wide", page_icon="[D]")

# -- Styles --
st.markdown(
    """
    <style>
        /* -- Base theme -- */
        .stApp {
            background: radial-gradient(ellipse at 20% 0%, #1a1f35 0%, #0e1120 50%, #090b12 100%);
            color: #e4e8f2;
        }
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 1rem !important;
            max-width: 1360px;
        }

        /* -- Panel cards -- */
        .panel {
            border: 1px solid rgba(99, 120, 255, 0.15);
            border-radius: 14px;
            background: linear-gradient(170deg, rgba(22,28,48,0.97) 0%, rgba(13,17,30,0.97) 100%);
            padding: 20px 22px;
            margin-bottom: 14px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
        }

        /* -- Header -- */
        .app-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(99, 120, 255, 0.1);
        }
        .app-logo {
            font-size: 36px;
            line-height: 1;
        }
        .app-title {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.3px;
            background: linear-gradient(135deg, #e8ecff, #8b9aff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .app-subtitle {
            color: #7a839e;
            font-size: 13px;
            margin-top: 2px;
        }

        /* -- KPI numbers -- */
        .kpi {
            font-size: 42px;
            font-weight: 800;
            line-height: 1;
            background: linear-gradient(135deg, #f0f3ff, #7b8cff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .muted {
            color: #7a839e;
            font-size: 13px;
            margin-top: 4px;
        }

        /* -- Status badges -- */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .status-ready {
            background: rgba(52, 211, 153, 0.12);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.2);
        }
        .status-missing {
            background: rgba(251, 191, 36, 0.12);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.2);
        }

        /* -- Detective animation -- */
        .worker-card {
            display: flex;
            align-items: center;
            gap: 16px;
            border: 1px solid rgba(99, 120, 255, 0.2);
            border-radius: 12px;
            padding: 14px 18px;
            background: linear-gradient(135deg, rgba(22,28,48,0.9), rgba(15,20,38,0.9));
            margin: 8px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        }
        .worker-scene {
            position: relative;
            width: 56px;
            height: 56px;
            flex-shrink: 0;
        }
        .worker-text {
            flex: 1;
        }
        .worker-title {
            color: #c8d0f0;
            font-size: 14px;
            font-weight: 600;
        }
        .worker-detail {
            color: #6b7394;
            font-size: 12px;
            margin-top: 2px;
        }

        /* -- Detective character (CSS art) -- */
        .detective {
            position: relative;
            width: 40px;
            height: 48px;
            margin: 4px auto;
            animation: detectiveBob 1.8s ease-in-out infinite;
        }
        .detective-hat {
            position: absolute;
            top: 0; left: 4px;
            width: 32px; height: 12px;
            background: #4a3f6b;
            border-radius: 8px 8px 2px 2px;
            box-shadow: 0 2px 0 #3d3460;
        }
        .detective-hat::before {
            content: '';
            position: absolute;
            bottom: -2px; left: -2px;
            width: 36px; height: 4px;
            background: #5c4f82;
            border-radius: 2px;
        }
        .detective-head {
            position: absolute;
            top: 14px; left: 8px;
            width: 24px; height: 18px;
            background: #ffd5a3;
            border-radius: 12px 12px 8px 8px;
        }
        .detective-eye {
            position: absolute;
            width: 4px; height: 5px;
            background: #2a2040;
            border-radius: 50%;
            top: 6px;
            animation: eyeBlink 3s ease-in-out infinite;
        }
        .left-eye { left: 5px; }
        .right-eye { right: 5px; }
        .detective-body {
            position: absolute;
            top: 30px; left: 6px;
            width: 28px; height: 18px;
            background: #6366f1;
            border-radius: 4px 4px 8px 8px;
        }
        .detective-body::before {
            content: '';
            position: absolute;
            top: 2px; left: 50%;
            transform: translateX(-50%);
            width: 2px; height: 14px;
            background: rgba(255,255,255,0.15);
        }
        .detective-magnifier {
            position: absolute;
            top: 18px; right: -8px;
            animation: magnifierMove 2.2s ease-in-out infinite;
        }
        .magnifier-glass {
            width: 14px; height: 14px;
            border: 2.5px solid #fbbf24;
            border-radius: 50%;
            background: rgba(251,191,36,0.1);
        }
        .magnifier-handle {
            position: absolute;
            bottom: -6px; right: -2px;
            width: 3px; height: 8px;
            background: #a78b5a;
            border-radius: 0 0 2px 2px;
            transform: rotate(-35deg);
        }

        /* -- Smoke/search particles -- */
        .smoke-particles {
            position: absolute;
            top: 8px; right: 0;
        }
        .particle {
            position: absolute;
            width: 4px; height: 4px;
            background: rgba(139, 154, 255, 0.4);
            border-radius: 50%;
            animation: float 2.5s ease-out infinite;
        }
        .p1 { left: 0; animation-delay: 0s; }
        .p2 { left: 8px; animation-delay: 0.8s; }
        .p3 { left: 4px; animation-delay: 1.6s; }

        @keyframes detectiveBob {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-3px); }
        }
        @keyframes eyeBlink {
            0%, 42%, 48%, 100% { transform: scaleY(1); }
            45% { transform: scaleY(0.1); }
        }
        @keyframes magnifierMove {
            0%, 100% { transform: translateX(0) rotate(0deg); }
            50% { transform: translateX(4px) rotate(10deg); }
        }
        @keyframes float {
            0% { opacity: 0.7; transform: translateY(0) scale(1); }
            100% { opacity: 0; transform: translateY(-18px) scale(0.3); }
        }

        /* -- Pipeline badge -- */
        .pipeline-info {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-top: 8px;
        }
        .pipe-tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            background: rgba(99,102,241,0.12);
            color: #8b9aff;
            border: 1px solid rgba(99,102,241,0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Header --
st.markdown(
    """
    <div class="app-header">
        <div class="app-logo">[D]</div>
        <div>
            <div class="app-title">RF-DETR Cigarette Brand Detector</div>
            <div class="app-subtitle">Automated shelf survey analysis with RF-DETR detection and visual + text recognition</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -- Pipeline tags --
st.markdown(
    """
    <div class="pipeline-info">
        <span class="pipe-tag">RF-DETR</span>
        <span class="pipe-tag">DINOv2-base</span>
        <span class="pipe-tag">Brand Classifier</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# -- Health check --
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5)
    health.raise_for_status()
except Exception as health_exc:
    st.error(
        "Cannot reach the FastAPI backend at "
        f"`{BACKEND_URL}` ({health_exc!s}).\n\n"
        "**Local dev** (from repo root): "
        "`uvicorn backend.main:app --host 127.0.0.1 --port 8000`\n\n"
        "**Production server**: `sudo systemctl restart chhat-backend` "
        "(ensure `BACKEND_URL` in `.env` is `http://127.0.0.1:8000` for same-host Streamlit)."
    )
    st.stop()

# -- Index status (sidebar-style) --
idx_data = _fetch_index_status()
index_exists = bool(idx_data.get("exists", False))

# -- Tabs --
tab_batch, tab_single, tab_index, tab_label, tab_train = st.tabs([
    "Batch CSV Detection", "Single Image Test", "Reference Index",
    "Label Training Crops", "Training",
])

# ════════════════════════════════════════════════════════
# TAB 1: Batch CSV Detection
# ════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Batch Brand Detection")
    st.caption("Upload a CSV with image URL columns from field surveys.")

    if not index_exists:
        st.warning("No brand classifier found. Run `python brand_classifier.py` first or use the **Reference Index** tab to train one.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", key="csv_upload")

    preview_df = None
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

    run_disabled = uploaded is None or not index_exists
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        run_local = st.button("Run Detection (Local)", disabled=run_disabled, type="primary", key="run_batch")
    with run_col2:
        run_gpu = st.button("Run Detection (RunPod GPU)", disabled=uploaded is None, type="secondary", key="run_batch_gpu")
        st.caption("Spins up A100 pod, processes, auto-terminates")

    use_gpu = run_gpu
    if run_local or run_gpu:
        job_id = None
        try:
            files = {"csv_file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            run_res = requests.post(
                f"{BACKEND_URL}/run-pipeline",
                files=files,
                data={"use_gpu": "true" if use_gpu else "false"},
                timeout=60,
            )
            run_res.raise_for_status()
            job_id = run_res.json()["job_id"]

            progress_bar = st.progress(0)
            status_box = st.empty()
            anim_box = st.empty()
            render_working_animation(anim_box, "Running brand detection...", 0)

            with st.spinner("Processing..."):
                started = False
                for pct, message in stream_progress(job_id):
                    started = True
                    progress_bar.progress(pct)
                    status_box.info(message)
                    render_working_animation(anim_box, "Running brand detection...", pct)
                if not started:
                    status_box.info("Finishing...")
            anim_box.empty()
            status_box.empty()
            progress_bar.empty()

            dl_res = requests.get(f"{BACKEND_URL}/download/{job_id}", timeout=60)
            dl_res.raise_for_status()
            csv_bytes = dl_res.content

            st.success("Detection complete!")
            st.download_button(
                "Download Results CSV",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                width="stretch",
            )

            result_df = pd.read_csv(BytesIO(csv_bytes))
            _show_result_preview(result_df)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            if job_id:
                try:
                    dl_res = requests.get(f"{BACKEND_URL}/download/{job_id}", timeout=60)
                    if dl_res.status_code == 200:
                        csv_bytes = dl_res.content
                        st.warning("Partial results are available for download.")
                        st.download_button(
                            "Download Partial Results CSV",
                            data=csv_bytes,
                            file_name="partial_results.csv",
                            mime="text/csv",
                            width="stretch",
                        )
                        result_df = pd.read_csv(BytesIO(csv_bytes))
                        _show_result_preview(result_df)
                except Exception:
                    pass

    # ── Batch Processing History ──
    st.markdown("---")
    st.markdown("##### Processing History")
    try:
        hist_resp = requests.get(f"{BACKEND_URL}/batch-history", params={"limit": 20}, timeout=5)
        if hist_resp.status_code == 200:
            hist_jobs = hist_resp.json().get("jobs", [])
            if hist_jobs:
                for hj in hist_jobs:
                    job_id = hj.get("job_id", "?")
                    fname = hj.get("filename", "unknown")
                    status = hj.get("status", "?")
                    rows = hj.get("rows")
                    start = hj.get("start_time", "")[:19].replace("T", " ")
                    end = hj.get("end_time", "")[:19].replace("T", " ") if hj.get("end_time") else ""

                    # Status indicator
                    if status == "done":
                        icon = "[OK]"
                    elif status == "running":
                        icon = "[...]"
                    elif "error" in status:
                        icon = "[!]"
                    else:
                        icon = "[?]"

                    row_info = f" -- {rows} rows" if rows else ""
                    time_info = f" ({start})" if start else ""

                    col_info, col_dl = st.columns([3, 1])
                    with col_info:
                        st.caption(f"{icon} **{fname}**{row_info}{time_info}")
                        if status == "running":
                            st.caption("Processing...")
                        elif hj.get("error"):
                            with st.expander("Error details"):
                                st.code(hj["error"][:500])
                    with col_dl:
                        if hj.get("result_file") or status == "done":
                            try:
                                dl = requests.get(f"{BACKEND_URL}/download/{job_id}", timeout=10)
                                if dl.status_code == 200:
                                    st.download_button(
                                        "Download",
                                        data=dl.content,
                                        file_name=hj.get("result_file", f"{job_id}_results.csv"),
                                        mime="text/csv",
                                        key=f"hist_dl_{job_id}",
                                    )
                            except Exception:
                                st.caption("unavailable")
            else:
                st.caption("No batch jobs yet.")
    except Exception:
        st.caption("Could not load history.")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2: Single Image Test
# ════════════════════════════════════════════════════════
with tab_single:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Single Image Detection")
    st.caption("Upload a shelf photo to see RF-DETR bounding boxes. Hover over each box to see the detected brand and confidence.")

    if not index_exists:
        st.warning("No brand classifier found. Run `python brand_classifier.py` first or use the **Reference Index** tab to train one.")

    _MODEL_OPTIONS = {
        "RF-DETR Medium (default, 576px)": "medium",
        "RF-DETR Large (704px)": "large",
        "RF-DETR 2XLarge (880px, highest accuracy)": "2xlarge",
        "RF-DETR XLarge (700px)": "xlarge",
        "RF-DETR Base (560px, balanced)": "base",
        "RF-DETR Small (faster)": "small",
        "RF-DETR Nano (fastest, lightweight)": "nano",
    }
    selected_model_label = st.selectbox(
        "RF-DETR Model",
        options=list(_MODEL_OPTIONS.keys()),
        index=0,
        key="rfdetr_model_select",
    )
    selected_model_size = _MODEL_OPTIONS[selected_model_label]

    det_threshold = st.number_input(
        "Detection confidence threshold",
        min_value=0.05, max_value=1.0, value=0.25, step=0.05,
        format="%.2f",
        help="RF-DETR boxes below this confidence are hidden. Lower = more boxes, higher = fewer but more confident.",
        key="det_conf_threshold",
    )

    img_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
        key="single_img_upload",
    )

    if "single_img_key" not in st.session_state:
        st.session_state.single_img_key = None
    if "single_detect_done" not in st.session_state:
        st.session_state.single_detect_done = False
    if "single_interactive_html" not in st.session_state:
        st.session_state.single_interactive_html = None
    if "single_last_num_boxes" not in st.session_state:
        st.session_state.single_last_num_boxes = 0
    if "single_img_w" not in st.session_state:
        st.session_state.single_img_w = 1
    if "single_img_h" not in st.session_state:
        st.session_state.single_img_h = 1

    img_bytes = None
    if img_file is not None:
        img_bytes = img_file.getvalue()
        new_key = f"{img_file.name}:{len(img_bytes)}"
        if st.session_state.single_img_key != new_key:
            # New image uploaded -> reset so we show preview again.
            st.session_state.single_img_key = new_key
            st.session_state.single_detect_done = False
        st.session_state.single_interactive_html = None
        st.session_state.single_last_num_boxes = 0
        st.session_state.single_img_w = 1
        st.session_state.single_img_h = 1

    detect_disabled = img_file is None or not index_exists
    preview_placeholder = st.empty()
    if img_bytes is not None and not st.session_state.single_detect_done:
        st.caption('After you click "Detect Brands", this preview will be replaced by the interactive detection view.')
        preview_placeholder.image(img_bytes, caption="Uploaded image", width="stretch")

    run_single_clicked = st.button(
        "Detect Brands",
        disabled=detect_disabled,
        type="primary",
        width="stretch",
        key="run_single",
    )

    if run_single_clicked:
        anim_box = st.empty()
        render_working_animation(anim_box, "Analyzing image...", 40)

        try:
            files = {"image_file": (img_file.name, img_bytes, img_file.type or "image/jpeg")}
            resp = requests.post(
                f"{BACKEND_URL}/detect-single",
                files=files,
                data={"model_size": selected_model_size, "det_threshold": str(det_threshold)},
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            anim_box.empty()

            img_b64 = result["image_b64"]
            img_w = result["image_width"]
            img_h = result["image_height"]
            boxes = result["boxes"]
            num_boxes = result["num_boxes"]

            # -- Build interactive overlay HTML --
            # Color palette for boxes
            colors = [
                "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
                "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16",
                "#e879f9", "#facc15", "#fb923c", "#4ade80", "#a78bfa",
            ]

            overlay_divs = ""
            for i, box in enumerate(boxes):
                if box.get("is_full_image"):
                    continue
                # Convert pixel coords to percentages
                left_pct = (box["x1"] / img_w) * 100
                top_pct = (box["y1"] / img_h) * 100
                width_pct = ((box["x2"] - box["x1"]) / img_w) * 100
                height_pct = ((box["y2"] - box["y1"]) / img_h) * 100
                color = colors[i % len(colors)]

                # Build tooltip content from per-box brands
                brand_lines = ""
                box_brands = box.get("brands", [])
                if box_brands:
                    for br in box_brands:
                        conf = br["confidence"]
                        conf_pct = int(conf * 100)
                        bar_color = "#22c55e" if conf >= 0.70 else "#f59e0b" if conf >= 0.50 else "#ef4444"
                        brand_lines += f"""
                            <div style="margin-bottom:4px;">
                                <div style="display:flex;justify-content:space-between;font-size:11px;">
                                    <span>{br["brand"].replace("_", " ")}</span>
                                    <span style="color:{bar_color};font-weight:700;">{conf:.3f}</span>
                                </div>
                                <div style="background:rgba(255,255,255,0.1);border-radius:3px;height:4px;margin-top:2px;">
                                    <div style="background:{bar_color};height:4px;border-radius:3px;width:{conf_pct}%;"></div>
                                </div>
                            </div>"""
                else:
                    brand_lines = '<div style="font-size:11px;color:#9ea7be;">No brand match</div>'

                tooltip_pos = "below" if top_pct < 20 else "above"
                overlay_divs += f"""
                <div class="det-box" style="
                    left:{left_pct:.2f}%; top:{top_pct:.2f}%;
                    width:{width_pct:.2f}%; height:{height_pct:.2f}%;
                    border-color:{color};
                ">
                    <div class="det-label" style="background:{color};">#{i+1}</div>
                    <div class="det-tooltip {tooltip_pos}">
                        <div style="font-weight:700;font-size:12px;margin-bottom:6px;color:{color};">
                            Box #{i+1}
                            <span style="color:#9ea7be;font-weight:400;"> &middot; RF-DETR {box["det_conf"]:.2f}</span>
                        </div>
                        {brand_lines}
                    </div>
                </div>"""

            interactive_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px 0 10px 0; }}
.det-container {{
    position: relative;
    display: inline-block;
    width: 100%;
    line-height: 0;
}}
.det-container img {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
}}
.det-box {{
    position: absolute;
    border: 2px solid;
    border-radius: 4px;
    cursor: pointer;
    transition: border-width 0.15s, box-shadow 0.15s;
}}
.det-box:hover {{
    border-width: 3px;
    box-shadow: 0 0 12px rgba(255,255,255,0.15);
    z-index: 100;
}}
.det-label {{
    position: absolute;
    top: -1px; left: -1px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 700;
    color: #000;
    border-radius: 0 0 4px 0;
    line-height: 1.4;
    pointer-events: none;
}}
.det-tooltip {{
    display: none;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(15,18,30,0.97);
    border: 1px solid rgba(99,120,255,0.3);
    border-radius: 10px;
    padding: 10px 14px;
    min-width: 200px;
    max-width: 280px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    color: #e4e8f2;
    line-height: 1.5;
    z-index: 200;
    pointer-events: none;
}}
.det-tooltip.above {{
    bottom: calc(100% + 8px);
}}
.det-tooltip.above::after {{
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: rgba(15,18,30,0.97);
}}
.det-tooltip.below {{
    top: calc(100% + 8px);
}}
.det-tooltip.below::after {{
    content: '';
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-bottom-color: rgba(15,18,30,0.97);
}}
.det-box:hover .det-tooltip {{
    display: block;
}}
</style></head><body>
<div class="det-container">
    <img src="data:image/jpeg;base64,{img_b64}" alt="detection result" />
    {overlay_divs}
</div>
<script>
(function() {{
    var img = document.querySelector('.det-container img');
    function onReady() {{
        document.querySelectorAll('.det-box').forEach(function(box) {{
            var tooltip = box.querySelector('.det-tooltip');
            if (!tooltip) return;
            var boxTop = parseFloat(box.style.top);
            if (boxTop < 20) {{
                tooltip.classList.remove('above');
                tooltip.classList.add('below');
            }} else {{
                tooltip.classList.remove('below');
                tooltip.classList.add('above');
            }}
        }});
    }}
    if (img) {{
        img.addEventListener('load', onReady);
        if (img.complete) onReady();
    }}
}})();
</script>
</body></html>"""

            # Replace the upload preview with the zoomable interactive overlay.
            st.session_state.single_interactive_html = interactive_html
            st.session_state.single_last_num_boxes = num_boxes
            st.session_state.single_img_w = img_w
            st.session_state.single_img_h = img_h
            st.session_state.single_detect_done = True
            preview_placeholder.empty()
            st.markdown("##### Interactive Detection View")
            st.caption(f"{num_boxes} pack(s) detected. Hover over each box to see brand matches and confidence.")
            est_height = _single_detection_iframe_height(img_w, img_h)
            components.html(interactive_html, height=est_height, scrolling=True)

            # -- Brand summary table below --
            st.write("")
            col_brands, col_boxes = st.columns([1.2, 1], gap="medium")

            with col_brands:
                st.markdown("##### All Detected Brands")
                brands = result["brands"]
                confs = result["confidence"]
                if brands and brands[0] != "NO_DETECTION":
                    rows = []
                    for b, c in zip(brands, confs):
                        rows.append({"Brand": b, "Confidence": f"{c:.3f}", "Score": int(c * 100)})
                    brand_df = pd.DataFrame(rows)
                    st.dataframe(
                        brand_df,
                        width="stretch",
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Score", min_value=0, max_value=100, format="%d%%",
                            ),
                        },
                        hide_index=True,
                    )
                    above_thresh = sum(1 for c in confs if c >= 0.70)
                    st.caption(f"{above_thresh} brand(s) above 0.70 threshold, {len(brands)} total candidates")
                else:
                    st.info("No brands detected in this image.")

            with col_boxes:
                st.markdown("##### RF-DETR Bounding Boxes")
                display_boxes = [b for b in boxes if not b.get("is_full_image")]
                if display_boxes:
                    box_rows = []
                    for i, b in enumerate(display_boxes):
                        top_brand = b["brands"][0]["brand"] if b.get("brands") else "-"
                        top_conf = b["brands"][0]["confidence"] if b.get("brands") else 0.0
                        box_rows.append({
                            "#": i + 1,
                            "Det conf": b["det_conf"],
                            "Top brand": top_brand.replace("_", " "),
                            "Brand conf": round(top_conf, 3),
                        })
                    st.dataframe(pd.DataFrame(box_rows), width="stretch", hide_index=True)
                else:
                    st.info("RF-DETR found no cigarette packs. The full image was used for detection.")

        except Exception as exc:
            anim_box.empty()
            st.error(f"Detection failed: {exc}")
    else:
        if st.session_state.single_detect_done and st.session_state.get("single_interactive_html"):
            preview_placeholder.empty()
            last_num_boxes = st.session_state.get("single_last_num_boxes", 0)
            last_img_w = st.session_state.get("single_img_w", 1)
            last_img_h = st.session_state.get("single_img_h", 1)
            st.markdown("##### Interactive Detection View")
            st.caption(f"{last_num_boxes} pack(s) detected. Hover over each box to see brand matches and confidence.")
            est_height = _single_detection_iframe_height(last_img_w, last_img_h)
            components.html(st.session_state["single_interactive_html"], height=est_height, scrolling=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3: Reference Index
# ════════════════════════════════════════════════════════
with tab_index:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Reference Index")

    if index_exists:
        st.markdown(f'<div class="kpi">{idx_data.get("brand_count", 0)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Brand references indexed</div>', unsafe_allow_html=True)
        st.markdown('<span class="status-badge status-ready">&#9679; Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-missing">&#9679; No index</span>', unsafe_allow_html=True)
        st.caption("Build an index from reference images to start detecting.")

    st.write("")
    if st.button("Rebuild Index", width="stretch", key="rebuild_idx"):
        try:
            res = requests.post(f"{BACKEND_URL}/build-index", timeout=15)
            res.raise_for_status()
            job_id = res.json()["job_id"]

            progress_bar = st.progress(0)
            status_box = st.empty()
            anim_box = st.empty()
            render_working_animation(anim_box, "Retraining brand classifier...", 30)

            with st.spinner("Building..."):
                started = False
                for pct, message in stream_progress(job_id):
                    started = True
                    progress_bar.progress(pct)
                    status_box.info(message)
                    render_working_animation(anim_box, "Retraining brand classifier...", pct)
                if not started:
                    status_box.info("Finishing...")
            anim_box.empty()

            st.success("Index rebuilt successfully")
            st.rerun()
        except Exception as exc:
            st.error(f"Index rebuild failed: {exc}")

    # Brand/Product registry with reference counts
    st.markdown("---")
    st.markdown("##### Brand & Product Registry")
    try:
        brand_hierarchy, reg_data = _fetch_brand_hierarchy()

        st.caption(
            f"{reg_data.get('total_brands', 0)} brands, "
            f"{reg_data.get('total_products', 0)} products total -- "
            f"{reg_data.get('products_with_refs', 0)} with references, "
            f"{reg_data.get('products_missing', 0)} missing"
        )

        # Quick product picker with image preview gallery
        st.markdown("###### Product Reference Viewer")
        picker_col1, picker_col2 = st.columns(2)
        with picker_col1:
            picked_brand = st.selectbox(
                "Select brand",
                options=["-- Select brand --"] + sorted(brand_hierarchy.keys()),
                key="ref_picker_brand",
            )
        with picker_col2:
            picked_products = []
            if picked_brand != "-- Select brand --":
                picked_products = brand_hierarchy.get(picked_brand, [])
            picked_product_display = st.selectbox(
                "Select product",
                options=["-- Select product --"] + [p["display_name"] for p in picked_products],
                key="ref_picker_product",
            )

        if picked_brand != "-- Select brand --" and picked_product_display != "-- Select product --":
            picked_internal = ""
            for p in picked_products:
                if p["display_name"] == picked_product_display:
                    picked_internal = p["internal_name"]
                    break
            if picked_internal:
                ref_type = st.radio("Reference type", options=["pack", "box"], horizontal=True, key="ref_viewer_type")
                with st.expander(f"Show {ref_type} references for {picked_product_display}", expanded=True):
                    try:
                        ref_data = _fetch_reference_listing(picked_internal, ref_type)
                        filenames = ref_data.get("filenames", []) if ref_data else []
                        if not filenames:
                            st.caption(f"No {ref_type} reference images yet for this product.")
                        if filenames:
                            st.caption(f"{len(filenames)} {ref_type} reference images")

                            # Persistent selection across pages
                            sel_key = f"selected_refs_{ref_type}_{picked_internal}"
                            if sel_key not in st.session_state:
                                st.session_state[sel_key] = set()
                            # Clean out stale selections (files that no longer exist)
                            st.session_state[sel_key] &= set(filenames)

                            PAGE_SIZE = 50
                            page_key = f"ref_page_{ref_type}_{picked_internal}"
                            if page_key not in st.session_state:
                                st.session_state[page_key] = 0
                            total_pages = max(1, (len(filenames) + PAGE_SIZE - 1) // PAGE_SIZE)
                            current_page = st.session_state[page_key]
                            start = current_page * PAGE_SIZE
                            page_filenames = filenames[start:start + PAGE_SIZE]
                            if len(filenames) > PAGE_SIZE:
                                pg_col1, pg_col2, pg_col3, pg_col4 = st.columns([1, 2, 1, 2])
                                with pg_col1:
                                    if st.button("Prev", key=f"prev_{page_key}", disabled=current_page == 0):
                                        # Sync checkboxes on current page before navigating
                                        for _f in page_filenames:
                                            if st.session_state.get(f"sel_{ref_type}_{_f}", False):
                                                st.session_state[sel_key].add(_f)
                                            else:
                                                st.session_state[sel_key].discard(_f)
                                        st.session_state[page_key] = max(0, current_page - 1)
                                        st.rerun()
                                with pg_col2:
                                    st.caption(f"Page {current_page + 1}/{total_pages} ({len(filenames)} total)")
                                with pg_col3:
                                    if st.button("Next", key=f"next_{page_key}", disabled=current_page >= total_pages - 1):
                                        for _f in page_filenames:
                                            if st.session_state.get(f"sel_{ref_type}_{_f}", False):
                                                st.session_state[sel_key].add(_f)
                                            else:
                                                st.session_state[sel_key].discard(_f)
                                        st.session_state[page_key] = current_page + 1
                                        st.rerun()
                                with pg_col4:
                                    total_selected = len(st.session_state[sel_key])
                                    if total_selected:
                                        st.caption(f"{total_selected} selected across all pages")
                            cols = st.columns(min(5, len(page_filenames)))
                            for img_idx, fname in enumerate(page_filenames):
                                with cols[img_idx % len(cols)]:
                                    img_bytes = _fetch_reference_image_bytes(ref_type, fname)
                                    if img_bytes:
                                        st.image(img_bytes, caption=fname, width=110)
                                    st.checkbox("Select", key=f"sel_{ref_type}_{fname}", value=fname in st.session_state[sel_key])

                            # Sync current page checkboxes into persistent set
                            for fname in page_filenames:
                                if st.session_state.get(f"sel_{ref_type}_{fname}", False):
                                    st.session_state[sel_key].add(fname)
                                else:
                                    st.session_state[sel_key].discard(fname)
                            to_delete = sorted(st.session_state[sel_key])

                            if to_delete:
                                if st.button(f"Delete {len(to_delete)} selected", type="secondary", key=f"batch_del_{ref_type}_{picked_internal}"):
                                    deleted = 0
                                    for fname in to_delete:
                                        try:
                                            resp = requests.delete(f"{BACKEND_URL}/reference-image/{ref_type}/{fname}", timeout=5)
                                            if resp.status_code == 200:
                                                deleted += 1
                                        except Exception:
                                            pass
                                    if deleted:
                                        st.session_state[sel_key] = set()
                                        _fetch_reference_listing.clear()
                                        _fetch_brand_hierarchy.clear()
                                        st.rerun()

                                st.markdown("---")
                                st.markdown(f"**Move {len(to_delete)} selected to:**")
                                move_brand = st.selectbox(
                                    "Brand",
                                    options=["-- Select brand --"] + sorted(brand_hierarchy.keys()),
                                    key=f"move_brand_{ref_type}_{picked_internal}",
                                )
                                move_products_list = []
                                if move_brand != "-- Select brand --":
                                    move_products_list = brand_hierarchy.get(move_brand, [])
                                move_product_display = st.selectbox(
                                    "Product",
                                    options=["-- Select product --"] + [p["display_name"] for p in move_products_list],
                                    key=f"move_product_{ref_type}_{picked_internal}",
                                )
                                if move_brand != "-- Select brand --" and move_product_display != "-- Select product --":
                                    target_internal = next(p["internal_name"] for p in move_products_list if p["display_name"] == move_product_display)
                                    # Show brand book sample image for target product
                                    sample_bytes = _fetch_brand_book_sample(target_internal)
                                    if sample_bytes:
                                        st.image(sample_bytes, caption=f"Brand book: {move_product_display}", width=150)
                                    if st.button(f"Move {len(to_delete)} images", key=f"batch_move_{ref_type}_{picked_internal}"):
                                        moved = 0
                                        for fname in to_delete:
                                            try:
                                                resp = requests.post(
                                                    f"{BACKEND_URL}/reference-image/move",
                                                    json={"filename": fname, "packaging_type": ref_type, "target_product": target_internal},
                                                    timeout=5,
                                                )
                                                if resp.status_code == 200:
                                                    moved += 1
                                            except Exception:
                                                pass
                                        if moved:
                                            st.session_state[sel_key] = set()
                                            _fetch_reference_listing.clear()
                                            _fetch_brand_hierarchy.clear()
                                            st.success(f"Moved {moved} images to {move_brand} / {move_product_display}")
                                            st.rerun()

                    except Exception as exc:
                        st.caption(f"Could not load references: {exc}")

                    st.markdown("---")
                    st.markdown(f"**Add new reference image to {picked_product_display}:**")
                    upload_ref = st.file_uploader(
                        "Upload reference image",
                        type=["jpg", "jpeg", "png", "webp", "bmp"],
                        key=f"upload_ref_{ref_type}_{picked_internal}",
                    )
                    if upload_ref:
                        if st.button("Add to references", key=f"upload_ref_btn_{ref_type}_{picked_internal}"):
                            try:
                                resp = requests.post(
                                    f"{BACKEND_URL}/add-reference",
                                    files={"image_file": (upload_ref.name, upload_ref.getvalue(), upload_ref.type or "image/jpeg")},
                                    data={"product_name": picked_internal, "packaging_type": ref_type},
                                    timeout=15,
                                )
                                resp.raise_for_status()
                                result = resp.json()
                                st.success(f"Added as {ref_type}/{result['filename']}")
                                _fetch_reference_listing.clear()
                                _fetch_brand_hierarchy.clear()
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed: {exc}")

        st.markdown("---")

        for brand_idx, (brand_name, products) in enumerate(sorted(brand_hierarchy.items()), 1):
            total_refs = sum(p["reference_count"] for p in products)
            has_refs = total_refs > 0
            status = f"({total_refs} images)" if has_refs else "(no references)"

            with st.expander(f"{brand_idx}. {brand_name} -- {len(products)} products {status}", expanded=False):
                for prod_idx, prod in enumerate(products, 1):
                    count = prod["reference_count"]
                    pack_c = prod.get("pack_count", count)
                    box_c = prod.get("box_count", 0)
                    name = prod["display_name"]
                    internal = prod["internal_name"]
                    if count > 0:
                        type_detail = f"pack={pack_c}" + (f", box={box_c}" if box_c > 0 else "")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{brand_idx}.{prod_idx} **{name}** -- {count} refs ({type_detail})")
                        for vtype in ("pack", "box"):
                            vcount = pack_c if vtype == "pack" else box_c
                            if vcount == 0:
                                continue
                            with st.expander(f"View {name} {vtype} ({vcount})", expanded=False):
                                try:
                                    ref_data = _fetch_reference_listing(internal, vtype)
                                    if ref_data and ref_data.get("filenames"):
                                        fnames = ref_data["filenames"][:20]
                                        vcols = st.columns(min(5, len(fnames)))
                                        for vi, vfname in enumerate(fnames):
                                            with vcols[vi % len(vcols)]:
                                                vbytes = _fetch_reference_image_bytes(vtype, vfname)
                                                if vbytes:
                                                    st.image(vbytes, caption=vfname, width=120)
                                        if len(ref_data["filenames"]) > 20:
                                            st.caption(f"... and {len(ref_data['filenames']) - 20} more")
                                except Exception:
                                    st.caption("Could not load images")
                    else:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{brand_idx}.{prod_idx} ~~{name}~~ -- missing `(need: pack/{internal}_1.jpg)`")

    except Exception as exc:
        st.warning(f"Could not load brand registry: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 4: Label Training Crops
# ════════════════════════════════════════════════════════
with tab_label:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Label Training Crops")
    st.caption("Upload a shelf image, review detected crops, confirm brand and product, and add to training data.")

    # Load brand registry hierarchy (cached)
    brand_hierarchy, reg_data = _fetch_brand_hierarchy()
    if brand_hierarchy:
        st.caption(
            f"{reg_data.get('total_brands', 0)} brands, "
            f"{reg_data.get('total_products', 0)} products "
            f"({reg_data.get('products_with_refs', 0)} with references, "
            f"{reg_data.get('products_missing', 0)} missing)"
        )
    else:
        st.warning("Could not load brand registry from backend.")

    # Step 1: Upload image
    label_image = st.file_uploader(
        "Upload a shelf image to crop packs from",
        type=["jpg", "jpeg", "png", "webp"],
        key="label_upload",
    )

    if label_image is not None:
        # Auto-detect on upload: run detection if this is a new file
        _upload_id = f"{label_image.name}_{label_image.size}"
        if st.session_state.get("_label_last_upload") != _upload_id:
            st.session_state["_label_last_upload"] = _upload_id
            with st.spinner("Running RF-DETR detection..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/generate-crops",
                        files={"image_file": (label_image.name, label_image.getvalue(), "image/jpeg")},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    crop_data = resp.json()
                    st.session_state["label_crops"] = crop_data.get("crops", [])
                    st.session_state["label_results"] = {}
                    # Bump generation counter so widget keys reset for new image
                    st.session_state["label_gen"] = st.session_state.get("label_gen", 0) + 1
                    st.success(f"Detected {crop_data['num_crops']} packs")
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

    # Step 2: Review and label each crop
    crops = st.session_state.get("label_crops", [])
    if crops:
        st.markdown("---")
        st.markdown(f"##### Review {len(crops)} detected crops")
        st.caption("AI auto-suggests brand/product. Verify and click Add to confirm.")

        brand_names = sorted(brand_hierarchy.keys())
        import base64 as b64lib

        @st.fragment
        def _render_crop_card(crop, brand_names, brand_hierarchy):
            """Render a single crop card as a fragment -- selectbox changes only rerun this card."""
            col_img, col_form = st.columns([1, 2])

            with col_img:
                crop_bytes = b64lib.b64decode(crop["image_b64"])
                st.image(crop_bytes, caption=f"Crop #{crop['index']+1} ({crop['width']}x{crop['height']})", width=200)
                if crop.get("suggested_confidence", 0) > 0:
                    st.caption(f"AI suggests: **{crop.get('suggested_brand', '?')}** / {crop.get('suggested_product', '?').replace('_', ' ')} ({crop['suggested_confidence']:.0%})")
                detected_type = crop.get("packaging_type", "pack")
                if detected_type == "box":
                    st.caption("Detected as: **box**")

            with col_form:
                gen = st.session_state.get("label_gen", 0)
                crop_key = f"crop_{gen}_{crop['index']}"
                products_for_brand = []
                product_internals = {}
                selected_product = ""

                # Packaging type: default pack for label training (most refs are pack); RF-DETR hint shown on image
                type_options = ["pack", "box"]
                selected_type = st.radio(
                    "Type",
                    options=type_options,
                    index=0,
                    key=f"{crop_key}_type",
                    horizontal=True,
                )

                # Pre-select brand from AI suggestion
                suggested_brand = crop.get("suggested_brand", "")
                default_brand_idx = 0
                if suggested_brand in brand_names:
                    default_brand_idx = brand_names.index(suggested_brand) + 1

                selected_brand = st.selectbox(
                    "Brand",
                    options=["-- skip --"] + brand_names,
                    index=default_brand_idx,
                    key=f"{crop_key}_brand",
                )

                if selected_brand and selected_brand != "-- skip --":
                    products_for_brand = brand_hierarchy.get(selected_brand, [])
                    product_options = [p["display_name"] for p in products_for_brand]
                    product_internals = {p["display_name"]: p["internal_name"] for p in products_for_brand}

                    # Pre-select product from AI suggestion
                    suggested_product = crop.get("suggested_product", "")
                    default_prod_idx = 0
                    for pidx, p in enumerate(products_for_brand):
                        if p["internal_name"] == suggested_product:
                            default_prod_idx = pidx
                            break

                    selected_product = st.selectbox(
                        "Product",
                        options=product_options,
                        index=default_prod_idx,
                        key=f"{crop_key}_product",
                    )

                    # Show brand book sample for selected product
                    label_internal = product_internals.get(selected_product, "")
                    if label_internal:
                        sample_bytes = _fetch_brand_book_sample(label_internal)
                        if sample_bytes:
                            st.image(sample_bytes, caption=f"Brand book: {selected_product}", width=120)

                    if st.button("Add to references", key=f"{crop_key}_add"):
                        internal_name = product_internals.get(selected_product, "")
                        if internal_name:
                            try:
                                crop_bytes_data = b64lib.b64decode(crop["image_b64"])
                                resp = requests.post(
                                    f"{BACKEND_URL}/add-reference",
                                    files={"image_file": ("crop.jpg", crop_bytes_data, "image/jpeg")},
                                    data={"product_name": internal_name, "packaging_type": selected_type},
                                    timeout=15,
                                )
                                resp.raise_for_status()
                                result = resp.json()
                                st.success(f"Added as {selected_type}/{result['filename']} ({result['total_for_product']} total for {selected_product} [{selected_type}])")
                                _fetch_reference_listing.clear()
                                _fetch_brand_hierarchy.clear()
                                _fetch_reference_image_bytes.clear()
                            except Exception as exc:
                                st.error(f"Failed: {exc}")

                    # Show reference preview for selected type
                    internal_name = product_internals.get(selected_product, "")
                    if internal_name:
                        ref_data = _fetch_reference_listing(internal_name, selected_type)
                        if ref_data and ref_data.get("filenames"):
                            first_ref = ref_data["filenames"][0]
                            img_bytes = _fetch_reference_image_bytes(selected_type, first_ref)
                            if img_bytes:
                                st.image(
                                    img_bytes,
                                    caption=f"Reference preview: {selected_product} [{selected_type}]",
                                    width=170,
                                )
                            st.caption(f"{ref_data['count']} {selected_type} refs total")
                        elif ref_data:
                            st.caption(f"No {selected_type} reference images yet")
                        else:
                            st.caption("Reference preview unavailable")

            st.markdown("---")

        for crop in crops:
            _render_crop_card(crop, brand_names, brand_hierarchy)

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 5: Training
# ════════════════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Model Training")

    device_info = "unknown"
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        device_info = h.get("device", "unknown")
    except Exception:
        pass
    st.caption(f"Server device: **{device_info}**")
    training_version = "v1"
    last_trained_version = None
    try:
        reg = requests.get(f"{BACKEND_URL}/model-registry", params={"limit": 1}, timeout=5).json()
        training_version = reg.get("current_version", "v1")
        last_trained_version = reg.get("last_trained_version")
    except Exception:
        pass
    st.caption(f"Training version: **{training_version}**")
    if last_trained_version:
        st.caption(f"Latest trained version: **{last_trained_version}**")

    if st.button("Load Recommended Settings", key="btn_load_recommended_training"):
        st.session_state["cls_epochs"] = 100
        st.session_state["cls_lr"] = 0.001
        st.session_state["cls_batch"] = 64
        st.session_state["cls_embed_batch"] = 8
        st.session_state["dino_epochs"] = 30
        st.session_state["dino_lr"] = 0.00001
        st.session_state["dino_layers"] = 4
        st.session_state["dino_batch"] = 8
        st.rerun()

    train_col1, train_col2 = st.columns(2)

    # --- Fetch training dataset summary (shared by both columns) ---
    _train_hierarchy, _train_reg_data = _fetch_brand_hierarchy()
    _train_pack_total = 0
    _train_box_total = 0
    _train_products_with_refs = 0
    if _train_reg_data:
        # Use per-type totals from disk (references/pack, references/box). Summing
        # per-product pack_count only counts filenames that match BRAND_REGISTRY; unregistered
        # labels still train in brand_classifier.py but would show 0 here otherwise.
        _per_type = _train_reg_data.get("per_type") or {}
        _train_pack_total = int(_per_type.get("pack", 0) or 0)
        _train_box_total = int(_per_type.get("box", 0) or 0)
        _train_products_with_refs = int(_train_reg_data.get("products_with_refs", 0) or 0)

    # --- Brand Classifier (frozen DINOv2 + MLP) ---
    with train_col1:
        st.markdown("##### Brand Classifier")
        st.caption("Frozen DINOv2 + MLP head. Fast on CPU; **Run on RunPod** uses a GPU (default RTX 4090, set `RUNPOD_CLASSIFIER_GPU_ID` on the API server to change).")
        st.caption("Recommended: epochs=100, lr=0.001, batch=64, embed_batch=8")

        # Dataset summary
        st.markdown(f"**Training data (pack references):** {_train_pack_total} images, {_train_products_with_refs} products")
        if _train_pack_total == 0:
            st.warning("No pack reference images found. Add references in the Label Crops tab first.")
        else:
            with st.expander("View training data breakdown"):
                for _brand, _prods in sorted(_train_reg_data.get("brands", {}).items()):
                    for _p in _prods:
                        pc = _p.get("pack_count", 0)
                        if pc > 0:
                            st.caption(f"  {_p['display_name']}: {pc} images")

        cls_epochs = st.number_input("Epochs", value=100, min_value=10, max_value=500, key="cls_epochs")
        cls_lr = st.number_input("Learning rate", value=0.001, format="%.4f", key="cls_lr")
        cls_batch = st.number_input("Batch size", value=64, min_value=8, max_value=256, key="cls_batch")
        cls_embed_batch = st.number_input("Embed batch size", value=8, min_value=1, max_value=64, key="cls_embed_batch")
        cls_packaging_type = st.selectbox(
            "Packaging type",
            options=["all", "pack", "box"],
            index=0,
            key="cls_packaging_type",
            help="'all' trains one classifier on both pack and box references (recommended)",
        )
        cls_use_runpod = st.checkbox(
            "Run on RunPod GPU (faster; needs RUNPOD_API_KEY + SSH key on API server)",
            value=False,
            key="cls_use_runpod",
        )

        if st.button("Train Classifier", type="primary", key="btn_train_cls", disabled=_train_pack_total == 0):
            try:
                rp = "true" if cls_use_runpod else "false"
                resp = requests.post(
                    f"{BACKEND_URL}/train-classifier",
                    params={
                        "epochs": cls_epochs,
                        "lr": cls_lr,
                        "batch_size": cls_batch,
                        "embed_batch_size": cls_embed_batch,
                        "packaging_type": cls_packaging_type,
                        "version": training_version,
                        "use_runpod": rp,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("skipped"):
                    st.warning(
                        f"Skipped: same dataset/settings already trained for {training_version} "
                        f"(job: {str(result.get('existing_job_id', ''))[:8]}...)."
                    )
                else:
                    st.session_state["train_cls_job"] = result["job_id"]
                    st.success(f"Training started (job: {result['job_id'][:8]}...)")
            except Exception as exc:
                st.error(f"Failed to start: {exc}")

    # --- DINOv2 Fine-tune ---
    with train_col2:
        st.markdown("##### DINOv2 Fine-tune")
        st.caption("Unfreeze DINOv2 layers. Needs **GPU + ~16GB VRAM** for sensible speed.")
        st.caption(
            "By default this runs on the **same server as the API** (your droplet is CPU-only: very slow). "
            "Enable **Run on RunPod** to spin up a GPU pod (requires `RUNPOD_API_KEY` and an SSH key on the server, same as batch GPU)."
        )
        st.caption("Recommended: epochs=30, lr=1e-5, unfreeze_layers=4, batch=8")

        # Dataset summary (uses both pack + box)
        _dino_total = _train_pack_total + _train_box_total
        st.markdown(f"**Training data (all references):** {_dino_total} images (pack={_train_pack_total}, box={_train_box_total})")
        if _dino_total == 0:
            st.warning("No reference images found.")
        dino_epochs = st.number_input("Epochs", value=30, min_value=5, max_value=100, key="dino_epochs")
        dino_lr = st.number_input("Learning rate", value=0.00001, format="%.6f", key="dino_lr")
        dino_layers = st.number_input("Unfreeze layers", value=4, min_value=1, max_value=12, key="dino_layers")
        dino_batch = st.number_input("Batch size", value=8, min_value=2, max_value=32, key="dino_batch")
        dino_use_runpod = st.checkbox(
            "Run on RunPod GPU (recommended if API server has no GPU)",
            value=False,
            key="dino_use_runpod",
        )

        if st.button("Fine-tune DINOv2", type="primary", key="btn_train_dino"):
            try:
                rp = "true" if dino_use_runpod else "false"
                resp = requests.post(
                    f"{BACKEND_URL}/finetune-dinov2",
                    params={
                        "epochs": dino_epochs, "lr": dino_lr,
                        "batch_size": dino_batch, "unfreeze_layers": dino_layers,
                        "version": training_version,
                        "use_runpod": rp,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("skipped"):
                    st.warning(
                        f"Skipped: same dataset/settings already trained for {training_version} "
                        f"(job: {str(result.get('existing_job_id', ''))[:8]}...)."
                    )
                else:
                    st.session_state["train_dino_job"] = result["job_id"]
                    st.success(f"Training started (job: {result['job_id'][:8]}...)")
            except Exception as exc:
                st.error(f"Failed to start: {exc}")

    # --- Training Progress Monitor ---
    st.markdown("---")
    st.markdown("##### Training Progress")

    active_jobs = []
    for key in ["train_cls_job", "train_rfdetr_job", "train_dino_job"]:
        job_id = st.session_state.get(key, "")
        if job_id:
            label = key.replace("train_", "").replace("_job", "").upper()
            active_jobs.append((label, job_id))

    if active_jobs:
        @st.fragment(run_every=3)
        def _training_progress_panel(active_jobs):
            """Auto-refreshing fragment -- only this panel reruns, not the whole page."""
            for label, job_id in active_jobs:
                try:
                    resp = requests.get(f"{BACKEND_URL}/training-status/{job_id}", timeout=5)
                    if resp.status_code == 200:
                        status = resp.json()
                        progress = status.get("progress", {})
                        epoch = progress.get("epoch", 0)
                        total = progress.get("total_epochs", 1)
                        train_acc = progress.get("train_acc", 0)
                        val_acc = progress.get("val_acc", 0)
                        best_acc = progress.get("best_val_acc", 0)
                        job_status = status.get("status", "unknown")
                        last_update = status.get("last_update", "")
                        error_msg = status.get("error", "") or (progress.get("error", "") if progress else "")

                        pct = epoch / total if total > 0 else 0
                        st.markdown(f"**{label}** -- {job_status}")
                        st.progress(pct)
                        st.caption(
                            f"Epoch {epoch}/{total} | "
                            f"Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f} | "
                            f"Best: {best_acc:.3f}"
                        )
                        if last_update:
                            st.caption(f"Last update: {last_update}")

                        if error_msg:
                            with st.expander("Error details"):
                                st.code(str(error_msg)[-1000:])
                        if job_status in ("running", "queued", "stopping"):
                            if st.button(f"Terminate {label}", key=f"btn_stop_{job_id}"):
                                try:
                                    stop_resp = requests.post(f"{BACKEND_URL}/training-stop/{job_id}", timeout=10)
                                    stop_resp.raise_for_status()
                                    stop_data = stop_resp.json()
                                    st.warning(f"{label}: {stop_data.get('message', 'Stop requested')}")
                                except Exception as exc:
                                    st.error(f"Failed to terminate {label}: {exc}")
                except Exception:
                    st.caption(f"{label}: Could not fetch status")

        _training_progress_panel(active_jobs)
    else:
        st.caption("No active training jobs. Start one above.")

    st.markdown("---")
    st.markdown("##### Training History")
    try:
        history_resp = requests.get(f"{BACKEND_URL}/training-history", params={"limit": 50}, timeout=8)
        history_resp.raise_for_status()
        history_items = history_resp.json().get("items", [])
        if history_items:
            rows = []
            for item in history_items:
                progress = item.get("progress", {}) or {}
                rows.append({
                    "job_id": item.get("job_id", "")[:8],
                    "type": item.get("type", ""),
                    "version": item.get("version", "v1"),
                    "status": item.get("status", ""),
                    "epoch": f"{progress.get('epoch', 0)}/{progress.get('total_epochs', 0)}",
                    "val_acc": progress.get("val_acc", 0),
                    "best_val_acc": progress.get("best_val_acc", 0),
                    "start_time": item.get("start_time", ""),
                    "end_time": item.get("end_time", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No training history yet.")
    except Exception as exc:
        st.caption(f"Could not load training history: {exc}")

    st.markdown("##### Accuracy Trend")
    try:
        reg_resp = requests.get(f"{BACKEND_URL}/model-registry", params={"limit": 200}, timeout=8)
        reg_resp.raise_for_status()
        reg_items = reg_resp.json().get("items", [])
        trend_rows = []
        for item in reversed(reg_items):
            best = item.get("best_val_acc")
            val = item.get("val_acc")
            metric = best if best is not None else val
            if metric is None:
                continue
            trend_rows.append({
                "run": f"{item.get('model_type', '')}:{str(item.get('job_id', ''))[:8]}",
                "model_type": item.get("model_type", ""),
                "version": item.get("version", "v1"),
                "accuracy": float(metric),
            })
        if trend_rows:
            trend_df = pd.DataFrame(trend_rows)
            st.line_chart(trend_df.set_index("run")["accuracy"], use_container_width=True)
            st.caption("Shows validation accuracy (best_val_acc when available) across runs.")
        else:
            st.caption("No accuracy points yet (run classifier or DINOv2 fine-tune first).")
    except Exception as exc:
        st.caption(f"Could not load accuracy trend: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)
