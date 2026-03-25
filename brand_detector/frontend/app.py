import json
from io import BytesIO
from typing import Generator

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"
BATCH_MODE_LABEL = "ALL ROWS"


def stream_progress(job_id: str) -> Generator[tuple[int, str], None, None]:
    url = f"{BACKEND_URL}/progress/{job_id}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
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


def render_working_animation(target, text: str = "Model is working..."):
    target.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;
                    border:1px solid rgba(129,149,255,0.25);
                    border-radius:12px;padding:10px 12px;
                    background:rgba(19,24,39,0.75);margin-bottom:8px;">
            <div class="worker-wrap">
                <span class="worker-emoji">👩‍💻</span>
            </div>
            <div style="color:#cfd7ef;font-size:14px;">{text}</div>
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
        display_cols = ["final_brand", "overall_confidence"] + [
            c for c in result_df.columns
            if c.endswith("_detected_brands") or c.endswith("_confidence")
        ]
        display_cols = [c for c in display_cols if c in result_df.columns]
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


st.set_page_config(page_title="Brand Detector", layout="wide", page_icon="🔍")
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #151a2b, #0c0f18 45%, #090b12);
            color: #e9edf5;
        }
        .block-container {
            padding-top: 4.6rem !important;
            padding-bottom: 1.2rem !important;
            max-width: 1300px;
        }
        .panel {
            border: 1px solid rgba(129, 149, 255, 0.18);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(20,26,43,0.95), rgba(12,16,27,0.95));
            padding: 16px 18px;
            margin-bottom: 16px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.3);
        }
        .kpi {
            font-size: 40px;
            font-weight: 800;
            line-height: 1.0;
            color: #f2f5ff;
        }
        .muted {
            color: #9ea7be;
            font-size: 13px;
        }
        .app-title {
            font-size: 30px;
            font-weight: 800;
            margin-bottom: 4px;
            letter-spacing: 0.2px;
        }
        .app-subtitle {
            color: #9ea7be;
            margin-bottom: 16px;
        }
        .worker-wrap {
            width: 34px;
            text-align: center;
            animation: bounce 1.1s ease-in-out infinite;
        }
        .worker-emoji {
            font-size: 22px;
            display: inline-block;
            animation: blink 1.6s steps(1, end) infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-4px); }
        }
        @keyframes blink {
            0%, 45%, 100% { opacity: 1; }
            50%, 55% { opacity: 0.55; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Local Cigarette Brand Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">YOLO + DINO + OCR pipeline for shelf-brand detection</div>',
    unsafe_allow_html=True,
)

try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5)
    health.raise_for_status()
except Exception:
    st.error("Backend not running. Start it with:\ncd backend && uvicorn main:app --port 8000")
    st.stop()

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Reference Index Status")
    index_exists = False
    idx_data = {"brand_count": 0, "brands": []}
    try:
        idx_resp = requests.get(f"{BACKEND_URL}/index-status", timeout=10)
        idx_resp.raise_for_status()
        idx_data = idx_resp.json()
        index_exists = bool(idx_data.get("exists", False))
        if index_exists:
            st.markdown(f'<div class="kpi">{idx_data.get("brand_count", 0)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="muted">Indexed brand references ready for matching</div>', unsafe_allow_html=True)
            st.success("Index available")
        else:
            st.warning("No index found")
    except Exception as exc:
        st.error(f"Failed to fetch index status: {exc}")

    if st.button("Rebuild Index", width="stretch"):
        try:
            res = requests.post(f"{BACKEND_URL}/build-index", timeout=15)
            res.raise_for_status()
            job_id = res.json()["job_id"]

            progress_bar = st.progress(0)
            status_box = st.empty()
            anim_box = st.empty()
            render_working_animation(anim_box, "Rebuilding FAISS index...")

            with st.spinner("Waiting for index progress..."):
                started = False
                for pct, message in stream_progress(job_id):
                    started = True
                    progress_bar.progress(pct)
                    status_box.info(message)
                if not started:
                    status_box.info("Finishing...")
            anim_box.empty()

            st.success("Index rebuilt successfully")
            st.rerun()
        except Exception as exc:
            st.error(f"Index rebuild failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

    if index_exists:
        with st.expander("Indexed brand list", expanded=False):
            st.json(idx_data.get("brands", []))

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Detect Brands")
    st.caption("Batch process CSV files from field surveys and shelf photo links.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    preview_df = None
    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
            st.dataframe(preview_df.head(10), width="stretch")
            st.caption(f"Total rows: {len(preview_df)}. Batch mode: {BATCH_MODE_LABEL}.")
            uploaded.seek(0)
        except Exception as exc:
            st.error(f"Could not preview CSV: {exc}")

    run_disabled = uploaded is None or not index_exists
    if st.button("Run Detection", disabled=run_disabled, width="stretch"):
        job_id = None
        try:
            files = {"csv_file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            run_res = requests.post(f"{BACKEND_URL}/run-pipeline", files=files, timeout=60)
            run_res.raise_for_status()
            job_id = run_res.json()["job_id"]

            progress_bar = st.progress(0)
            status_box = st.empty()
            anim_box = st.empty()
            render_working_animation(anim_box, "Running detection on uploaded batch...")

            with st.spinner("Waiting for detection progress..."):
                started = False
                for pct, message in stream_progress(job_id):
                    started = True
                    progress_bar.progress(pct)
                    status_box.info(message)
                if not started:
                    status_box.info("Finishing...")
            anim_box.empty()

            dl_res = requests.get(f"{BACKEND_URL}/download/{job_id}", timeout=60)
            dl_res.raise_for_status()
            csv_bytes = dl_res.content

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
    st.markdown("</div>", unsafe_allow_html=True)
