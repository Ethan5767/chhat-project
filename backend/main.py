import io
import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests as http_requests  # avoid clash with fastapi.requests

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

try:
    from .pipeline import (
        CLASSIFIER_WEIGHTS,
        CLASS_MAPPING_JSON,
        build_index,
        get_device,
        load_dino,
        load_index,
        load_classifier,
        load_rfdetr,
        reload_rfdetr,
        reload_classifiers,
        reload_dino,
        run_pipeline,
        classify_embeddings,
        _detect_brands_from_image,
        _format_brand_scores,
        label_to_product,
        _aggregate_to_products,
        MIN_OUTPUT_CONFIDENCE,
        RFDETR_CONF_THRESHOLD,
        PACKAGING_TYPES,
    )
except ImportError:
    from pipeline import (
        CLASSIFIER_WEIGHTS,
        CLASS_MAPPING_JSON,
        build_index,
        get_device,
        load_dino,
        load_index,
        load_classifier,
        load_rfdetr,
        reload_rfdetr,
        reload_classifiers,
        reload_dino,
        run_pipeline,
        classify_embeddings,
        _detect_brands_from_image,
        _format_brand_scores,
        label_to_product,
        _aggregate_to_products,
        MIN_OUTPUT_CONFIDENCE,
        RFDETR_CONF_THRESHOLD,
        PACKAGING_TYPES,
    )

app = FastAPI(title="Local Cigarette Brand Detector")

_BACKEND_ROOT = Path(__file__).resolve().parent
# Persistent data root -- survives code deploys. Default: /opt/chhat-data (server) or backend/ (local dev).
_DATA_ROOT = Path(os.environ.get("CHHAT_DATA_ROOT", str(_BACKEND_ROOT)))
_DATA_ROOT.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = _DATA_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = UPLOADS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_BATCH_HISTORY_PATH = _DATA_ROOT / "batch_history.json"

jobs_lock = threading.Lock()
jobs: dict[str, dict] = {}
_batch_history_lock = threading.Lock()


def _result_meta_path(job_id: str) -> Path:
    return RESULTS_DIR / f"{job_id}.json"


def _save_result_meta(job_id: str, result_path: Path) -> None:
    meta = {"job_id": job_id, "result": str(result_path)}
    _result_meta_path(job_id).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _load_result_meta(job_id: str) -> Path | None:
    meta_path = _result_meta_path(job_id)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        result = payload.get("result")
        if not result:
            return None
        path = Path(result)
        return path if path.exists() else None
    except Exception:
        return None


def _load_batch_history() -> list[dict]:
    if not _BATCH_HISTORY_PATH.exists():
        return []
    try:
        return json.loads(_BATCH_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _append_batch_history(entry: dict) -> None:
    with _batch_history_lock:
        rows = _load_batch_history()
        rows.append(entry)
        _BATCH_HISTORY_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_batch_history(job_id: str, updates: dict) -> None:
    with _batch_history_lock:
        rows = _load_batch_history()
        for row in rows:
            if row.get("job_id") == job_id:
                row.update(updates)
        _BATCH_HISTORY_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "result": None,
            "error": None,
            "queue": queue.Queue(),
        }
    return job_id


def update_progress(job_id: str, current: int, total: int, message: str):
    pct = int((current / total) * 100) if total > 0 else 0
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job["queue"].put((pct, message))


def run_build_index_job(job_id: str):
    try:
        device = get_device()
        build_index(device, progress_cb=lambda c, t, m: update_progress(job_id, c, t, f"Indexing: {m}"))
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = "INDEX_REBUILT"
    except Exception:
        err = traceback.format_exc()
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = err


def run_pipeline_job(job_id: str, csv_path: Path):
    start_time = datetime.now(timezone.utc).isoformat()
    _append_batch_history({
        "job_id": job_id,
        "filename": csv_path.name,
        "status": "running",
        "start_time": start_time,
        "end_time": None,
        "rows": None,
        "error": None,
    })
    try:
        out_path = run_pipeline(
            csv_path,
            progress_cb=lambda c, t, m: update_progress(job_id, c, t, m),
        )
        job_out_path = RESULTS_DIR / f"{job_id}_{out_path.name}"
        out_path.replace(job_out_path)
        _save_result_meta(job_id, job_out_path)
        # Count rows in result
        try:
            import pandas as pd
            result_df = pd.read_csv(job_out_path)
            row_count = len(result_df)
        except Exception:
            row_count = None
        _update_batch_history(job_id, {
            "status": "done",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "rows": row_count,
            "result_file": job_out_path.name,
        })
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = str(job_out_path)
    except Exception:
        err = traceback.format_exc()
        partial_path = csv_path.parent / f"{csv_path.stem}_results.csv"
        if partial_path.exists() and partial_path.stat().st_size > 0:
            job_out_path = RESULTS_DIR / f"{job_id}_{partial_path.name}"
            partial_path.replace(job_out_path)
            _save_result_meta(job_id, job_out_path)
            _update_batch_history(job_id, {
                "status": "error_partial",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": str(err)[:500],
                "result_file": job_out_path.name,
            })
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = err
                jobs[job_id]["result"] = str(job_out_path)
        else:
            _update_batch_history(job_id, {
                "status": "error",
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error": str(err)[:500],
            })
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = err


# ── RunPod GPU batch / DINO jobs ──
# All SSH targets below are ephemeral RunPod pods under /workspace/... — not your production droplet.
# (rm -rf / git clone on pods is scoped to that pod only.)

RUNPOD_API_URL = "https://api.runpod.io/graphql"
RUNPOD_GPU_ID = "NVIDIA A100 80GB PCIe"
RUNPOD_TEMPLATE = "runpod-torch-v240"
RUNPOD_REPO = "https://github.com/Ethan5767/chhat-project.git"


CLASSIFIER_GPU_FALLBACK_CHAIN = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A4000",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA RTX A4500",
    "NVIDIA Tesla T4",
    "NVIDIA GeForce RTX 3070",
    "NVIDIA GeForce RTX 3060",
]


def _classifier_gpu_candidates():
    """Preferred GPU first (RUNPOD_CLASSIFIER_GPU_ID), then fallbacks — avoids a single busy type failing the job."""
    preferred = (os.environ.get("RUNPOD_CLASSIFIER_GPU_ID") or "").strip()
    out = []
    if preferred:
        out.append(preferred)
    for g in CLASSIFIER_GPU_FALLBACK_CHAIN:
        if g not in out:
            out.append(g)
    return out


def _classifier_cloud_types():
    """RunPod cloudType values to try in order.

    Default SECURE then COMMUNITY — fewer 'no resources' failures (SECURE is usually easier to place).
    Cheaper-first: RUNPOD_CLASSIFIER_CLOUD_TYPE=COMMUNITY,SECURE
    """
    raw = (os.environ.get("RUNPOD_CLASSIFIER_CLOUD_TYPE") or "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["SECURE", "COMMUNITY"]


def _classifier_volume_gb() -> int:
    """Workspace volume size for classifier pods (smaller can be easier to schedule)."""
    try:
        v = int((os.environ.get("RUNPOD_CLASSIFIER_VOLUME_GB") or "50").strip())
    except ValueError:
        v = 50
    return max(10, min(v, 200))


def _classifier_deploy_retry_settings():
    """(rounds, seconds_between_rounds) — full GPU×cloud matrix retried when everything is busy."""
    try:
        rounds = int((os.environ.get("RUNPOD_CLASSIFIER_DEPLOY_RETRIES") or "5").strip())
    except ValueError:
        rounds = 5
    try:
        pause = int((os.environ.get("RUNPOD_CLASSIFIER_DEPLOY_RETRY_SECS") or "40").strip())
    except ValueError:
        pause = 40
    return max(1, min(rounds, 20)), max(15, min(pause, 300))


def _get_runpod_classifier_gpu_id() -> str:
    """Log label / default GPU id for classifier RunPod jobs."""
    c = _classifier_gpu_candidates()
    return c[0] if c else CLASSIFIER_GPU_FALLBACK_CHAIN[0]


def _log_runpod(msg: str) -> None:
    """Console log for RunPod / GPU jobs (visible in journalctl for systemd)."""
    print(f"[runpod] {msg}", flush=True)


def _tar_backend_references_for_runpod(refs_tar: Path) -> subprocess.CompletedProcess:
    """Tar backend/references for upload to a RunPod pod.

    Copies to a temp directory first (rsync or shutil.copytree) so the archive is taken from a
    stable snapshot — avoids tar exit 1 from concurrent writes under the live references tree.
    """
    src = _DATA_ROOT / "references"
    if not src.is_dir():
        return subprocess.CompletedProcess(
            args=("_tar_backend_references_for_runpod",),
            returncode=1,
            stdout="",
            stderr=f"references directory missing: {src}",
        )

    staging_root = Path(tempfile.mkdtemp(prefix="refs_tar_stage_", dir="/tmp"))
    staging_refs = staging_root / "references"
    try:
        rsync_bin = shutil.which("rsync")
        if rsync_bin:
            r0 = subprocess.run(
                [rsync_bin, "-a", "--", f"{str(src)}/", f"{str(staging_refs)}/"],
                capture_output=True,
                text=True,
                timeout=3600,
            )
        else:
            _log_runpod("refs: rsync not found, using shutil.copytree (slower)")
            try:
                shutil.copytree(src, staging_refs)
            except Exception as exc:
                return subprocess.CompletedProcess(
                    args=("copytree",),
                    returncode=1,
                    stdout="",
                    stderr=str(exc),
                )
            r0 = subprocess.CompletedProcess(
                args=("copytree",), returncode=0, stdout="", stderr="",
            )

        if r0.returncode != 0:
            return r0

        return subprocess.run(
            ["tar", "-czf", str(refs_tar), "-C", str(staging_root), "references"],
            capture_output=True,
            text=True,
            timeout=900,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _fmt_subprocess_fail(proc: subprocess.CompletedProcess, label: str) -> str:
    """Human-readable snippet for failed tar/rsync (stderr is often empty)."""
    parts = [f"{label} rc={proc.returncode}"]
    if proc.stderr and proc.stderr.strip():
        parts.append(f"stderr={proc.stderr.strip()[:800]}")
    if proc.stdout and proc.stdout.strip():
        parts.append(f"stdout={proc.stdout.strip()[:800]}")
    if len(parts) == 1:
        parts.append("(no stderr/stdout captured)")
    return " | ".join(parts)


def _mask_secret_hint(value: str) -> str:
    """Non-secret fingerprint for logs (length + tiny prefix/suffix)."""
    if not value:
        return "(empty)"
    v = value.strip()
    if len(v) <= 12:
        return f"len={len(v)}"
    return f"len={len(v)} prefix={v[:4]}… suffix=…{v[-3:]}"


def _get_runpod_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    env_file = _BACKEND_ROOT.parent / ".env"
    if env_file.exists():
        # utf-8-sig strips BOM so the first line still matches RUNPOD_API_KEY=
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if stripped.startswith("RUNPOD_API_KEY="):
                return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _runpod_gql(api_key: str, query: str, variables: dict = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = http_requests.post(RUNPOD_API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        err_msg = data["errors"][0].get("message", data["errors"])
        _log_runpod(f"GraphQL error: {err_msg}")
        raise RuntimeError(f"RunPod API: {err_msg}")
    return data["data"]


RUNPOD_SSH_HOST = "ssh.runpod.io"


def _ssh_scp_common_opts(key: str) -> list:
    """OpenSSH options for non-interactive access to ephemeral RunPod pods."""
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-o", "IdentitiesOnly=yes",
        "-i", key,
    ]


def _runpod_ssh_user(pod_id: str, pod_host_id: str) -> str:
    """Build the SSH username for RunPod proxy.

    machine.podHostId already contains the full SSH username (e.g. 'podid-64411df3').
    Use it directly. Only prepend pod_id if it's not already included.
    """
    if pod_host_id.startswith(pod_id):
        return pod_host_id
    return f"{pod_id}-{pod_host_id}"


def _paramiko_connect(pod_id: str, pod_host_id: str, key: str, connect_timeout: int = 45):
    """Connect to RunPod pod directly via Paramiko through ssh.runpod.io.

    RunPod's proxy acts as a transparent relay -- Paramiko connects to it
    with the pod username, and the proxy forwards to the pod's sshd.
    No ssh -W or PTY needed.
    """
    import paramiko

    user = _runpod_ssh_user(pod_id, pod_host_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=RUNPOD_SSH_HOST,
        port=22,
        username=user,
        key_filename=key,
        timeout=connect_timeout,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def _paramiko_proxy_exec(pod_id: str, pod_host_id: str, key: str, command: str,
                         timeout: int = 300) -> subprocess.CompletedProcess:
    """Execute command on RunPod pod via interactive shell through ssh.runpod.io.

    RunPod's proxy doesn't support exec_command -- only interactive shell.
    We invoke_shell(), send the command + exit, and read all output.
    We embed a unique marker to extract the exit code.
    """
    import time as _time

    try:
        client = _paramiko_connect(pod_id, pod_host_id, key)
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["paramiko", command], returncode=255, stdout="",
            stderr=f"paramiko connect failed: {exc}",
        )
    try:
        channel = client.invoke_shell(term="xterm", width=200, height=50)
        channel.settimeout(5.0)

        # Wait for shell prompt / banner to flush
        _time.sleep(2)
        banner = b""
        while channel.recv_ready():
            banner += channel.recv(65536)

        # Send command with exit code marker, then exit shell
        marker = "__PARAMIKO_EXIT_CODE__"
        full_cmd = f"{command}\necho {marker}$?\nexit\n"
        channel.sendall(full_cmd.encode())

        # Read all output until channel closes
        out_chunks = []
        deadline = _time.monotonic() + float(timeout)
        while _time.monotonic() < deadline:
            try:
                data = channel.recv(65536)
                if not data:
                    break
                out_chunks.append(data.decode("utf-8", errors="replace"))
            except Exception:
                if channel.closed:
                    break
                continue

        channel.close()
        full_output = "".join(out_chunks)

        # Extract exit code from marker
        exit_code = 0
        for line in full_output.splitlines():
            if line.strip().startswith(marker):
                try:
                    exit_code = int(line.strip().replace(marker, ""))
                except ValueError:
                    pass
                break

        return subprocess.CompletedProcess(
            args=["paramiko", command], returncode=exit_code,
            stdout=full_output, stderr="",
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["paramiko", command], returncode=255, stdout="",
            stderr=f"paramiko shell failed: {exc}",
        )
    finally:
        client.close()


def _do_spaces_client():
    """Create a boto3 S3 client for DigitalOcean Spaces."""
    import boto3
    return boto3.client(
        "s3",
        region_name=os.environ.get("DO_SPACES_REGION", "sgp1"),
        endpoint_url=os.environ.get("DO_SPACES_ENDPOINT", "https://sgp1.digitaloceanspaces.com"),
        aws_access_key_id=os.environ.get("DO_SPACES_KEY", ""),
        aws_secret_access_key=os.environ.get("DO_SPACES_SECRET", ""),
    )


def _do_spaces_bucket() -> str:
    return os.environ.get("DO_SPACES_BUCKET", "chhat")


def _paramiko_proxy_upload(pod_id: str, pod_host_id: str, key: str,
                           local: str, remote: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Upload file to RunPod pod via DO Spaces.

    SFTP doesn't work through RunPod's ssh.runpod.io proxy.
    Flow: server -> DO Spaces (pre-signed URL) -> pod wget.
    """
    import uuid as _uuid
    s3_key = f"runpod-xfer/{_uuid.uuid4().hex}/{Path(local).name}"
    bucket = _do_spaces_bucket()
    try:
        s3 = _do_spaces_client()
        file_size = Path(local).stat().st_size
        _log_runpod(f"  DO Spaces: uploading {Path(local).name} ({file_size / 1e6:.1f} MB)…")
        s3.upload_file(local, bucket, s3_key)
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=3600,
        )
        _log_runpod(f"  DO Spaces: upload done, pod wget…")
        r = _paramiko_proxy_exec(
            pod_id, pod_host_id, key,
            f'wget -q -O "{remote}" "{url}"',
            timeout=timeout,
        )
        if r.returncode != 0:
            return subprocess.CompletedProcess(
                args=["do-spaces-upload", local, remote], returncode=1,
                stdout=r.stdout, stderr=f"pod wget failed: {r.stderr}",
            )
        _log_runpod(f"  DO Spaces: pod wget done")
        return subprocess.CompletedProcess(
            args=["do-spaces-upload", local, remote], returncode=0, stdout="", stderr="",
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["do-spaces-upload", local, remote], returncode=1,
            stdout="", stderr=f"DO Spaces upload failed: {exc}",
        )
    finally:
        try:
            _do_spaces_client().delete_object(Bucket=bucket, Key=s3_key)
        except Exception:
            pass


def _paramiko_proxy_download(pod_id: str, pod_host_id: str, key: str,
                             remote: str, local: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Download file from RunPod pod via DO Spaces.

    SFTP doesn't work through RunPod's ssh.runpod.io proxy.
    Flow: pod curl PUT -> DO Spaces (pre-signed URL) -> server download.
    """
    import uuid as _uuid
    s3_key = f"runpod-xfer/{_uuid.uuid4().hex}/{Path(remote).name}"
    bucket = _do_spaces_bucket()
    try:
        s3 = _do_spaces_client()
        put_url = s3.generate_presigned_url(
            "put_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=3600,
        )
        _log_runpod(f"  DO Spaces: pod uploading {Path(remote).name}…")
        r = _paramiko_proxy_exec(
            pod_id, pod_host_id, key,
            f'curl -s -X PUT -T "{remote}" "{put_url}"',
            timeout=timeout,
        )
        if r.returncode != 0:
            return subprocess.CompletedProcess(
                args=["do-spaces-download", remote, local], returncode=1,
                stdout=r.stdout, stderr=f"pod curl upload failed: {r.stderr}",
            )
        _log_runpod(f"  DO Spaces: downloading {Path(remote).name} to server…")
        s3.download_file(bucket, s3_key, local)
        _log_runpod(f"  DO Spaces: download done")
        return subprocess.CompletedProcess(
            args=["do-spaces-download", remote, local], returncode=0, stdout="", stderr="",
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["do-spaces-download", remote, local], returncode=1,
            stdout="", stderr=f"DO Spaces download failed: {exc}",
        )
    finally:
        try:
            _do_spaces_client().delete_object(Bucket=bucket, Key=s3_key)
        except Exception:
            pass


def _ssh_cmd(host: str, port: int, key: str, command: str, timeout: int = 300,
             pod_id: str = "", pod_host_id: str = "") -> subprocess.CompletedProcess:
    """Run command on remote via SSH. Uses Paramiko proxy if pod_id/pod_host_id provided."""
    if pod_id and pod_host_id:
        return _paramiko_proxy_exec(pod_id, pod_host_id, key, command, timeout)
    opts = _ssh_scp_common_opts(key)
    return subprocess.run(
        ["ssh", "-T", *opts,
         "-o", "ServerAliveInterval=30",
         "-p", str(port), f"root@{host}", command],
        capture_output=True, text=True, timeout=timeout,
    )


def _scp_to(host: str, port: int, key: str, local: str, remote: str, timeout: int = 300,
            pod_id: str = "", pod_host_id: str = ""):
    """Upload file to remote host. Uses DO Spaces transfer if pod_id/pod_host_id provided."""
    if pod_id and pod_host_id:
        return _paramiko_proxy_upload(pod_id, pod_host_id, key, local, remote, timeout)
    opts = _ssh_scp_common_opts(key)
    return subprocess.run(
        ["scp", *opts, "-P", str(port), local, f"root@{host}:{remote}"],
        capture_output=True, text=True, timeout=timeout,
    )


def _scp_from(host: str, port: int, key: str, remote: str, local: str, timeout: int = 300,
              pod_id: str = "", pod_host_id: str = ""):
    """Download file from remote host. Uses DO Spaces transfer if pod_id/pod_host_id provided."""
    if pod_id and pod_host_id:
        return _paramiko_proxy_download(pod_id, pod_host_id, key, remote, local, timeout)
    opts = _ssh_scp_common_opts(key)
    return subprocess.run(
        ["scp", *opts, "-P", str(port), f"root@{host}:{remote}", local],
        capture_output=True, text=True, timeout=timeout,
    )


def run_pipeline_gpu_job(job_id: str, csv_path: Path):
    """Process a CSV batch on a RunPod GPU pod. Creates pod, uploads, runs, downloads, terminates."""
    import os
    import time as _time

    api_key = _get_runpod_api_key()
    if not api_key:
        _log_runpod("gpu-batch: RUNPOD_API_KEY missing — set in .env and restart chhat-backend")
        _append_batch_history({
            "job_id": job_id, "filename": csv_path.name, "status": "error",
            "start_time": datetime.now(timezone.utc).isoformat(), "end_time": datetime.now(timezone.utc).isoformat(),
            "error": "RUNPOD_API_KEY not set. Add it to .env or environment variables.", "rows": None,
        })
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "RUNPOD_API_KEY not set"
        return

    _log_runpod(f"gpu-batch job={job_id[:8]}… csv={csv_path.name} key={_mask_secret_hint(api_key)}")
    start_time = datetime.now(timezone.utc).isoformat()
    _append_batch_history({
        "job_id": job_id, "filename": csv_path.name, "status": "running_gpu",
        "start_time": start_time, "end_time": None, "rows": None, "error": None,
    })

    pod_id = None
    try:
        update_progress(job_id, 1, 100, "Creating RunPod GPU pod...")

        # 1. Create pod
        pod = _runpod_gql(api_key, """
            mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
                podFindAndDeployOnDemand(input: $input) { id costPerHr machine { podHostId } }
            }""", {"input": {
                "name": f"batch-{job_id[:8]}",
                "templateId": RUNPOD_TEMPLATE,
                "gpuTypeId": RUNPOD_GPU_ID,
                "cloudType": "COMMUNITY",
                "containerDiskInGb": 20,
                "volumeInGb": 50,
                "volumeMountPath": "/workspace",
                "gpuCount": 1,
                "ports": "22/tcp",
            }})["podFindAndDeployOnDemand"]
        pod_id = pod["id"]
        pod_host_id = (pod.get("machine") or {}).get("podHostId", "")
        if not pod_host_id:
            _log_runpod("gpu-batch: WARNING podHostId empty -- SSH proxy disabled, falling back to direct IP")
        cost_hr = pod.get("costPerHr", 0)
        _log_runpod(f"gpu-batch: pod created id={pod_id} host={pod_host_id} ~${cost_hr}/hr template={RUNPOD_TEMPLATE} gpu={RUNPOD_GPU_ID}")
        update_progress(job_id, 5, 100, f"Pod created ({pod_id[:8]}..., ${cost_hr}/hr). Waiting for SSH...")

        # 2. Wait for SSH port
        ssh_host = ssh_port = None
        for attempt in range(30):
            d = _runpod_gql(api_key, f"""query {{
                pod(input: {{ podId: "{pod_id}" }}) {{
                    runtime {{ uptimeInSeconds ports {{ ip publicPort privatePort type }} }}
                }}
            }}""")
            rt = d["pod"].get("runtime")
            if rt and rt.get("ports"):
                ssh_ports = [p for p in rt["ports"] if p["privatePort"] == 22]
                if ssh_ports:
                    ssh_host = ssh_ports[0]["ip"]
                    ssh_port = ssh_ports[0]["publicPort"]
                    break
            _time.sleep(10)
            if attempt % 3 == 0 and attempt > 0:
                _log_runpod(f"gpu-batch: still waiting for SSH port (attempt {attempt}/30) pod={pod_id[:8]}…")

        if not ssh_host:
            raise RuntimeError("Pod SSH port not available after 5 minutes")

        _log_runpod(f"gpu-batch: SSH endpoint root@{ssh_host}:{ssh_port}")
        update_progress(job_id, 10, 100, "Pod ready. Waiting for SSH daemon...")
        _time.sleep(60)

        # Find SSH key on server
        ssh_key = None
        for key_path in (
            os.path.expanduser("~/.ssh/runpod_ed25519"),
            os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go"),
            os.path.expanduser("~/.ssh/id_ed25519"),
            os.path.expanduser("~/.ssh/id_rsa"),
        ):
            if os.path.exists(key_path):
                ssh_key = key_path
                break
        if not ssh_key:
            raise RuntimeError("No SSH key found (~/.ssh/runpod_ed25519, ~/.runpod/ssh/ or ~/.ssh/)")

        _log_runpod(f"gpu-batch: using SSH identity {ssh_key}")

        # 3. Test SSH connection
        for attempt in range(20):
            r = _ssh_cmd(ssh_host, ssh_port, ssh_key, "echo SSH_OK", timeout=60, pod_id=pod_id, pod_host_id=pod_host_id)
            if "SSH_OK" in (r.stdout or ""):
                _log_runpod(f"gpu-batch: SSH OK (attempt {attempt + 1})")
                break
            out_h = (r.stdout or "").strip()[:100]
            err_h = (r.stderr or "").strip()[:160]
            _log_runpod(f"gpu-batch: SSH probe failed rc={r.returncode} out={out_h!r} err={err_h!r}")
            _time.sleep(15)
        else:
            raise RuntimeError(f"SSH connection failed after 20 attempts: {r.stderr[:200]}")

        update_progress(job_id, 15, 100, "SSH connected. Setting up environment...")

        # 4. Clone repo and bootstrap
        r = _ssh_cmd(ssh_host, ssh_port, ssh_key,
                      f"ls /workspace/chhat-project/train.py 2>/dev/null && echo REPO_EXISTS || echo NEED_CLONE",
                      timeout=30, pod_id=pod_id, pod_host_id=pod_host_id)
        if "NEED_CLONE" in r.stdout:
            _log_runpod("gpu-batch: cloning repo + bootstrap on pod (may take several minutes)")
            update_progress(job_id, 18, 100, "Cloning repository and installing dependencies...")
            r = _ssh_cmd(ssh_host, ssh_port, ssh_key,
                          f"cd /workspace && git clone --depth 1 {RUNPOD_REPO} && cd chhat-project && bash runpod/bootstrap_training_pod.sh",
                          timeout=1200, pod_id=pod_id, pod_host_id=pod_host_id)
            if r.returncode != 0:
                raise RuntimeError(f"Bootstrap failed: {r.stdout[-500:]}")
            _log_runpod("gpu-batch: bootstrap finished")
        else:
            _log_runpod("gpu-batch: repo exists on pod; git pull")
            _ssh_cmd(ssh_host, ssh_port, ssh_key, "cd /workspace/chhat-project && git pull", timeout=60, pod_id=pod_id, pod_host_id=pod_host_id)

        update_progress(job_id, 25, 100, "Uploading CSV...")
        _log_runpod(f"gpu-batch: uploading CSV ({csv_path.name})")

        # 5. Upload CSV
        r = _scp_to(ssh_host, ssh_port, ssh_key, str(csv_path),
                     "/workspace/chhat-project/backend/uploads/", timeout=120, pod_id=pod_id, pod_host_id=pod_host_id)
        if r.returncode != 0:
            raise RuntimeError(f"CSV upload failed: {r.stderr[:200]}")

        # 6. Kill zombie GPU processes and verify CUDA before pipeline
        _log_runpod("gpu-batch: clearing zombie GPU processes and verifying CUDA…")
        _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | "
            "xargs -r -I{} sh -c 'ps -p {} >/dev/null 2>&1 || kill -9 {} 2>/dev/null'; "
            "sleep 2; nvidia-smi",
            timeout=30, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        cuda_chk = _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "cd /workspace/chhat-project && source .venv/bin/activate && "
            "python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available'; "
            "print(f'CUDA OK device={torch.cuda.get_device_name(0)}')\"",
            timeout=60, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if "CUDA OK" not in (cuda_chk.stdout or ""):
            _log_runpod(f"gpu-batch: WARNING CUDA check failed: {(cuda_chk.stdout or cuda_chk.stderr or '')[:300]}")
            raise RuntimeError("CUDA not available on pod — aborting to avoid CPU pipeline")
        _log_runpod(f"gpu-batch: {(cuda_chk.stdout or '').strip()}")

        # Run pipeline on pod
        update_progress(job_id, 30, 100, "Running detection pipeline on GPU...")
        _log_runpod("gpu-batch: starting run_pipeline on pod (long-running; up to 2h timeout)")
        remote_csv = f"/workspace/chhat-project/backend/uploads/{csv_path.name}"
        remote_out = f"/workspace/chhat-project/backend/uploads/{csv_path.stem}_results.csv"

        r = _ssh_cmd(ssh_host, ssh_port, ssh_key,
                      f"cd /workspace/chhat-project && source .venv/bin/activate && "
                      f"CUDA_VISIBLE_DEVICES=0 "
                      f"python -c \""
                      f"from backend.pipeline import run_pipeline; "
                      f"out = run_pipeline('{remote_csv}'); "
                      f"print(f'RESULT_PATH={{out}}')"
                      f"\"",
                      timeout=7200, pod_id=pod_id, pod_host_id=pod_host_id)  # 2 hour timeout for large batches

        if r.returncode != 0:
            raise RuntimeError(f"Pipeline failed on GPU: {r.stdout[-500:]}")
        _log_runpod(f"gpu-batch: remote pipeline finished rc=0 tail_stdout_chars={len(r.stdout or '')}")

        # Parse actual output path
        result_line = [l for l in r.stdout.splitlines() if "RESULT_PATH=" in l]
        if result_line:
            remote_result = result_line[-1].split("RESULT_PATH=")[1].strip()
        else:
            remote_result = remote_out

        update_progress(job_id, 90, 100, "Downloading results...")

        # 7. Download results
        job_out_path = RESULTS_DIR / f"{job_id}_{csv_path.stem}_results.csv"
        r = _scp_from(ssh_host, ssh_port, ssh_key, remote_result, str(job_out_path), timeout=120, pod_id=pod_id, pod_host_id=pod_host_id)
        if r.returncode != 0:
            raise RuntimeError(f"Result download failed: {r.stderr[:200]}")

        _save_result_meta(job_id, job_out_path)

        try:
            import pandas as _pd
            result_df = _pd.read_csv(job_out_path)
            row_count = len(result_df)
        except Exception:
            row_count = None

        _update_batch_history(job_id, {
            "status": "done",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "rows": row_count,
            "result_file": job_out_path.name,
        })
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = str(job_out_path)

        update_progress(job_id, 100, 100, f"GPU processing complete! ({row_count or '?'} rows)")
        _log_runpod(f"gpu-batch: complete job={job_id[:8]}… rows={row_count}")

    except Exception:
        err = traceback.format_exc()
        _log_runpod(f"gpu-batch: ERROR job={job_id[:8]}… {err[:400]}")
        _update_batch_history(job_id, {
            "status": "error",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "error": str(err)[:500],
        })
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = err
    finally:
        # Always terminate pod to avoid charges
        if pod_id:
            try:
                _runpod_gql(api_key, f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
                _log_runpod(f"gpu-batch: pod {pod_id} terminated")
            except Exception as exc:
                _log_runpod(f"gpu-batch: WARNING failed to terminate pod {pod_id}: {exc}")


def run_dinov2_finetune_gpu_job(
    job_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    unfreeze_layers: int,
):
    """Run finetune_dinov2.py on a RunPod GPU (uploads references, downloads weights)."""
    import time as time_module

    api_key = _get_runpod_api_key()
    if not api_key:
        err = "RUNPOD_API_KEY not set. Add it to .env on the server (same as batch GPU)."
        _log_runpod(f"dino-gpu job={job_id[:8]}… FAILED: {err}")
        with _training_lock:
            _training_jobs[job_id]["status"] = "error"
            _training_jobs[job_id]["error"] = err
            _training_jobs[job_id]["end_time"] = _now_iso()
        _update_training_history(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
        _update_model_registry(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
        return

    _log_runpod(
        f"dino-gpu job={job_id[:8]}… start epochs={epochs} batch={batch_size} lr={lr} "
        f"unfreeze={unfreeze_layers} key={_mask_secret_hint(api_key)}",
    )
    refs_tar = Path(f"/tmp/references_dino_{job_id}.tar.gz")
    pod_id = None
    try:
        with _training_lock:
            _training_jobs[job_id].update({
                "status": "running",
                "progress": {
                    "epoch": 0,
                    "total_epochs": epochs,
                    "status": "starting",
                    "note": "Provisioning RunPod GPU for DINOv2 fine-tune",
                },
                "error": None,
                "last_update": _now_iso(),
            })
        _update_model_registry(job_id, {"status": "running", "last_update": _now_iso()})

        _log_runpod(f"dino-gpu: tarring backend/references -> {refs_tar.name}")
        r_tar = _tar_backend_references_for_runpod(refs_tar)
        if r_tar.returncode != 0:
            raise RuntimeError(_fmt_subprocess_fail(r_tar, "Packaging references failed"))
        sz = refs_tar.stat().st_size if refs_tar.exists() else 0
        _log_runpod(f"dino-gpu: references archive OK size_bytes={sz}")

        _log_runpod("dino-gpu: creating RunPod pod…")
        gpu_candidates = _classifier_gpu_candidates()
        cloud_types = _classifier_cloud_types()
        vol_gb = _classifier_volume_gb()
        deploy_rounds, deploy_pause = _classifier_deploy_retry_settings()
        pod = None
        gpu_id = gpu_candidates[0] if gpu_candidates else RUNPOD_GPU_ID
        used_cloud = None
        for round_idx in range(deploy_rounds):
            if round_idx > 0:
                _log_runpod(
                    f"dino-gpu: capacity retry {round_idx + 1}/{deploy_rounds} "
                    f"(waiting {deploy_pause}s)…",
                )
                time_module.sleep(deploy_pause)
            for try_cloud in cloud_types:
                for try_gpu in gpu_candidates:
                    try:
                        _log_runpod(f"dino-gpu: try round={round_idx + 1} cloud={try_cloud} gpu={try_gpu}…")
                        pod = _runpod_gql(api_key, """
                            mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
                                podFindAndDeployOnDemand(input: $input) { id costPerHr machine { podHostId } }
                            }""", {"input": {
                                "name": f"dino-{job_id[:8]}",
                                "templateId": RUNPOD_TEMPLATE,
                                "gpuTypeId": try_gpu,
                                "cloudType": try_cloud,
                                "containerDiskInGb": 20,
                                "volumeInGb": vol_gb,
                                "volumeMountPath": "/workspace",
                                "gpuCount": 1,
                                "ports": "22/tcp",
                            }})["podFindAndDeployOnDemand"]
                        gpu_id = try_gpu
                        used_cloud = try_cloud
                        break
                    except RuntimeError as exc:
                        _log_runpod(f"dino-gpu: cloud={try_cloud} gpu={try_gpu} failed: {exc}")
                        continue
                if pod is not None:
                    break
            if pod is not None:
                break
        if pod is None:
            raise RuntimeError(
                f"No RunPod capacity for DINOv2 after {deploy_rounds} round(s). "
                "Try later or set RUNPOD_CLASSIFIER_GPU_ID to an available GPU.",
            )
        if used_cloud:
            _log_runpod(f"dino-gpu: using cloud={used_cloud} gpu={gpu_id}")
        pod_id = pod["id"]
        pod_host_id = (pod.get("machine") or {}).get("podHostId", "")
        if not pod_host_id:
            _log_runpod("dino-gpu: WARNING podHostId empty -- SSH proxy disabled, falling back to direct IP")
        _log_runpod(f"dino-gpu: pod id={pod_id} host={pod_host_id} cost~=${pod.get('costPerHr', 0)}/hr gpu={gpu_id}")

        ssh_host = None
        ssh_port = None
        for attempt in range(30):
            d = _runpod_gql(api_key, f"""query {{
                pod(input: {{ podId: "{pod_id}" }}) {{
                    runtime {{ uptimeInSeconds ports {{ ip publicPort privatePort type }} }}
                }}
            }}""")
            rt = d["pod"].get("runtime")
            if rt and rt.get("ports"):
                ssh_ports = [p for p in rt["ports"] if p["privatePort"] == 22]
                if ssh_ports:
                    ssh_host = ssh_ports[0]["ip"]
                    ssh_port = ssh_ports[0]["publicPort"]
                    break
            time_module.sleep(10)
            if attempt % 3 == 0 and attempt > 0:
                _log_runpod(f"dino-gpu: waiting for SSH port ({attempt}/30) pod={pod_id[:8]}…")
        if not ssh_host:
            raise RuntimeError("RunPod SSH port not available after ~5 minutes")

        _log_runpod(f"dino-gpu: SSH root@{ssh_host}:{ssh_port} (sleep 60s for sshd)")
        time_module.sleep(60)

        ssh_key = None
        for key_path in (
            os.path.expanduser("~/.ssh/runpod_ed25519"),
            os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go"),
            os.path.expanduser("~/.ssh/id_ed25519"),
            os.path.expanduser("~/.ssh/id_rsa"),
        ):
            if os.path.exists(key_path):
                ssh_key = key_path
                break
        if not ssh_key:
            raise RuntimeError("No SSH private key found (~/.ssh/runpod_ed25519, ~/.runpod/ssh/ or ~/.ssh/)")

        _log_runpod(f"dino-gpu: SSH identity {ssh_key}")
        last_chk = None
        for attempt in range(20):
            last_chk = _ssh_cmd(ssh_host, ssh_port, ssh_key, "echo SSH_OK", timeout=60, pod_id=pod_id, pod_host_id=pod_host_id)
            if "SSH_OK" in (last_chk.stdout or ""):
                _log_runpod(f"dino-gpu: SSH OK (attempt {attempt + 1})")
                break
            o = (last_chk.stdout or "").strip()[:100]
            e = (last_chk.stderr or "").strip()[:160]
            _log_runpod(f"dino-gpu: SSH probe rc={last_chk.returncode} out={o!r} err={e!r}")
            time_module.sleep(15)
        else:
            raise RuntimeError(f"SSH to pod failed after 20 attempts: {(last_chk.stderr or last_chk.stdout or '')[:300]}")

        chk = _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "test -f /workspace/chhat-project/finetune_dinov2.py && echo OK || echo MISSING",
            timeout=40, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if "MISSING" in (chk.stdout or ""):
            _log_runpod("dino-gpu: repo missing on pod — clone + bootstrap (long)")
            # Ephemeral RunPod pod only (not production server):
            br = _ssh_cmd(
                ssh_host, ssh_port, ssh_key,
                f"cd /workspace && rm -rf chhat-project && git clone --depth 1 {RUNPOD_REPO} chhat-project "
                f"&& cd chhat-project && bash runpod/bootstrap_training_pod.sh",
                timeout=1200, pod_id=pod_id, pod_host_id=pod_host_id,
            )
            if br.returncode != 0:
                raise RuntimeError(f"Pod bootstrap failed: {(br.stdout or '')[-800:]}")
            _log_runpod("dino-gpu: bootstrap complete")
        else:
            _log_runpod("dino-gpu: repo present on pod")

        _log_runpod("dino-gpu: upload references archive to pod…")
        up = _scp_to(
            ssh_host, ssh_port, ssh_key, str(refs_tar),
            "/workspace/chhat-project/backend/references_upload.tar.gz",
            timeout=600, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if up.returncode != 0:
            raise RuntimeError(f"references upload to pod failed: {(up.stderr or '')[:400]}")

        # Ephemeral RunPod pod filesystem only:
        un = _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "cd /workspace/chhat-project/backend && rm -rf references && mkdir -p references && "
            "tar xzf references_upload.tar.gz && rm -f references_upload.tar.gz",
            timeout=180, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if un.returncode != 0:
            raise RuntimeError(f"Unpack references on pod failed: {(un.stdout or un.stderr or '')[-600:]}")
        _log_runpod("dino-gpu: references uploaded + extracted OK")

        # Kill zombie GPU processes and verify CUDA before training
        _log_runpod("dino-gpu: clearing zombie GPU processes and verifying CUDA…")
        _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | "
            "xargs -r -I{} sh -c 'ps -p {} >/dev/null 2>&1 || kill -9 {} 2>/dev/null'; "
            "sleep 2; nvidia-smi",
            timeout=30, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        cuda_chk = _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "cd /workspace/chhat-project && source .venv/bin/activate && "
            "python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available'; "
            "print(f'CUDA OK device={torch.cuda.get_device_name(0)}')\"",
            timeout=60, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if "CUDA OK" not in (cuda_chk.stdout or ""):
            _log_runpod(f"dino-gpu: WARNING CUDA check failed: {(cuda_chk.stdout or cuda_chk.stderr or '')[:300]}")
            raise RuntimeError("CUDA not available on pod — aborting to avoid CPU training")
        _log_runpod(f"dino-gpu: {(cuda_chk.stdout or '').strip()}")

        progress_remote = "/tmp/dino_progress.json"
        ft_cmd = (
            f"cd /workspace/chhat-project && source .venv/bin/activate && "
            f"CUDA_VISIBLE_DEVICES=0 "
            f"python finetune_dinov2.py --epochs {epochs} --batch-size {batch_size} --lr {lr} "
            f"--unfreeze-layers {unfreeze_layers} --progress-file {progress_remote}"
        )

        stop_poll = threading.Event()
        poll_state = {"last_epoch": -1}

        def _poll_remote_progress():
            while not stop_poll.wait(12):
                try:
                    pr = _ssh_cmd(
                        ssh_host, ssh_port, ssh_key,
                        f"cat {progress_remote} 2>/dev/null || true",
                        timeout=25, pod_id=pod_id, pod_host_id=pod_host_id,
                    )
                    txt = (pr.stdout or "").strip()
                    if txt.startswith("{"):
                        data = json.loads(txt)
                        with _training_lock:
                            _training_jobs[job_id]["progress"] = data
                            _training_jobs[job_id]["last_update"] = _now_iso()
                        _update_model_registry(
                            job_id,
                            {
                                "status": "running",
                                "progress": data,
                                "last_update": _now_iso(),
                                **_metrics_from_progress(data),
                            },
                        )
                        ep = int(data.get("epoch", 0) or 0)
                        if ep != poll_state["last_epoch"]:
                            poll_state["last_epoch"] = ep
                            st = data.get("status", "")
                            va = data.get("val_acc", 0)
                            _log_runpod(
                                f"dino-gpu: remote progress epoch={ep}/{epochs} status={st} val_acc={va}",
                            )
                except Exception:
                    pass

        poller = threading.Thread(target=_poll_remote_progress, daemon=True)
        poller.start()

        _log_runpod("dino-gpu: starting finetune_dinov2.py on pod (up to 8h) — progress polled every ~12s")
        r_ft = _ssh_cmd(ssh_host, ssh_port, ssh_key, ft_cmd, timeout=8 * 3600, pod_id=pod_id, pod_host_id=pod_host_id)
        stop_poll.set()
        poller.join(timeout=8)

        if r_ft.returncode != 0:
            _log_runpod(f"dino-gpu: finetune exited rc={r_ft.returncode}")
            raise RuntimeError(f"DINOv2 fine-tune on pod failed:\n{(r_ft.stdout or '')[-3500:]}")
        _log_runpod("dino-gpu: finetune subprocess on pod finished rc=0")

        out_dir = _DATA_ROOT / "classifier_model"
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("dinov2_finetuned_head.pth", "dinov2_finetuned_full.pth", "class_mapping.json"):
            remote_p = f"/workspace/chhat-project/backend/classifier_model/{fname}"
            local_p = str(out_dir / fname)
            _log_runpod(f"dino-gpu: downloading {fname} from pod…")
            dl = _scp_from(ssh_host, ssh_port, ssh_key, remote_p, local_p, timeout=600, pod_id=pod_id, pod_host_id=pod_host_id)
            if dl.returncode != 0:
                raise RuntimeError(f"Download {fname} from pod failed: {(dl.stderr or '')[:400]}")
            _log_runpod(f"dino-gpu: saved {local_p}")

        with _training_lock:
            completed_version = _training_jobs[job_id].get("version", DEFAULT_TRAINING_VERSION)
            _training_jobs[job_id]["status"] = "done"
            _training_jobs[job_id]["end_time"] = _now_iso()
            _training_jobs[job_id]["progress"] = {
                "epoch": epochs,
                "total_epochs": epochs,
                "status": "complete",
                "note": "Weights saved under backend/classifier_model/",
            }

        try:
            reload_dino()
            reload_classifiers()
            _log_runpod("dino-gpu: reload_dino + reload_classifiers OK")
        except Exception as exc:
            _log_runpod(f"dino-gpu: WARNING reload after fine-tune: {exc}")

        next_version = _mark_training_completed(str(completed_version))
        _update_training_history(job_id, {
            "status": "done",
            "end_time": _now_iso(),
            "next_version": next_version,
        })
        _update_model_registry(job_id, {
            "status": "done",
            "end_time": _now_iso(),
            "next_version": next_version,
            **_metrics_from_progress(_training_jobs[job_id].get("progress", {})),
        })
        _log_runpod(f"dino-gpu: job DONE job={job_id[:8]}… next_version={next_version}")

    except Exception:
        err = traceback.format_exc()
        _log_runpod(f"dino-gpu: ERROR job={job_id[:8]}…\n{err[:800]}")
        with _training_lock:
            _training_jobs[job_id]["status"] = "error"
            _training_jobs[job_id]["error"] = err
            _training_jobs[job_id]["end_time"] = _now_iso()
        _update_training_history(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
        _update_model_registry(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
    finally:
        refs_tar.unlink(missing_ok=True)
        if pod_id:
            try:
                _runpod_gql(api_key, f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
                _log_runpod(f"dino-gpu: pod terminated id={pod_id}")
            except Exception as exc:
                _log_runpod(f"dino-gpu: WARNING pod terminate failed id={pod_id}: {exc}")


def run_classifier_training_runpod_job(
    job_id: str,
    epochs: int,
    batch_size: int,
    embed_batch_size: int,
    lr: float,
    packaging_type: str,
):
    """Run brand_classifier.py on a RunPod GPU (uploads references, downloads weights for one packaging type)."""
    import time as time_module

    api_key = _get_runpod_api_key()
    if not api_key:
        err = "RUNPOD_API_KEY not set. Add it to .env on the server."
        _log_runpod(f"classifier-gpu job={job_id[:8]}… FAILED: {err}")
        with _training_lock:
            _training_jobs[job_id]["status"] = "error"
            _training_jobs[job_id]["error"] = err
            _training_jobs[job_id]["end_time"] = _now_iso()
        _update_training_history(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
        _update_model_registry(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
        return

    gpu_id = _get_runpod_classifier_gpu_id()
    _log_runpod(
        f"classifier-gpu job={job_id[:8]}… start epochs={epochs} batch={batch_size} "
        f"embed_batch={embed_batch_size} lr={lr} packaging={packaging_type} gpu={gpu_id} "
        f"key={_mask_secret_hint(api_key)}",
    )
    refs_tar = Path(f"/tmp/references_classifier_{job_id}.tar.gz")
    pod_id = None
    try:
        with _training_lock:
            _training_jobs[job_id].update({
                "status": "running",
                "progress": {
                    "epoch": 0,
                    "total_epochs": epochs,
                    "status": "starting",
                    "note": "Provisioning RunPod GPU for brand classifier",
                },
                "error": None,
                "last_update": _now_iso(),
            })
        _update_model_registry(job_id, {"status": "running", "last_update": _now_iso()})

        _log_runpod(f"classifier-gpu: tarring backend/references -> {refs_tar.name}")
        r_tar = _tar_backend_references_for_runpod(refs_tar)
        if r_tar.returncode != 0:
            raise RuntimeError(_fmt_subprocess_fail(r_tar, "Packaging references failed"))
        sz = refs_tar.stat().st_size if refs_tar.exists() else 0
        _log_runpod(f"classifier-gpu: references archive OK size_bytes={sz}")

        _log_runpod("classifier-gpu: creating RunPod pod…")
        gpu_candidates = _classifier_gpu_candidates()
        cloud_types = _classifier_cloud_types()
        vol_gb = _classifier_volume_gb()
        deploy_rounds, deploy_pause = _classifier_deploy_retry_settings()
        pod = None
        used_cloud = None
        for round_idx in range(deploy_rounds):
            if round_idx > 0:
                _log_runpod(
                    f"classifier-gpu: capacity retry {round_idx + 1}/{deploy_rounds} "
                    f"(waiting {deploy_pause}s, then retry all cloud×GPU combos)…",
                )
                time_module.sleep(deploy_pause)
            for try_cloud in cloud_types:
                for try_gpu in gpu_candidates:
                    try:
                        _log_runpod(
                            f"classifier-gpu: try round={round_idx + 1} cloud={try_cloud} "
                            f"gpu={try_gpu} volume={vol_gb}Gi…",
                        )
                        pod = _runpod_gql(api_key, """
                            mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
                                podFindAndDeployOnDemand(input: $input) { id costPerHr machine { podHostId } }
                            }""", {"input": {
                                "name": f"cls-{job_id[:8]}",
                                "templateId": RUNPOD_TEMPLATE,
                                "gpuTypeId": try_gpu,
                                "cloudType": try_cloud,
                                "containerDiskInGb": 20,
                                "volumeInGb": vol_gb,
                                "volumeMountPath": "/workspace",
                                "gpuCount": 1,
                                "ports": "22/tcp",
                            }})["podFindAndDeployOnDemand"]
                        gpu_id = try_gpu
                        used_cloud = try_cloud
                        break
                    except RuntimeError as exc:
                        _log_runpod(f"classifier-gpu: cloud={try_cloud} gpu={try_gpu} failed: {exc}")
                        continue
                if pod is not None:
                    break
            if pod is not None:
                break
        if pod is None:
            raise RuntimeError(
                f"No RunPod capacity for classifier after {deploy_rounds} round(s) × "
                f"clouds={cloud_types} gpus={len(gpu_candidates)} types volume={vol_gb}Gi. "
                "Try later, reduce RUNPOD_CLASSIFIER_VOLUME_GB, set RUNPOD_CLASSIFIER_CLOUD_TYPE=SECURE, "
                "or set RUNPOD_CLASSIFIER_GPU_ID to a type shown available in the RunPod UI.",
            )
        if used_cloud:
            _log_runpod(f"classifier-gpu: using cloud={used_cloud} gpu={gpu_id} volume={vol_gb}Gi")
        pod_id = pod["id"]
        pod_host_id = (pod.get("machine") or {}).get("podHostId", "")
        if not pod_host_id:
            _log_runpod("classifier-gpu: WARNING podHostId empty -- SSH proxy disabled, falling back to direct IP")
        with _training_lock:
            _training_jobs[job_id]["runpod_pod_id"] = pod_id
        _log_runpod(
            f"classifier-gpu: pod id={pod_id} cost~${pod.get('costPerHr', 0)}/hr "
            f"gpu={gpu_id}",
        )

        ssh_host = None
        ssh_port = None
        for attempt in range(30):
            d = _runpod_gql(api_key, f"""query {{
                pod(input: {{ podId: "{pod_id}" }}) {{
                    runtime {{ uptimeInSeconds ports {{ ip publicPort privatePort type }} }}
                }}
            }}""")
            rt = d["pod"].get("runtime")
            if rt and rt.get("ports"):
                ssh_ports = [p for p in rt["ports"] if p["privatePort"] == 22]
                if ssh_ports:
                    ssh_host = ssh_ports[0]["ip"]
                    ssh_port = ssh_ports[0]["publicPort"]
                    break
            time_module.sleep(10)
            if attempt % 3 == 0 and attempt > 0:
                _log_runpod(f"classifier-gpu: waiting for SSH port ({attempt}/30) pod={pod_id[:8]}…")
        if not ssh_host:
            raise RuntimeError("RunPod SSH port not available after ~5 minutes")

        _log_runpod(f"classifier-gpu: SSH root@{ssh_host}:{ssh_port} (sleep 60s for sshd)")
        time_module.sleep(60)

        ssh_key = None
        for key_path in (
            os.path.expanduser("~/.ssh/runpod_ed25519"),
            os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go"),
            os.path.expanduser("~/.ssh/id_ed25519"),
            os.path.expanduser("~/.ssh/id_rsa"),
        ):
            if os.path.exists(key_path):
                ssh_key = key_path
                break
        if not ssh_key:
            raise RuntimeError("No SSH private key found (~/.ssh/runpod_ed25519, ~/.runpod/ssh/ or ~/.ssh/)")

        _log_runpod(f"classifier-gpu: SSH identity {ssh_key}")
        last_chk = None
        for attempt in range(20):
            last_chk = _ssh_cmd(ssh_host, ssh_port, ssh_key, "echo SSH_OK", timeout=60, pod_id=pod_id, pod_host_id=pod_host_id)
            if "SSH_OK" in (last_chk.stdout or ""):
                _log_runpod(f"classifier-gpu: SSH OK (attempt {attempt + 1})")
                break
            o = (last_chk.stdout or "").strip()[:100]
            e = (last_chk.stderr or "").strip()[:160]
            _log_runpod(f"classifier-gpu: SSH probe rc={last_chk.returncode} out={o!r} err={e!r}")
            time_module.sleep(15)
        else:
            raise RuntimeError(f"SSH to pod failed after 20 attempts (~5min): {(last_chk.stderr or last_chk.stdout or '')[:300]}")

        chk = _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "test -f /workspace/chhat-project/brand_classifier.py && echo OK || echo MISSING",
            timeout=40, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if "MISSING" in (chk.stdout or ""):
            _log_runpod("classifier-gpu: repo missing on pod — clone + bootstrap (long)")
            br = _ssh_cmd(
                ssh_host, ssh_port, ssh_key,
                f"cd /workspace && rm -rf chhat-project && git clone --depth 1 {RUNPOD_REPO} chhat-project "
                f"&& cd chhat-project && bash runpod/bootstrap_training_pod.sh",
                timeout=1200, pod_id=pod_id, pod_host_id=pod_host_id,
            )
            if br.returncode != 0:
                raise RuntimeError(f"Pod bootstrap failed: {(br.stdout or '')[-800:]}")
            _log_runpod("classifier-gpu: bootstrap complete")
        else:
            _log_runpod("classifier-gpu: repo present on pod")

        _log_runpod("classifier-gpu: upload references archive to pod…")
        up = _scp_to(
            ssh_host, ssh_port, ssh_key, str(refs_tar),
            "/workspace/chhat-project/backend/references_upload.tar.gz",
            timeout=600, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if up.returncode != 0:
            raise RuntimeError(f"references upload to pod failed: {(up.stderr or '')[:400]}")

        un = _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "cd /workspace/chhat-project/backend && rm -rf references && mkdir -p references && "
            "tar xzf references_upload.tar.gz && rm -f references_upload.tar.gz",
            timeout=180, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if un.returncode != 0:
            raise RuntimeError(f"Unpack references on pod failed: {(un.stdout or un.stderr or '')[-600:]}")
        _log_runpod("classifier-gpu: references uploaded + extracted OK")

        # Kill zombie GPU processes and verify CUDA before training
        _log_runpod("classifier-gpu: clearing zombie GPU processes and verifying CUDA…")
        _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | "
            "xargs -r -I{} sh -c 'ps -p {} >/dev/null 2>&1 || kill -9 {} 2>/dev/null'; "
            "sleep 2; nvidia-smi",
            timeout=30, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        cuda_chk = _ssh_cmd(
            ssh_host, ssh_port, ssh_key,
            "cd /workspace/chhat-project && source .venv/bin/activate && "
            "python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available'; "
            "print(f'CUDA OK device={torch.cuda.get_device_name(0)}')\"",
            timeout=60, pod_id=pod_id, pod_host_id=pod_host_id,
        )
        if "CUDA OK" not in (cuda_chk.stdout or ""):
            _log_runpod(f"classifier-gpu: WARNING CUDA check failed: {(cuda_chk.stdout or cuda_chk.stderr or '')[:300]}")
            raise RuntimeError("CUDA not available on pod — aborting to avoid CPU training")
        _log_runpod(f"classifier-gpu: {(cuda_chk.stdout or '').strip()}")

        progress_remote = f"/tmp/classifier_progress_{job_id}.json"
        bc_cmd = (
            f"cd /workspace/chhat-project && source .venv/bin/activate && "
            f"CUDA_VISIBLE_DEVICES=0 "
            f"python brand_classifier.py --epochs {epochs} --batch-size {batch_size} "
            f"--embed-batch-size {embed_batch_size} --lr {lr} "
            f"--packaging-type {packaging_type} --progress-file {progress_remote}"
        )

        stop_poll = threading.Event()
        poll_state = {"last_epoch": -1}

        def _poll_remote_progress():
            while not stop_poll.wait(12):
                try:
                    pr = _ssh_cmd(
                        ssh_host, ssh_port, ssh_key,
                        f"cat {progress_remote} 2>/dev/null || true",
                        timeout=25, pod_id=pod_id, pod_host_id=pod_host_id,
                    )
                    txt = (pr.stdout or "").strip()
                    if txt.startswith("{"):
                        data = json.loads(txt)
                        with _training_lock:
                            _training_jobs[job_id]["progress"] = data
                            _training_jobs[job_id]["last_update"] = _now_iso()
                        _update_model_registry(
                            job_id,
                            {
                                "status": "running",
                                "progress": data,
                                "last_update": _now_iso(),
                                **_metrics_from_progress(data),
                            },
                        )
                        ep = int(data.get("epoch", 0) or 0)
                        if ep != poll_state["last_epoch"]:
                            poll_state["last_epoch"] = ep
                            st = data.get("status", "")
                            va = data.get("val_acc", 0)
                            _log_runpod(
                                f"classifier-gpu: remote progress epoch={ep}/{epochs} status={st} val_acc={va}",
                            )
                except Exception:
                    pass

        poller = threading.Thread(target=_poll_remote_progress, daemon=True)
        poller.start()

        _log_runpod("classifier-gpu: starting brand_classifier.py on pod (up to 8h) — progress polled ~12s")
        r_bc = _ssh_cmd(ssh_host, ssh_port, ssh_key, bc_cmd, timeout=8 * 3600, pod_id=pod_id, pod_host_id=pod_host_id)
        stop_poll.set()
        poller.join(timeout=8)

        if r_bc.returncode != 0:
            with _training_lock:
                if _training_jobs.get(job_id, {}).get("stop_requested"):
                    _training_jobs[job_id]["status"] = "stopped"
                    _training_jobs[job_id]["error"] = None
                    _training_jobs[job_id]["end_time"] = _now_iso()
                    _training_jobs[job_id]["progress"] = {
                        "status": "stopped",
                        "note": "Training stopped (RunPod pod terminated)",
                    }
            with _training_lock:
                if _training_jobs.get(job_id, {}).get("status") == "stopped":
                    _update_training_history(job_id, {"status": "stopped", "end_time": _now_iso()})
                    _update_model_registry(job_id, {"status": "stopped", "end_time": _now_iso()})
                    _log_runpod(f"classifier-gpu: stopped job={job_id[:8]}…")
                    return
            _log_runpod(f"classifier-gpu: brand_classifier exited rc={r_bc.returncode}")
            raise RuntimeError(f"Brand classifier on pod failed:\n{(r_bc.stdout or '')[-3500:]}")

        out_sub = _DATA_ROOT / "classifier_model" / packaging_type
        out_sub.mkdir(parents=True, exist_ok=True)
        for fname in ("best_classifier.pth", "classifier.pth", "class_mapping.json"):
            remote_p = f"/workspace/chhat-project/backend/classifier_model/{packaging_type}/{fname}"
            local_p = str(out_sub / fname)
            _log_runpod(f"classifier-gpu: downloading {packaging_type}/{fname} from pod…")
            dl = _scp_from(ssh_host, ssh_port, ssh_key, remote_p, local_p, timeout=600, pod_id=pod_id, pod_host_id=pod_host_id)
            if dl.returncode != 0:
                raise RuntimeError(f"Download {fname} from pod failed: {(dl.stderr or '')[:400]}")
            _log_runpod(f"classifier-gpu: saved {local_p}")

        with _training_lock:
            completed_version = _training_jobs[job_id].get("version", DEFAULT_TRAINING_VERSION)
            _training_jobs[job_id]["status"] = "done"
            _training_jobs[job_id]["end_time"] = _now_iso()
            _training_jobs[job_id]["progress"] = {
                "epoch": epochs,
                "total_epochs": epochs,
                "status": "complete",
                "note": f"Weights saved under backend/classifier_model/{packaging_type}/",
            }

        try:
            reload_classifiers()
            _log_runpod("classifier-gpu: reload_classifiers OK")
        except Exception as exc:
            _log_runpod(f"classifier-gpu: WARNING reload after classifier train: {exc}")

        next_version = _mark_training_completed(str(completed_version))
        with _training_lock:
            prog = dict(_training_jobs[job_id].get("progress", {}))
        _update_training_history(job_id, {
            "status": "done",
            "end_time": _now_iso(),
            "next_version": next_version,
        })
        _update_model_registry(job_id, {
            "status": "done",
            "end_time": _now_iso(),
            "next_version": next_version,
            **_metrics_from_progress(prog),
        })
        _log_runpod(f"classifier-gpu: job DONE job={job_id[:8]}… next_version={next_version}")

    except Exception:
        err = traceback.format_exc()
        with _training_lock:
            if _training_jobs.get(job_id, {}).get("stop_requested"):
                _training_jobs[job_id]["status"] = "stopped"
                _training_jobs[job_id]["error"] = None
                _training_jobs[job_id]["end_time"] = _now_iso()
                _training_jobs[job_id]["progress"] = {
                    "status": "stopped",
                    "note": "Training stopped (RunPod pod terminated)",
                }
            else:
                _training_jobs[job_id]["status"] = "error"
                _training_jobs[job_id]["error"] = err
                _training_jobs[job_id]["end_time"] = _now_iso()
        with _training_lock:
            st = _training_jobs.get(job_id, {}).get("status")
        if st == "stopped":
            _update_training_history(job_id, {"status": "stopped", "end_time": _now_iso()})
            _update_model_registry(job_id, {"status": "stopped", "end_time": _now_iso()})
        else:
            _update_training_history(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
            _update_model_registry(job_id, {"status": "error", "error": err, "end_time": _now_iso()})
            _log_runpod(f"classifier-gpu: ERROR job={job_id[:8]}…\n{err[:800]}")
    finally:
        refs_tar.unlink(missing_ok=True)
        if pod_id:
            try:
                _runpod_gql(api_key, f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
                _log_runpod(f"classifier-gpu: pod terminated id={pod_id}")
            except Exception as exc:
                _log_runpod(f"classifier-gpu: WARNING pod terminate failed id={pod_id}: {exc}")
        with _training_lock:
            _training_jobs[job_id].pop("runpod_pod_id", None)


@app.on_event("startup")
def startup_event():
    device = get_device()
    print(f"[startup] device={device}")
    rp = _get_runpod_api_key()
    if rp:
        _log_runpod(
            f"startup: RUNPOD_API_KEY available ({_mask_secret_hint(rp)}) — "
            f"GPU batch + RunPod DINO + classifier (gpu={_get_runpod_classifier_gpu_id()})",
        )
    else:
        _log_runpod("startup: RUNPOD_API_KEY not set — GPU batch and RunPod DINO will error until .env is configured")
    if not CLASSIFIER_WEIGHTS.exists() or not CLASS_MAPPING_JSON.exists():
        print("[startup] classifier not found -- run 'python brand_classifier.py' first")
        return
    try:
        load_classifier(device)
        print("[startup] classifier loaded")
    except Exception as exc:
        print(f"[startup] classifier load failed: {exc}")


@app.post("/build-index")
def build_index_endpoint():
    job_id = create_job()
    threading.Thread(target=run_build_index_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.post("/run-pipeline")
async def run_pipeline_endpoint(csv_file: UploadFile = File(...), use_gpu: str = Form("false")):
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / csv_file.filename
    data = await csv_file.read()
    save_path.write_bytes(data)

    job_id = create_job()
    gpu = use_gpu.lower() in ("true", "1", "yes")
    if gpu:
        threading.Thread(target=run_pipeline_gpu_job, args=(job_id, save_path), daemon=True).start()
    else:
        threading.Thread(target=run_pipeline_job, args=(job_id, save_path), daemon=True).start()
    return {"job_id": job_id, "gpu": gpu}


@app.get("/progress/{job_id}")
def progress_endpoint(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    def event_stream():
        while True:
            with jobs_lock:
                status = jobs[job_id]["status"]
                error = jobs[job_id]["error"]
            try:
                pct, message = job["queue"].get(timeout=0.5)
                yield f"data: {pct}|{message}\n\n"
            except queue.Empty:
                pass

            if status == "done":
                yield "data: DONE|\n\n"
                break
            if status == "error":
                yield f"data: ERROR|{error}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/download/{job_id}")
def download_endpoint(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    path = None
    if job and job.get("result") and job["result"] != "INDEX_REBUILT":
        path = Path(job["result"])
    if path is None or not path.exists():
        path = _load_result_meta(job_id)

    if path is None:
        raise HTTPException(status_code=404, detail="Result not available.")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing.")
    return FileResponse(path=str(path), filename=path.name, media_type="text/csv")


@app.get("/batch-history")
def batch_history(limit: int = 50):
    """List past batch processing jobs with status and download info."""
    rows = _load_batch_history()
    # Most recent first
    rows = list(reversed(rows))[:limit]
    return {"jobs": rows}


@app.get("/index-status")
def index_status():
    # Check pack/ subfolder first, then legacy flat path
    mapping_path = CLASS_MAPPING_JSON
    if not mapping_path.exists():
        legacy = CLASS_MAPPING_JSON.parent.parent / "class_mapping.json"
        if legacy.exists():
            mapping_path = legacy
        else:
            return {"exists": False, "brand_count": 0, "brands": [], "products": [], "total_images": 0, "num_labels": 0}
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    brands = list(mapping.get("label_to_idx", {}).keys())
    products = sorted(set(label_to_product(b) for b in brands))
    # Count reference images
    try:
        from brand_registry import audit_references
        audit = audit_references()
        total_images = audit.get("total_images", 0)
    except Exception:
        total_images = len(brands)
    return {"exists": True, "brand_count": len(brands), "brands": brands, "products": products,
            "total_images": total_images, "num_labels": len(brands)}


@app.post("/detect-single")
async def detect_single(image_file: UploadFile = File(...)):
    """Run detection on a single image. Returns per-box brand assignments for interactive UI."""
    from PIL import Image
    import base64
    try:
        from .pipeline import (
            embed_images_batch,
            classify_embeddings,
            _build_label_profiles,
            _run_ocr_on_image,
            _ocr_brand_scores_from_items,
            _aggregate_to_products,
            CLASSIFIER_TOP_K,
            OCR_ENABLED,
            OCR_FULLIMG_ENABLED,
            OCR_FALLBACK_THRESHOLD,
            OCR_FALLBACK_MARGIN,
            OCR_STRONG_THRESHOLD,
        )
    except ImportError:
        from pipeline import (
            embed_images_batch,
            classify_embeddings,
            _build_label_profiles,
            _run_ocr_on_image,
            _ocr_brand_scores_from_items,
            _aggregate_to_products,
            CLASSIFIER_TOP_K,
            OCR_ENABLED,
            OCR_FULLIMG_ENABLED,
            OCR_FALLBACK_THRESHOLD,
            OCR_FALLBACK_MARGIN,
            OCR_STRONG_THRESHOLD,
        )

    data = await image_file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    device = get_device()
    index, labels = load_index()
    processor, model = load_dino(device)
    rfdetr_model = load_rfdetr()
    img_w, img_h = pil_img.size
    label_profiles = _build_label_profiles(labels)

    # RF-DETR detection
    detections = rfdetr_model.predict(pil_img, threshold=RFDETR_CONF_THRESHOLD)
    crops = []
    boxes_data = []
    has_detections = detections is not None and len(detections) > 0

    if has_detections:
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img_w, x2 + pad_x)
            y2 = min(img_h, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(pil_img.crop((x1, y1, x2, y2)))
            pkg_type = "pack"
            if class_ids is not None and len(class_ids) > i:
                pkg_type = "box" if int(class_ids[i]) == 1 else "pack"
            boxes_data.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_conf": round(float(conf), 3),
                "packaging_type": pkg_type,
                "brands": [],
                "ocr_texts": [],
                "ocr_brand_scores": [],
            })
    else:
        crops.append(pil_img)
        boxes_data.append({
            "x1": 0, "y1": 0, "x2": img_w, "y2": img_h,
            "det_conf": 0.0,
            "is_full_image": True,
            "packaging_type": "pack",
            "brands": [],
            "ocr_texts": [],
            "ocr_brand_scores": [],
        })

    # Classify crops grouped by packaging type
    all_vecs = embed_images_batch(crops, processor, model, device)
    crop_pkg_types = [b.get("packaging_type", "pack") for b in boxes_data]

    type_indices: dict[str, list[int]] = {}
    for idx, pkg_type in enumerate(crop_pkg_types):
        type_indices.setdefault(pkg_type, []).append(idx)

    all_cls_results: list[list[tuple[str, float]]] = [[] for _ in crops]
    for pkg_type, indices in type_indices.items():
        try:
            type_vecs = all_vecs[indices]
            type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type=pkg_type)
            for local_idx, global_idx in enumerate(indices):
                all_cls_results[global_idx] = type_results[local_idx]
        except FileNotFoundError:
            # Fall back to pack classifier
            type_vecs = all_vecs[indices]
            type_results = classify_embeddings(type_vecs, device, top_k=CLASSIFIER_TOP_K, packaging_type="pack")
            for local_idx, global_idx in enumerate(indices):
                all_cls_results[global_idx] = type_results[local_idx]

    all_brand_scores = {}

    for crop_idx in range(len(crops)):
        crop_cls_ranked = all_cls_results[crop_idx]
        crop_cls = dict(crop_cls_ranked)
        top1 = float(crop_cls_ranked[0][1]) if crop_cls_ranked else 0.0
        top2 = float(crop_cls_ranked[1][1]) if len(crop_cls_ranked) > 1 else 0.0
        margin = top1 - top2
        should_run_ocr = OCR_ENABLED and (top1 < OCR_FALLBACK_THRESHOLD or margin < OCR_FALLBACK_MARGIN)

        # OCR fallback per crop
        ocr_items = _run_ocr_on_image(crops[crop_idx]) if should_run_ocr else []
        ocr_texts = []
        for item in ocr_items:
            try:
                text = str(item[1]).strip()
                conf = float(item[2])
            except Exception:
                continue
            if not text or text.lower() in ("nan",) or len(text) < 2:
                continue
            ocr_texts.append({"text": text, "confidence": round(conf, 3)})
        ocr_texts.sort(key=lambda x: x["confidence"], reverse=True)
        boxes_data[crop_idx]["ocr_texts"] = ocr_texts[:8]

        # OCR brand scores
        ocr_scores = _ocr_brand_scores_from_items(ocr_items, label_profiles) if ocr_items else {}
        ocr_product_scores = _aggregate_to_products(ocr_scores)
        boxes_data[crop_idx]["ocr_brand_scores"] = [
            {"brand": b, "confidence": round(s, 3)}
            for b, s in sorted(ocr_product_scores.items(), key=lambda x: -x[1])[:5]
        ]

        # Classifier + OCR fallback fusion for this crop
        label_profile_map = {p["label"]: p for p in label_profiles}
        ocr_families = {}
        for label, ocr_conf in ocr_scores.items():
            prof = label_profile_map.get(label, {})
            family = prof.get("brand", "")
            if family:
                ocr_families[family] = max(ocr_families.get(family, 0.0), ocr_conf)

        fused = {}
        for label, cls_conf in crop_cls.items():
            out_conf = float(cls_conf)
            if should_run_ocr:
                prof = label_profile_map.get(label, {})
                brand_family = prof.get("brand", "")
                ocr_fam = ocr_families.get(brand_family, 0.0) if brand_family else 0.0
                if ocr_fam >= OCR_STRONG_THRESHOLD:
                    out_conf = min(1.0, out_conf + ocr_fam * 0.25)
                elif ocr_fam > 0:
                    out_conf = min(1.0, out_conf + ocr_fam * 0.10)
            fused[label] = out_conf

        crop_products = _aggregate_to_products(fused)
        crop_brands = [{"brand": p, "confidence": round(c, 3)} for p, c in crop_products.items()]
        crop_brands.sort(key=lambda x: x["confidence"], reverse=True)
        boxes_data[crop_idx]["brands"] = crop_brands[:5]

        for label, conf in fused.items():
            if conf > all_brand_scores.get(label, 0.0):
                all_brand_scores[label] = conf

    # Encode image
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    all_product_scores = _aggregate_to_products(all_brand_scores)
    all_sorted = sorted(all_product_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "image_b64": img_b64,
        "image_width": img_w,
        "image_height": img_h,
        "boxes": boxes_data,
        "ocr_independent": [],
        "brands": [b for b, _ in all_sorted],
        "confidence": [round(c, 3) for _, c in all_sorted],
        "num_boxes": sum(1 for b in boxes_data if not b.get("is_full_image")),
    }


@app.post("/upload-coco")
async def upload_coco(coco_file: UploadFile = File(...)):
    """Upload a COCO JSON annotation file or ZIP export for RF-DETR training data."""
    filename = (coco_file.filename or "").strip()
    lower = filename.lower()
    if not (lower.endswith(".json") or lower.endswith(".zip")):
        raise HTTPException(status_code=400, detail="Only .json or .zip COCO files accepted.")

    DATASETS_DIR = _BACKEND_ROOT.parent / "datasets" / "cigarette_packs"
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    data = await coco_file.read()
    MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1 GB
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 1 GB)")
    if lower.endswith(".json"):
        try:
            coco_data = json.loads(data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON file.")

        n_images = len(coco_data.get("images", []))
        n_annotations = len(coco_data.get("annotations", []))
        save_path = DATASETS_DIR / Path(filename).name
        save_path.write_bytes(data)
        return {
            "status": "uploaded",
            "file_type": "json",
            "filename": filename,
            "images": n_images,
            "annotations": n_annotations,
            "saved_to": str(save_path),
        }

    try:
        zf = zipfile.ZipFile(BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")

    members = [m for m in zf.namelist() if not m.endswith("/")]
    ann_candidates = [m for m in members if m.lower().endswith("_annotations.coco.json")]
    if not ann_candidates:
        raise HTTPException(status_code=400, detail="ZIP must include _annotations.coco.json.")

    def _safe_parts(member_name: str) -> list[str]:
        return [p for p in member_name.replace("\\", "/").split("/") if p not in ("", ".", "..")]

    def _extract_member(member_name: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member_name) as src, out_path.open("wb") as dst:
            dst.write(src.read())

    target_splits = ("train", "valid", "test")
    split_ann_map: dict[str, str] = {}
    for ann in ann_candidates:
        parts = [p.lower() for p in _safe_parts(ann)]
        for split in target_splits:
            if split in parts:
                split_ann_map[split] = ann
    if not split_ann_map:
        split_ann_map["train"] = ann_candidates[0]

    extracted_images = 0
    extracted_annotations = 0
    extracted_files = 0
    saved_dirs: list[str] = []

    for split, ann_member in split_ann_map.items():
        split_dir = DATASETS_DIR / split
        saved_dirs.append(str(split_dir))
        ann_out = split_dir / "_annotations.coco.json"
        _extract_member(ann_member, ann_out)
        extracted_files += 1

        try:
            coco_data = json.loads(ann_out.read_text(encoding="utf-8"))
        except Exception:
            continue

        extracted_images += len(coco_data.get("images", []))
        extracted_annotations += len(coco_data.get("annotations", []))

        ann_parts = _safe_parts(ann_member)
        ann_prefix = "/".join(ann_parts[:-1])
        for img in coco_data.get("images", []):
            img_name = img.get("file_name", "")
            if not img_name:
                continue
            safe_img_name = "/".join(_safe_parts(img_name))
            if not safe_img_name:
                continue
            prefixed_candidate = f"{ann_prefix}/{safe_img_name}" if ann_prefix else safe_img_name
            if prefixed_candidate in members:
                _extract_member(prefixed_candidate, split_dir / safe_img_name)
                extracted_files += 1
            elif safe_img_name in members:
                _extract_member(safe_img_name, split_dir / safe_img_name)
                extracted_files += 1

    return {
        "status": "uploaded",
        "file_type": "zip",
        "filename": filename,
        "images": extracted_images,
        "annotations": extracted_annotations,
        "extracted_files": extracted_files,
        "splits": sorted(split_ann_map.keys()),
        "saved_to": saved_dirs,
    }


@app.post("/download-roboflow-coco")
def download_roboflow_coco(url: str = Form(...), clean: bool = Form(False)):
    """Download a Roboflow raw dataset URL and extract COCO files/images.

    If clean=True, deletes the entire datasets/cigarette_packs directory first (destructive).
    """
    from urllib.parse import parse_qs, urlparse
    import requests
    import shutil

    parsed = urlparse(url.strip())
    if "roboflow.com" not in parsed.netloc:
        raise HTTPException(status_code=400, detail="URL must be from roboflow.com")
    if "key" not in parse_qs(parsed.query):
        raise HTTPException(status_code=400, detail="Roboflow URL must include ?key=...")

    datasets_dir = _BACKEND_ROOT.parent / "datasets" / "cigarette_packs"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if clean and datasets_dir.exists():
        shutil.rmtree(datasets_dir)
        datasets_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url.strip(), timeout=120)
        resp.raise_for_status()
        payload = resp.content
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {exc}")

    # Reuse upload extraction path by treating payload as uploaded zip
    try:
        zf = zipfile.ZipFile(BytesIO(payload))
    except Exception:
        raise HTTPException(status_code=400, detail="Downloaded file is not a valid ZIP.")

    members = [m for m in zf.namelist() if not m.endswith("/")]
    ann_candidates = [m for m in members if m.lower().endswith("_annotations.coco.json")]
    if not ann_candidates:
        raise HTTPException(status_code=400, detail="ZIP must include _annotations.coco.json.")

    def _safe_parts(member_name: str) -> list[str]:
        return [p for p in member_name.replace("\\", "/").split("/") if p not in ("", ".", "..")]

    def _extract_member(member_name: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member_name) as src, out_path.open("wb") as dst:
            dst.write(src.read())

    target_splits = ("train", "valid", "test")
    split_ann_map: dict[str, str] = {}
    for ann in ann_candidates:
        parts = [p.lower() for p in _safe_parts(ann)]
        for split in target_splits:
            if split in parts:
                split_ann_map[split] = ann
    if not split_ann_map:
        split_ann_map["train"] = ann_candidates[0]

    extracted_images = 0
    extracted_annotations = 0
    extracted_files = 0
    saved_dirs: list[str] = []

    for split, ann_member in split_ann_map.items():
        split_dir = datasets_dir / split
        saved_dirs.append(str(split_dir))
        ann_out = split_dir / "_annotations.coco.json"
        _extract_member(ann_member, ann_out)
        extracted_files += 1
        try:
            coco_data = json.loads(ann_out.read_text(encoding="utf-8"))
        except Exception:
            continue

        extracted_images += len(coco_data.get("images", []))
        extracted_annotations += len(coco_data.get("annotations", []))

        ann_parts = _safe_parts(ann_member)
        ann_prefix = "/".join(ann_parts[:-1])
        for img in coco_data.get("images", []):
            img_name = img.get("file_name", "")
            if not img_name:
                continue
            safe_img_name = "/".join(_safe_parts(img_name))
            if not safe_img_name:
                continue
            prefixed_candidate = f"{ann_prefix}/{safe_img_name}" if ann_prefix else safe_img_name
            if prefixed_candidate in members:
                _extract_member(prefixed_candidate, split_dir / safe_img_name)
                extracted_files += 1
            elif safe_img_name in members:
                _extract_member(safe_img_name, split_dir / safe_img_name)
                extracted_files += 1

    return {
        "status": "downloaded",
        "source": "roboflow_url",
        "images": extracted_images,
        "annotations": extracted_annotations,
        "extracted_files": extracted_files,
        "splits": sorted(split_ann_map.keys()),
        "saved_to": saved_dirs,
    }


@app.post("/generate-crops")
async def generate_crops(image_file: UploadFile = File(...)):
    """Run RF-DETR on an image, classify each crop, and return for labeling."""
    from PIL import Image
    import base64
    try:
        from .pipeline import (
            embed_images_batch, classify_embeddings,
        )
        from .brand_registry import get_brand, resolve_internal_name
    except ImportError:
        from pipeline import (
            embed_images_batch, classify_embeddings,
        )
        from brand_registry import get_brand, resolve_internal_name

    data = await image_file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    device = get_device()
    rfdetr_model = load_rfdetr()
    detections = rfdetr_model.predict(pil_img, threshold=RFDETR_CONF_THRESHOLD)

    crop_images = []
    crop_meta = []
    if detections is not None and len(detections) > 0:
        width, height = pil_img.size
        class_ids = detections.class_id if hasattr(detections, "class_id") and detections.class_id is not None else None
        for i, (box, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = [int(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.05), int(bh * 0.05)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = pil_img.crop((x1, y1, x2, y2))
            crop_images.append(crop)
            pkg_type = "pack"
            if class_ids is not None and len(class_ids) > i:
                pkg_type = "box" if int(class_ids[i]) == 1 else "pack"
            crop_meta.append({"index": i, "w": x2 - x1, "h": y2 - y1, "conf": float(conf), "packaging_type": pkg_type})

    suggested_labels = []
    if crop_images:
        try:
            processor, model = load_dino(device)

            vecs = embed_images_batch(crop_images, processor, model, device)

            # Group by packaging type for classification
            type_indices: dict[str, list[int]] = {}
            for idx, meta in enumerate(crop_meta):
                pkg = meta["packaging_type"]
                type_indices.setdefault(pkg, []).append(idx)

            per_crop_results: list[tuple[str, float]] = [("unknown", 0.0)] * len(crop_images)
            for pkg_type, indices in type_indices.items():
                try:
                    type_vecs = vecs[indices]
                    cls_results = classify_embeddings(type_vecs, device, top_k=3, packaging_type=pkg_type)
                    for local_idx, global_idx in enumerate(indices):
                        if cls_results[local_idx]:
                            per_crop_results[global_idx] = cls_results[local_idx][0]
                except FileNotFoundError:
                    pass  # No classifier for this type yet

            for crop_idx, top_pred in enumerate(per_crop_results):
                internal_name = resolve_internal_name(top_pred[0])
                cls_conf = top_pred[1]
                brand = get_brand(internal_name)
                suggested_labels.append({
                    "internal_name": internal_name,
                    "brand": brand,
                    "confidence": round(cls_conf, 3),
                })
        except Exception:
            suggested_labels = [{"internal_name": "", "brand": "", "confidence": 0.0}] * len(crop_images)

    # Build response
    crops = []
    for idx, (crop, meta) in enumerate(zip(crop_images, crop_meta)):
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
        crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        suggestion = suggested_labels[idx] if idx < len(suggested_labels) else {}
        crops.append({
            "index": meta["index"],
            "image_b64": crop_b64,
            "width": meta["w"],
            "height": meta["h"],
            "det_conf": round(meta["conf"], 3),
            "packaging_type": meta["packaging_type"],
            "suggested_brand": suggestion.get("brand", ""),
            "suggested_product": suggestion.get("internal_name", ""),
            "suggested_confidence": suggestion.get("confidence", 0.0),
        })

    return {"num_crops": len(crops), "crops": crops}


@app.post("/add-reference")
async def add_reference(
    image_file: UploadFile = File(...),
    product_name: str = Form(...),
    packaging_type: str = Form("pack"),
):
    """Add a confirmed crop as a reference image for a specific product and packaging type."""
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name is required.")
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'.")

    try:
        from .brand_registry import BRAND_REGISTRY
    except ImportError:
        from brand_registry import BRAND_REGISTRY

    # Validate product_name exists in registry
    valid_internals = set()
    for brand, products in BRAND_REGISTRY.items():
        for _, internal in products:
            valid_internals.add(internal)

    if product_name not in valid_internals:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown product '{product_name}'. Valid products: {sorted(valid_internals)}",
        )

    data = await image_file.read()
    from PIL import Image
    try:
        Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    TYPE_DIR = _DATA_ROOT / "references" / packaging_type
    TYPE_DIR.mkdir(parents=True, exist_ok=True)

    import re
    existing = list(TYPE_DIR.glob(f"{product_name}_*.*"))
    max_idx = 0
    for p in existing:
        match = re.search(r"_(\d+)$", p.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    next_idx = max_idx + 1

    save_path = TYPE_DIR / f"{product_name}_{next_idx}.jpg"
    save_path.write_bytes(data)

    return {
        "status": "added",
        "product": product_name,
        "packaging_type": packaging_type,
        "filename": save_path.name,
        "total_for_product": next_idx,
    }


@app.get("/brand-registry")
def get_brand_registry():
    """Return the full brand->products hierarchy with per-type reference counts."""
    try:
        from .brand_registry import BRAND_REGISTRY, audit_references
    except ImportError:
        from brand_registry import BRAND_REGISTRY, audit_references

    audit = audit_references()

    hierarchy = {}
    for brand, products in BRAND_REGISTRY.items():
        hierarchy[brand] = []
        for display_name, internal_name in products:
            found_entry = audit["found"].get(internal_name, {})
            # found_entry is now {pkg_type: count} dict
            pack_count = found_entry.get("pack", 0) if isinstance(found_entry, dict) else found_entry
            box_count = found_entry.get("box", 0) if isinstance(found_entry, dict) else 0
            hierarchy[brand].append({
                "display_name": display_name,
                "internal_name": internal_name,
                "reference_count": pack_count + box_count,
                "pack_count": pack_count,
                "box_count": box_count,
            })

    return {
        "brands": hierarchy,
        "total_brands": len(BRAND_REGISTRY),
        "total_products": sum(len(p) for p in BRAND_REGISTRY.values()),
        "products_with_refs": audit.get("total_products_found", len(audit.get("found", {}))),
        "products_missing": audit.get("total_products_missing", len(audit.get("missing", []))),
        "total_images": audit.get("total_images", 0),
        "per_type": audit.get("per_type", {}),
    }


@app.get("/reference-image/{packaging_type}/{filename}")
def get_reference_image(packaging_type: str, filename: str):
    """Serve a reference image by packaging type and filename."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _DATA_ROOT / "references" / packaging_type
    path = REFERENCES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.delete("/reference-image/{packaging_type}/{filename}")
def delete_reference_image(packaging_type: str, filename: str):
    """Delete a reference image by packaging type and filename."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _DATA_ROOT / "references" / packaging_type
    path = REFERENCES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    if not path.resolve().parent == REFERENCES_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid path")
    path.unlink()
    return {"status": "deleted", "packaging_type": packaging_type, "filename": filename}


@app.get("/reference-images/{product_name}")
def list_reference_images(product_name: str, packaging_type: str = "pack"):
    """List all reference image filenames for a product in a packaging type subfolder."""
    if packaging_type not in ("pack", "box"):
        raise HTTPException(status_code=400, detail="packaging_type must be 'pack' or 'box'")
    REFERENCES_DIR = _DATA_ROOT / "references" / packaging_type
    files = sorted(REFERENCES_DIR.glob(f"{product_name}_*.*")) if REFERENCES_DIR.exists() else []
    return {
        "product": product_name,
        "packaging_type": packaging_type,
        "count": len(files),
        "filenames": [f.name for f in files],
    }


@app.get("/dataset-status")
def dataset_status():
    """Check if COCO dataset splits exist for RF-DETR training."""
    ds_root = _BACKEND_ROOT.parent / "datasets" / "cigarette_packs"
    splits = {}
    for split in ("train", "valid", "test"):
        ann = ds_root / split / "_annotations.coco.json"
        if ann.exists():
            import json as json_mod
            try:
                data = json_mod.loads(ann.read_text(encoding="utf-8"))
                splits[split] = {
                    "exists": True,
                    "images": len(data.get("images", [])),
                    "annotations": len(data.get("annotations", [])),
                }
            except Exception:
                splits[split] = {"exists": True, "images": 0, "annotations": 0}
        else:
            splits[split] = {"exists": False, "images": 0, "annotations": 0}
    ready = splits.get("train", {}).get("exists", False) and splits.get("valid", {}).get("exists", False)
    return {"ready": ready, "splits": splits}


_TRAINING_PROGRESS_DIR = _BACKEND_ROOT.parent
_training_jobs: dict[str, dict] = {}
_training_processes: dict[str, object] = {}
_training_lock = threading.Lock()
_TRAINING_HISTORY_PATH = _DATA_ROOT / "training_history.json"
_MODEL_REGISTRY_PATH = _DATA_ROOT / "model_registry.json"
_VERSION_STATE_PATH = _DATA_ROOT / "training_version_state.json"
DEFAULT_TRAINING_VERSION = "v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_training_history() -> list[dict]:
    if not _TRAINING_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(_TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _save_training_history(items: list[dict]) -> None:
    _TRAINING_HISTORY_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_training_history(entry: dict) -> None:
    with _training_lock:
        history = _load_training_history()
        history.append(entry)
        _save_training_history(history)


def _update_training_history(job_id: str, patch: dict) -> None:
    with _training_lock:
        history = _load_training_history()
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("job_id") == job_id:
                history[i].update(patch)
                _save_training_history(history)
                return


def _load_model_registry() -> list[dict]:
    if not _MODEL_REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(_MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _save_model_registry(items: list[dict]) -> None:
    _MODEL_REGISTRY_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_model_registry(entry: dict) -> None:
    with _training_lock:
        rows = _load_model_registry()
        rows.append(entry)
        _save_model_registry(rows)


def _update_model_registry(job_id: str, patch: dict) -> None:
    with _training_lock:
        rows = _load_model_registry()
        for i in range(len(rows) - 1, -1, -1):
            if rows[i].get("job_id") == job_id:
                rows[i].update(patch)
                _save_model_registry(rows)
                return


def _parse_version_num(version: str) -> int:
    if isinstance(version, str) and version.startswith("v") and version[1:].isdigit():
        return int(version[1:])
    return 1


def _next_version(version: str) -> str:
    return f"v{_parse_version_num(version) + 1}"


def _load_version_state() -> dict:
    if not _VERSION_STATE_PATH.exists():
        return {"current_version": DEFAULT_TRAINING_VERSION, "last_trained_version": None}
    try:
        payload = json.loads(_VERSION_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"current_version": DEFAULT_TRAINING_VERSION, "last_trained_version": None}
        payload.setdefault("current_version", DEFAULT_TRAINING_VERSION)
        payload.setdefault("last_trained_version", None)
        return payload
    except Exception:
        return {"current_version": DEFAULT_TRAINING_VERSION, "last_trained_version": None}


def _save_version_state(state: dict) -> None:
    _VERSION_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_current_training_version() -> str:
    return _load_version_state().get("current_version", DEFAULT_TRAINING_VERSION)


def _mark_training_completed(version: str) -> str:
    with _training_lock:
        state = _load_version_state()
        state["last_trained_version"] = version
        if state.get("current_version") == version:
            state["current_version"] = _next_version(version)
        _save_version_state(state)
        return state.get("current_version", DEFAULT_TRAINING_VERSION)


def _hash_dataset_dir(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    for p in files:
        rel = str(p.relative_to(path)).replace("\\", "/")
        stat = p.stat()
        h.update(rel.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()


def _dataset_hash_for_type(model_type: str) -> str:
    if model_type in ("classifier", "dinov2_finetune"):
        return _hash_dataset_dir(_DATA_ROOT / "references")
    if model_type == "rfdetr":
        return _hash_dataset_dir(_BACKEND_ROOT.parent / "datasets" / "cigarette_packs")
    return "unknown"


def _hparam_signature(model_type: str, version: str, params: dict) -> str:
    payload = {
        "model_type": model_type,
        "version": version,
        "params": params,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _find_duplicate_completed_run(model_type: str, version: str, dataset_hash: str, hparam_signature: str) -> dict | None:
    rows = _load_model_registry()
    for row in reversed(rows):
        if (
            row.get("model_type") == model_type
            and row.get("version") == version
            and row.get("dataset_hash") == dataset_hash
            and row.get("hparam_signature") == hparam_signature
            and row.get("status") == "done"
        ):
            return row
    return None


def _metrics_from_progress(progress: dict) -> dict:
    if not isinstance(progress, dict):
        return {}
    out = {}
    if "val_acc" in progress:
        out["val_acc"] = progress.get("val_acc")
    if "best_val_acc" in progress:
        out["best_val_acc"] = progress.get("best_val_acc")
    if "train_acc" in progress:
        out["train_acc"] = progress.get("train_acc")
    if "train_loss" in progress:
        out["train_loss"] = progress.get("train_loss")
    if "epoch" in progress:
        out["epoch"] = progress.get("epoch")
    if "total_epochs" in progress:
        out["total_epochs"] = progress.get("total_epochs")
    return out


def _run_training_job(job_id: str, script: str, args: list[str]):
    """Run a training script as a subprocess with progress file polling."""
    import subprocess
    import sys
    import time

    try:
        progress_file = _TRAINING_PROGRESS_DIR / f".training_progress_{job_id}.json"
        script_path = _TRAINING_PROGRESS_DIR / script
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")
        full_args = [sys.executable, str(script_path),
                     "--progress-file", str(progress_file)] + args

        with _training_lock:
            _training_jobs[job_id].update({
                "status": "running",
                "progress": {},
                "error": None,
                "last_update": _now_iso(),
            })
        _update_model_registry(job_id, {"status": "running", "last_update": _now_iso()})

        # Log to file — avoids PIPE deadlock when training prints more than the pipe buffer.
        log_path = _TRAINING_PROGRESS_DIR / f".training_log_{job_id}.log"
        log_f = open(log_path, "w", encoding="utf-8", errors="replace")
        try:
            process = subprocess.Popen(
                full_args, cwd=str(_TRAINING_PROGRESS_DIR),
                stdout=log_f, stderr=subprocess.STDOUT, text=True,
            )
        except Exception:
            log_f.close()
            raise
        with _training_lock:
            _training_processes[job_id] = process
            _training_jobs[job_id]["pid"] = process.pid

        jt = _training_jobs.get(job_id, {}).get("type", "?")
        print(
            f"[train-local] job={job_id[:8]}… type={jt} pid={process.pid} script={script} "
            f"log={log_path.name}",
            flush=True,
        )

        last_logged_epoch = -1
        # Poll progress file while process runs
        while process.poll() is None:
            time.sleep(2)
            if progress_file.exists():
                try:
                    progress = json.loads(progress_file.read_text())
                    with _training_lock:
                        _training_jobs[job_id]["progress"] = progress
                        _training_jobs[job_id]["last_update"] = _now_iso()
                    # Push to SSE queue if job has one
                    with jobs_lock:
                        job = jobs.get(job_id)
                    if job:
                        epoch = progress.get("epoch", 0)
                        total = progress.get("total_epochs", 1)
                        val_acc = progress.get("val_acc", 0)
                        pct = int((epoch / total) * 100) if total > 0 else 0
                        job["queue"].put((pct, f"Epoch {epoch}/{total} | Val acc: {val_acc:.3f}"))
                    _update_training_history(job_id, {
                        "status": "running",
                        "progress": progress,
                        "last_update": _now_iso(),
                    })
                    _update_model_registry(job_id, {
                        "status": "running",
                        "progress": progress,
                        "last_update": _now_iso(),
                        **_metrics_from_progress(progress),
                    })
                    ep = int(progress.get("epoch", 0) or 0)
                    if ep != last_logged_epoch:
                        last_logged_epoch = ep
                        print(
                            f"[train-local] job={job_id[:8]}… epoch={ep}/{progress.get('total_epochs', '?')} "
                            f"val_acc={progress.get('val_acc', 'n/a')}",
                            flush=True,
                        )
                except Exception:
                    pass

        # Final read
        if progress_file.exists():
            try:
                final_progress = json.loads(progress_file.read_text())
                with _training_lock:
                    _training_jobs[job_id]["progress"] = final_progress
                    _training_jobs[job_id]["last_update"] = _now_iso()
                _update_training_history(job_id, {
                    "progress": final_progress,
                    "last_update": _now_iso(),
                })
                _update_model_registry(job_id, {
                    "progress": final_progress,
                    "last_update": _now_iso(),
                    **_metrics_from_progress(final_progress),
                })
            except Exception:
                pass
            progress_file.unlink(missing_ok=True)

        try:
            log_f.close()
        except Exception:
            pass
        stdout = ""
        if log_path.exists():
            try:
                stdout = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        print(
            f"[train-local] job={job_id[:8]}… subprocess finished rc={process.returncode} "
            f"log_file={log_path}",
            flush=True,
        )

        if process.returncode == 0:
            with _training_lock:
                _training_jobs[job_id]["status"] = "done"
                _training_jobs[job_id]["end_time"] = _now_iso()
                completed_version = _training_jobs[job_id].get("version", DEFAULT_TRAINING_VERSION)
                model_type = _training_jobs[job_id].get("type", "")

            # Hot-reload the model so inference uses the new weights immediately
            if model_type == "rfdetr":
                try:
                    reload_rfdetr()
                    print(f"[train] RF-DETR model hot-reloaded after training (job {job_id[:8]})")
                except Exception as exc:
                    print(f"[train] WARNING: Failed to hot-reload RF-DETR: {exc}")
            elif model_type == "classifier":
                try:
                    reload_classifiers()
                    print(f"[train] Brand classifiers hot-reloaded after training (job {job_id[:8]})")
                except Exception as exc:
                    print(f"[train] WARNING: Failed to hot-reload classifiers: {exc}")
            elif model_type == "dinov2_finetune":
                try:
                    reload_dino()
                    reload_classifiers()
                    print(f"[train] DINOv2 + classifiers hot-reloaded after fine-tuning (job {job_id[:8]})")
                except Exception as exc:
                    print(f"[train] WARNING: Failed to hot-reload DINOv2: {exc}")

            with jobs_lock:
                job = jobs.get(job_id)
            if job:
                job["queue"].put((100, "Training complete"))
                jobs[job_id]["status"] = "done"
                jobs[job_id]["result"] = "TRAINING_COMPLETE"
            next_version = _mark_training_completed(str(completed_version))
            _update_training_history(job_id, {
                "status": "done",
                "end_time": _now_iso(),
                "next_version": next_version,
            })
            with _training_lock:
                progress = _training_jobs.get(job_id, {}).get("progress", {})
            _update_model_registry(job_id, {
                "status": "done",
                "end_time": _now_iso(),
                "next_version": next_version,
                **_metrics_from_progress(progress),
            })
        else:
            with _training_lock:
                stop_requested = bool(_training_jobs.get(job_id, {}).get("stop_requested"))
            if stop_requested:
                with _training_lock:
                    _training_jobs[job_id]["status"] = "stopped"
                    _training_jobs[job_id]["error"] = None
                    _training_jobs[job_id]["end_time"] = _now_iso()
                _update_training_history(job_id, {
                    "status": "stopped",
                    "end_time": _now_iso(),
                })
                _update_model_registry(job_id, {
                    "status": "stopped",
                    "end_time": _now_iso(),
                })
                return
            with _training_lock:
                _training_jobs[job_id]["status"] = "error"
                _training_jobs[job_id]["error"] = stdout[-2000:]
                _training_jobs[job_id]["end_time"] = _now_iso()
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = stdout[-2000:]
            _update_training_history(job_id, {
                "status": "error",
                "error": stdout[-2000:],
                "end_time": _now_iso(),
            })
            _update_model_registry(job_id, {
                "status": "error",
                "error": stdout[-2000:],
                "end_time": _now_iso(),
            })

    except Exception:
        err = traceback.format_exc()
        with _training_lock:
            _training_jobs[job_id]["status"] = "error"
            _training_jobs[job_id]["error"] = err
            _training_jobs[job_id]["end_time"] = _now_iso()
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = err
        _update_training_history(job_id, {
            "status": "error",
            "error": err,
            "end_time": _now_iso(),
        })
        _update_model_registry(job_id, {
            "status": "error",
            "error": err,
            "end_time": _now_iso(),
        })
    finally:
        with _training_lock:
            _training_processes.pop(job_id, None)


@app.post("/train-classifier")
def train_classifier_endpoint(
    epochs: int = 100,
    batch_size: int = 64,
    embed_batch_size: int = 8,
    lr: float = 0.001,
    packaging_type: str = "pack",
    version: str = "",
    force: bool = False,
    use_runpod: bool = False,
):
    """Train the brand classifier (frozen DINOv2 + MLP head). Default: this machine (CPU/GPU). Set use_runpod=true for RunPod GPU (see RUNPOD_CLASSIFIER_GPU_ID)."""
    version = (version or _get_current_training_version()).strip() or DEFAULT_TRAINING_VERSION
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "embed_batch_size": embed_batch_size,
        "lr": lr,
        "use_runpod": use_runpod,
    }
    dataset_hash = _dataset_hash_for_type("classifier")
    hp_sig = _hparam_signature("classifier", version, params)
    if not force:
        dup = _find_duplicate_completed_run("classifier", version, dataset_hash, hp_sig)
        if dup:
            return {
                "skipped": True,
                "reason": "Matching completed run found for same dataset+settings",
                "existing_job_id": dup.get("job_id"),
                "version": version,
                "best_val_acc": dup.get("best_val_acc"),
            }

    job_id = create_job()
    start_time = _now_iso()
    args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--embed-batch-size", str(embed_batch_size),
        "--lr", str(lr),
        "--packaging-type", packaging_type,
    ]
    with _training_lock:
        _training_jobs[job_id] = {
            "job_id": job_id,
            "type": "classifier",
            "version": version,
            "model_type": "classifier",
            "dataset_hash": dataset_hash,
            "hparam_signature": hp_sig,
            "status": "queued",
            "params": params,
            "progress": {},
            "error": None,
            "start_time": start_time,
            "last_update": start_time,
            "end_time": None,
        }
        snapshot = dict(_training_jobs[job_id])
    _append_training_history(snapshot)
    _append_model_registry(snapshot)
    print(
        f"[train-classifier] queued job_id={job_id} use_runpod={use_runpod} "
        f"epochs={epochs} packaging={packaging_type} version={version}",
        flush=True,
    )
    if use_runpod:
        threading.Thread(
            target=run_classifier_training_runpod_job,
            args=(job_id, epochs, batch_size, embed_batch_size, lr, packaging_type),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=_run_training_job,
            args=(job_id, "brand_classifier.py", args),
            daemon=True,
        ).start()
    return {
        "job_id": job_id,
        "type": "classifier",
        "epochs": epochs,
        "version": version,
        "skipped": False,
        "use_runpod": use_runpod,
    }


@app.post("/train-rfdetr")
def train_rfdetr_endpoint(
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 0.0001,
    roboflow_url: str = "",
    clean_dataset: bool = False,
    version: str = "",
    force: bool = False,
):
    """Train RF-DETR detection model. Requires GPU for reasonable speed."""
    version = (version or _get_current_training_version()).strip() or DEFAULT_TRAINING_VERSION
    dataset_import = None
    if roboflow_url.strip():
        dataset_import = download_roboflow_coco(url=roboflow_url.strip(), clean=clean_dataset)

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dataset_source": "roboflow_url" if roboflow_url.strip() else "existing_local",
    }
    dataset_hash = _dataset_hash_for_type("rfdetr")
    hp_sig = _hparam_signature("rfdetr", version, params)
    if not force:
        dup = _find_duplicate_completed_run("rfdetr", version, dataset_hash, hp_sig)
        if dup:
            return {
                "skipped": True,
                "reason": "Matching completed run found for same dataset+settings",
                "existing_job_id": dup.get("job_id"),
                "version": version,
            }

    job_id = create_job()
    start_time = _now_iso()
    args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
    ]
    with _training_lock:
        _training_jobs[job_id] = {
            "job_id": job_id,
            "type": "rfdetr",
            "version": version,
            "model_type": "rfdetr",
            "dataset_hash": dataset_hash,
            "hparam_signature": hp_sig,
            "status": "queued",
            "params": params,
            "dataset_import": dataset_import,
            "progress": {},
            "error": None,
            "start_time": start_time,
            "last_update": start_time,
            "end_time": None,
        }
        snapshot = dict(_training_jobs[job_id])
    _append_training_history(snapshot)
    _append_model_registry(snapshot)
    threading.Thread(
        target=_run_training_job,
        args=(job_id, "train.py", args),
        daemon=True,
    ).start()
    return {
        "job_id": job_id,
        "type": "rfdetr",
        "epochs": epochs,
        "version": version,
        "skipped": False,
        "dataset_import": dataset_import,
    }


@app.post("/finetune-dinov2")
def finetune_dinov2_endpoint(
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 0.00001,
    unfreeze_layers: int = 4,
    version: str = "",
    force: bool = False,
    use_runpod: bool = False,
):
    """Fine-tune DINOv2. Default: runs on this machine (CPU/GPU). Set use_runpod=true for RunPod GPU."""
    version = (version or _get_current_training_version()).strip() or DEFAULT_TRAINING_VERSION
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "unfreeze_layers": unfreeze_layers,
        "use_runpod": use_runpod,
    }
    dataset_hash = _dataset_hash_for_type("dinov2_finetune")
    hp_sig = _hparam_signature("dinov2_finetune", version, params)
    if not force:
        dup = _find_duplicate_completed_run("dinov2_finetune", version, dataset_hash, hp_sig)
        if dup:
            return {
                "skipped": True,
                "reason": "Matching completed run found for same dataset+settings",
                "existing_job_id": dup.get("job_id"),
                "version": version,
                "best_val_acc": dup.get("best_val_acc"),
            }

    job_id = create_job()
    start_time = _now_iso()
    args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--unfreeze-layers", str(unfreeze_layers),
    ]
    with _training_lock:
        _training_jobs[job_id] = {
            "job_id": job_id,
            "type": "dinov2_finetune",
            "version": version,
            "model_type": "dinov2_finetune",
            "dataset_hash": dataset_hash,
            "hparam_signature": hp_sig,
            "status": "queued",
            "params": params,
            "progress": {},
            "error": None,
            "start_time": start_time,
            "last_update": start_time,
            "end_time": None,
        }
        snapshot = dict(_training_jobs[job_id])
    _append_training_history(snapshot)
    _append_model_registry(snapshot)
    print(
        f"[finetune-dinov2] queued job_id={job_id} use_runpod={use_runpod} "
        f"epochs={epochs} batch_size={batch_size} lr={lr} version={version}",
        flush=True,
    )
    if use_runpod:
        threading.Thread(
            target=run_dinov2_finetune_gpu_job,
            args=(job_id, epochs, batch_size, lr, unfreeze_layers),
            daemon=True,
        ).start()
    else:
        threading.Thread(
            target=_run_training_job,
            args=(job_id, "finetune_dinov2.py", args),
            daemon=True,
        ).start()
    return {
        "job_id": job_id,
        "type": "dinov2_finetune",
        "epochs": epochs,
        "version": version,
        "skipped": False,
        "use_runpod": use_runpod,
    }


@app.get("/training-status/{job_id}")
def training_status(job_id: str):
    """Get training progress for a running job."""
    with _training_lock:
        if job_id in _training_jobs:
            return _training_jobs[job_id]
    raise HTTPException(status_code=404, detail="Training job not found")


@app.post("/training-stop/{job_id}")
def training_stop(job_id: str):
    """Stop a running training job by terminating its subprocess or RunPod pod."""
    with _training_lock:
        job = _training_jobs.get(job_id)
        process = _training_processes.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        status = job.get("status", "")
        if status not in ("queued", "running", "stopping"):
            return {"job_id": job_id, "status": status, "message": "Job is not running."}
        job["stop_requested"] = True
        job["status"] = "stopping"
        job["last_update"] = _now_iso()
        runpod_pod_id = job.get("runpod_pod_id")

    if process is None and runpod_pod_id and status == "running":
        api_key = _get_runpod_api_key()
        if api_key:
            try:
                _runpod_gql(api_key, f'mutation {{ podTerminate(input: {{ podId: "{runpod_pod_id}" }}) }}')
                _log_runpod(f"training-stop: terminated RunPod pod {runpod_pod_id} for job {job_id[:8]}…")
            except Exception as exc:
                _log_runpod(f"training-stop: RunPod pod terminate failed {runpod_pod_id}: {exc}")
        return {"job_id": job_id, "status": "stopping", "message": "RunPod pod termination requested."}

    if process is None:
        with _training_lock:
            _training_jobs[job_id]["status"] = "stopped"
            _training_jobs[job_id]["end_time"] = _now_iso()
        _update_training_history(job_id, {"status": "stopped", "end_time": _now_iso()})
        _update_model_registry(job_id, {"status": "stopped", "end_time": _now_iso()})
        return {"job_id": job_id, "status": "stopped", "message": "Queued job cancelled."}

    try:
        process.terminate()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to terminate process: {exc}")

    return {"job_id": job_id, "status": "stopping", "message": "Termination requested."}


@app.get("/training-progress/{job_id}")
def training_progress_stream(job_id: str):
    """SSE stream for live training progress updates."""
    import time

    def event_stream():
        last_payload = None
        while True:
            with _training_lock:
                state = _training_jobs.get(job_id)
                payload = json.dumps(state or {"status": "not_found"}, ensure_ascii=False)

            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload

            if not state or state.get("status") in ("done", "error"):
                break

            time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/training-history")
def training_history(limit: int = 30):
    """Return recent training jobs (latest first)."""
    history = _load_training_history()
    history = list(reversed(history))
    return {"count": len(history), "items": history[: max(1, min(limit, 200))]}


@app.get("/model-registry")
def model_registry(limit: int = 100):
    rows = _load_model_registry()
    rows = list(reversed(rows))
    version_state = _load_version_state()
    return {
        "count": len(rows),
        "items": rows[: max(1, min(limit, 500))],
        "current_version": version_state.get("current_version", DEFAULT_TRAINING_VERSION),
        "last_trained_version": version_state.get("last_trained_version"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": get_device()}
