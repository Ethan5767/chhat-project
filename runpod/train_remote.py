"""RunPod training launcher -- creates pod, uploads dataset, trains, downloads results.

Usage:
    python runpod/train_remote.py --dataset "Cigarette pack brand.coco (5)" --epochs 50
    python runpod/train_remote.py --dataset "Cigarette pack brand.coco (5)" --model dinov2 --epochs 30
    python runpod/train_remote.py --dataset "Cigarette pack brand.coco (5)" --model both

Handles the full lifecycle:
  1. Fix COCO annotations (string bbox values)
  2. Create tarball of dataset
  3. Create RunPod A100 pod via API
  4. Wait for SSH to be ready
  5. Upload dataset + code via SCP
  6. Run training via SSH
  7. Download checkpoints via SCP
  8. Terminate pod

Requires: RUNPOD_API_KEY env var or ~/.runpod/config.toml
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNPODCTL = PROJECT_ROOT / "runpod" / "runpodctl.exe"

# RunPod API
API_URL = "https://api.runpod.io/graphql"
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"
DEFAULT_TEMPLATE = "runpod-torch-v21"
DEFAULT_CLOUD = "COMMUNITY"
POLL_INTERVAL = 15
MAX_WAIT_SECONDS = 300


def get_api_key():
    key = os.environ.get("RUNPOD_API_KEY", "")
    if key:
        return key
    config = Path.home() / ".runpod" / "config.toml"
    if config.exists():
        for line in config.read_text().splitlines():
            if "apiKey" in line or "api_key" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("No RunPod API key found. Set RUNPOD_API_KEY or run: runpodctl config --apiKey YOUR_KEY")


def gql(api_key, query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {json.dumps(data['errors'], indent=2)}")
    return data["data"]


def fix_coco_annotations(ann_path: Path) -> int:
    """Fix Roboflow exports that have string bbox values. Returns count of fixed annotations."""
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    fixed = 0
    for ann in data["annotations"]:
        new_bbox = [float(v) if isinstance(v, str) else v for v in ann["bbox"]]
        if new_bbox != ann["bbox"]:
            ann["bbox"] = new_bbox
            fixed += 1
        if isinstance(ann.get("area"), str):
            ann["area"] = float(ann["area"])
    if fixed:
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return fixed


def create_dataset_tarball(dataset_folder: Path) -> Path:
    """Create a tarball from a Roboflow COCO export folder."""
    train_dir = dataset_folder / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"No train/ folder in {dataset_folder}")
    ann_path = train_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"No _annotations.coco.json in {train_dir}")

    # Fix annotations
    fixed = fix_coco_annotations(ann_path)
    if fixed:
        print(f"  Fixed {fixed} annotations with string bbox values")

    # Show dataset info
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cats = [c["name"] for c in data.get("categories", [])]
    print(f"  Images: {len(data['images'])}, Annotations: {len(data['annotations'])}")
    print(f"  Categories: {cats}")

    # Create tarball
    tar_path = PROJECT_ROOT / "datasets" / "rfdetr_coco_train.tar.gz"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Creating {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(train_dir), arcname="train")
        # Include valid/ and test/ if they exist
        for split in ("valid", "test"):
            split_dir = dataset_folder / split
            if split_dir.exists():
                tar.add(str(split_dir), arcname=split)

    size_mb = tar_path.stat().st_size / 1024 / 1024
    print(f"  Tarball: {size_mb:.1f} MB")
    return tar_path


def create_pod(api_key, gpu=DEFAULT_GPU, cloud=DEFAULT_CLOUD):
    """Create a RunPod GPU pod and return pod ID."""
    print(f"Creating {gpu} pod ({cloud})...")
    data = gql(api_key, """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id name desiredStatus costPerHr
            machine { gpuDisplayName }
        }
    }""", {"input": {
        "name": "rfdetr-training",
        "templateId": DEFAULT_TEMPLATE,
        "gpuTypeId": gpu,
        "cloudType": cloud,
        "containerDiskInGb": 20,
        "volumeInGb": 200,
        "volumeMountPath": "/workspace",
        "gpuCount": 1,
        "ports": "22/tcp",
    }})
    pod = data["podFindAndDeployOnDemand"]
    print(f"  Pod ID: {pod['id']}")
    print(f"  GPU: {pod['machine']['gpuDisplayName']}")
    print(f"  Cost: ${pod['costPerHr']}/hr")
    return pod["id"]


def wait_for_ssh(api_key, pod_id, timeout=MAX_WAIT_SECONDS):
    """Wait for pod SSH port to be available. Returns (host, port, proxy_user)."""
    print("Waiting for pod to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        data = gql(api_key, f"""query {{
            pod(input: {{ podId: "{pod_id}" }}) {{
                id desiredStatus
                machine {{ podHostId }}
                runtime {{
                    uptimeInSeconds
                    ports {{ ip publicPort privatePort type }}
                }}
            }}
        }}""")
        pod = data["pod"]
        runtime = pod.get("runtime")
        if runtime and runtime.get("ports"):
            ssh_ports = [p for p in runtime["ports"] if p["privatePort"] == 22]
            if ssh_ports:
                host = ssh_ports[0]["ip"]
                port = ssh_ports[0]["publicPort"]
                host_id = pod.get("machine", {}).get("podHostId", "")
                uptime = runtime.get("uptimeInSeconds", 0)
                print(f"  Pod running (uptime: {uptime}s)")
                print(f"  SSH: {host}:{port}")
                print(f"  Proxy: {host_id}@ssh.runpod.io")
                return host, port, host_id
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] waiting...")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Pod {pod_id} did not become ready in {timeout}s")


def wait_for_sshd(host, port, key_path, timeout=120):
    """Wait for SSH daemon to accept connections."""
    print("Waiting for SSH daemon...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = subprocess.run(
                ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                 "-i", str(key_path), "-p", str(port), f"root@{host}", "echo SSH_OK"],
                capture_output=True, text=True, timeout=15,
            )
            if "SSH_OK" in r.stdout:
                print("  SSH connected!")
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] sshd not ready...")
        time.sleep(10)
    return False


def find_ssh_key():
    """Find the SSH private key to use (RunPod-generated or user's)."""
    runpod_key = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
    if runpod_key.exists():
        return runpod_key
    ed25519_key = Path.home() / ".ssh" / "id_ed25519"
    if ed25519_key.exists():
        return ed25519_key
    rsa_key = Path.home() / ".ssh" / "id_rsa"
    if rsa_key.exists():
        return rsa_key
    raise FileNotFoundError("No SSH key found. Run: runpodctl config --apiKey YOUR_KEY")


def ssh_cmd(host, port, key_path, command, timeout=None):
    """Execute a command on the pod via SSH."""
    args = [
        "ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-i", str(key_path), "-p", str(port), f"root@{host}", command,
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def scp_upload(host, port, key_path, local_path, remote_path):
    """Upload a file to the pod via SCP."""
    args = [
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-i", str(key_path), "-P", str(port), str(local_path), f"root@{host}:{remote_path}",
    ]
    print(f"  Uploading {Path(local_path).name}...")
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"SCP upload failed: {r.stderr}")


def scp_download(host, port, key_path, remote_path, local_path):
    """Download a file/folder from the pod via SCP."""
    args = [
        "scp", "-r", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-i", str(key_path), "-P", str(port), f"root@{host}:{remote_path}", str(local_path),
    ]
    print(f"  Downloading {remote_path}...")
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"SCP download failed: {r.stderr}")


def terminate_pod(api_key, pod_id):
    """Terminate pod."""
    print(f"Terminating pod {pod_id}...")
    gql(api_key, f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
    print("  Pod terminated.")


def run_training(host, port, key_path, model, epochs, batch_size, lr):
    """Run training on the pod, streaming output."""
    if model == "rfdetr":
        cmd = (
            "cd /workspace/chhat-project && "
            "source .venv/bin/activate && "
            "python train.py "
            f"--epochs {epochs} --batch-size {batch_size} --lr {lr} --grad-accum-steps 4"
        )
    elif model == "dinov2":
        cmd = (
            "cd /workspace/chhat-project && "
            "source .venv/bin/activate && "
            "python finetune_dinov2.py "
            f"--epochs {epochs} --batch-size {batch_size} --lr {lr}"
        )
    elif model == "classifier":
        cmd = (
            "cd /workspace/chhat-project && "
            "source .venv/bin/activate && "
            "python brand_classifier.py "
            f"--epochs {epochs} --batch-size 64 --lr {lr}"
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    print(f"\nStarting {model} training (epochs={epochs}, batch={batch_size}, lr={lr})...")
    print("=" * 60)

    # Use subprocess.Popen for streaming output
    args = [
        "ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10",
        "-i", str(key_path), "-p", str(port), f"root@{host}", cmd,
    ]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.wait()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        process.terminate()
        return False

    print("=" * 60)
    if process.returncode == 0:
        print("Training completed successfully!")
        return True
    else:
        print(f"Training failed (exit code {process.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run training on RunPod GPU")
    parser.add_argument("--dataset", required=True, help="Path to Roboflow COCO export folder")
    parser.add_argument("--model", default="rfdetr", choices=["rfdetr", "dinov2", "classifier", "both"],
                        help="Which model to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--no-terminate", action="store_true", help="Keep pod alive after training")
    parser.add_argument("--pod-id", default="", help="Use existing pod instead of creating new one")
    args = parser.parse_args()

    api_key = get_api_key()
    ssh_key = find_ssh_key()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    # Step 1: Prepare dataset
    print("\n[1/7] Preparing dataset...")
    tar_path = create_dataset_tarball(dataset_path)

    # Step 2: Create pod (or use existing)
    pod_id = args.pod_id
    if not pod_id:
        print("\n[2/7] Creating RunPod pod...")
        pod_id = create_pod(api_key, gpu=args.gpu)
    else:
        print(f"\n[2/7] Using existing pod: {pod_id}")

    try:
        # Step 3: Wait for SSH
        print("\n[3/7] Waiting for pod SSH...")
        host, port, proxy_user = wait_for_ssh(api_key, pod_id)

        # Wait for sshd to actually start
        time.sleep(30)  # Give container time to boot
        if not wait_for_sshd(host, port, ssh_key):
            print("SSH daemon not responding. Trying RunPod proxy...")
            # Fallback: try proxy
            proxy_host = "ssh.runpod.io"
            proxy_port = 22
            proxy_target = f"{proxy_user}@{proxy_host}"
            if not wait_for_sshd(proxy_host, proxy_port, ssh_key, timeout=60):
                raise RuntimeError("Cannot connect via SSH (direct or proxy)")
            host, port = proxy_host, proxy_port

        # Step 4: Bootstrap pod
        print("\n[4/7] Bootstrapping pod...")
        r = ssh_cmd(host, port, ssh_key,
                     "ls /workspace/chhat-project/train.py 2>/dev/null && echo REPO_EXISTS || echo NEED_CLONE",
                     timeout=30)
        if "NEED_CLONE" in r.stdout:
            print("  Cloning repo...")
            r = ssh_cmd(host, port, ssh_key,
                         "cd /workspace && git clone https://github.com/Ethan5767/chhat-project.git && "
                         "cd chhat-project && bash runpod/bootstrap_training_pod.sh",
                         timeout=300)
            if r.returncode != 0:
                print(f"  Bootstrap output: {r.stdout[-500:]}")
                raise RuntimeError("Bootstrap failed")
            print("  Bootstrap complete.")
        else:
            print("  Repo already exists, pulling latest...")
            ssh_cmd(host, port, ssh_key,
                     "cd /workspace/chhat-project && git pull", timeout=60)

        # Step 5: Upload dataset
        print("\n[5/7] Uploading dataset...")
        scp_upload(host, port, ssh_key, tar_path, "/workspace/chhat-project/rfdetr_coco_train.tar.gz")

        # Extract dataset (RunPod pod workspace only — not your droplet)
        print("  Extracting on pod...")
        r = ssh_cmd(host, port, ssh_key,
                     "cd /workspace/chhat-project && "
                     "rm -rf datasets/cigarette_packs/train datasets/cigarette_packs/valid datasets/cigarette_packs/test && "
                     "mkdir -p datasets/cigarette_packs && "
                     "tar -xzf rfdetr_coco_train.tar.gz -C datasets/cigarette_packs && "
                     "ls datasets/cigarette_packs/",
                     timeout=120)
        print(f"  Extracted: {r.stdout.strip()}")

        # Step 6: Run training
        print("\n[6/7] Training...")
        models_to_train = [args.model] if args.model != "both" else ["rfdetr", "classifier"]
        all_success = True
        for model in models_to_train:
            success = run_training(host, port, ssh_key, model, args.epochs, args.batch_size, args.lr)
            if not success:
                all_success = False
                print(f"  {model} training failed!")

        # Step 7: Download results
        if all_success:
            print("\n[7/7] Downloading results...")
            results_dir = PROJECT_ROOT / "runs"
            results_dir.mkdir(exist_ok=True)
            try:
                scp_download(host, port, ssh_key, "/workspace/chhat-project/runs/", str(results_dir))
                print(f"  Checkpoints saved to {results_dir}")
            except Exception as e:
                print(f"  Download failed: {e}")
                print(f"  You can manually download: scp -r -P {port} root@{host}:/workspace/chhat-project/runs/ .")

    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        if not args.no_terminate and not args.pod_id:
            terminate_pod(api_key, pod_id)
        elif args.no_terminate:
            print(f"\nPod {pod_id} left running (--no-terminate). Remember to terminate it!")

    print("\nDone.")


if __name__ == "__main__":
    main()
