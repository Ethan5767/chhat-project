"""SSH into RunPod pod and check GPU utilization + training env."""
import os
import sys
import paramiko
import time

key = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser("~/.ssh/runpod_ed25519"))
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("ssh.runpod.io", port=22, username="l884w11zqg2cp2-64410a91", pkey=key, timeout=30)
shell = client.invoke_shell(term="dumb", width=4096, height=100)
time.sleep(2)
shell.recv(65536)
shell.send("stty -echo\n")
time.sleep(0.5)
shell.recv(65536)


def run(cmd, wait=5):
    shell.send(cmd + "\n")
    time.sleep(wait)
    out = b""
    while shell.recv_ready():
        out += shell.recv(65536)
        time.sleep(0.3)
    return out.decode(errors="replace")


# 1. Training process environment
print("=== TRAINING PROCESS ENV ===")
print(run("cat /proc/407/environ 2>/dev/null | tr '\\0' '\\n' | grep -E 'CUDA|PYTORCH|DEVICE|JIT' || echo NO_PROC", 3))

# 2. nvidia-smi pmon
print("=== GPU PMON (3 samples) ===")
print(run("nvidia-smi pmon -c 3 -s mu 2>/dev/null", 12))

# 3. nvidia-smi summary
print("=== GPU SUMMARY ===")
print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader", 3))

# 4. Write and run PTL test script
script_lines = [
    "import torch, os",
    "from pytorch_lightning import Trainer",
    "t = Trainer(accelerator='auto', devices=1, max_epochs=1)",
    "acc = t.accelerator.__class__.__name__",
    "print('PTL_ACC=' + acc)",
    "print('PTL_PREC=' + str(t.precision))",
]
for line in script_lines:
    shell.send("echo '" + line + "' >> /tmp/ptl_test.py\n")
    time.sleep(0.1)
time.sleep(0.5)
shell.recv(65536)

print("=== PTL AUTO RESOLUTION ===")
print(run("cd /workspace/chhat-project && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python /tmp/ptl_test.py 2>&1; echo PTL_DONE", 20))

# 5. Check current epoch progress
print("=== PROGRESS FILE ===")
job_id = "3d07cc12-30b1-47aa-9601-3c53edb932ef"
print(run("cat /tmp/rfdetr_progress_" + job_id + ".json 2>/dev/null || echo NO_FILE", 3))

shell.close()
client.close()
