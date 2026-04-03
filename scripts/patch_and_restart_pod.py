"""Patch pipeline.py on RunPod pod to enable optimize_for_inference, then restart batch."""
import paramiko
import time
import sys

POD_HOST_ID = "99tj7aaxx0tj6d-6441209c"
SSH_KEY = "/root/.ssh/runpod_ed25519"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
client.connect("ssh.runpod.io", port=22, username=POD_HOST_ID, pkey=key, timeout=30)
chan = client.invoke_shell(term="dumb", width=4096, height=50)
time.sleep(2)
while chan.recv_ready():
    chan.recv(65536)
chan.sendall(b"stty -echo\n")
time.sleep(0.5)
while chan.recv_ready():
    chan.recv(65536)


def run_cmd(cmd, wait=3):
    chan.sendall((cmd + "\n").encode())
    time.sleep(wait)
    out = b""
    while chan.recv_ready():
        out += chan.recv(65536)
    return out.decode("utf-8", errors="replace")


# 1. Kill current process
print("Killing process...")
print(run_cmd("kill -9 5668 2>/dev/null; sleep 1; echo KILLED"))

# 2. Patch pipeline.py using python on the pod
print("Patching pipeline.py...")
patch_script = r"""
import re
p = '/workspace/chhat-project/backend/pipeline.py'
txt = open(p).read()
old = '_is_runpod = os.environ.get("RUNPOD_POD_ID") or os.path.exists("/workspace")'
new = '_is_runpod = False  # patched: enable optimize_for_inference on A100'
if old in txt:
    txt = txt.replace(old, new)
    open(p, 'w').write(txt)
    print('PATCHED OK')
elif new in txt:
    print('ALREADY PATCHED')
else:
    print('ERROR: pattern not found')
"""
# Write patch script to pod
run_cmd("cat > /tmp/patch.py << 'PATCHEND'\n" + patch_script + "\nPATCHEND")
print(run_cmd("python3 /tmp/patch.py", wait=3))

# 3. Verify
print("Verifying...")
print(run_cmd('grep "_is_runpod" /workspace/chhat-project/backend/pipeline.py'))

# 4. Restart pipeline
print("Restarting pipeline...")
restart_cmd = (
    "cd /workspace/chhat-project && source .venv/bin/activate && "
    "CUDA_VISIBLE_DEVICES=0 nohup python /workspace/run_batch.py "
    "backend/uploads/12000_batch.csv > /workspace/pipeline_output.log 2>&1 &"
)
run_cmd(restart_cmd, wait=2)
print(run_cmd("echo PID=$!", wait=1))

# 5. Wait and check
time.sleep(5)
print("Checking process...")
print(run_cmd("ps aux | grep python | grep -v grep | head -3"))
print("Row count:")
print(run_cmd("wc -l /workspace/chhat-project/backend/uploads/12000_batch_results.csv"))

chan.sendall(b"exit\n")
time.sleep(2)
chan.close()
client.close()
print("Done.")
