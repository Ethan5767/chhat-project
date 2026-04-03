"""Fix pipeline.py on pod: remove global socket timeout, restore prefetch, restart batch."""
import paramiko
import time

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
print("Killing...")
print(run_cmd("kill -9 $(pgrep -f run_batch) 2>/dev/null; sleep 1; echo KILLED"))

# 2. Check current rows
print("Current rows:")
print(run_cmd("wc -l /workspace/chhat-project/backend/uploads/12000_batch_results.csv"))

# 3. Patch pipeline.py on the pod using a file-based approach
patch_script = r'''import re
p = "/workspace/chhat-project/backend/pipeline.py"
lines = open(p).readlines()
new_lines = []
for line in lines:
    # Remove the socket timeout line we added
    if line.strip().startswith("import socket; socket.setdefaulttimeout"):
        continue
    # Restore prefetch
    if "PREFETCH_AHEAD = 8" in line:
        line = line.replace("PREFETCH_AHEAD = 8", "PREFETCH_AHEAD = 48")
    new_lines.append(line)
open(p, "w").writelines(new_lines)
# Verify
for i, line in enumerate(open(p)):
    for pat in ["PREFETCH_AHEAD", "MULTI_ROW_BATCH", "SAVE_INTERVAL", "socket.setdefault", "_is_runpod"]:
        if pat in line:
            print(f"  L{i+1}: {line.rstrip()}")
print("PATCH_DONE")
'''

# Write patch script to pod
run_cmd("cat > /tmp/fix_pipeline.py << 'FIXEND'\n" + patch_script + "\nFIXEND")
print("Patching pipeline.py...")
print(run_cmd("python3 /tmp/fix_pipeline.py", wait=3))

# 4. Restart pipeline
print("Restarting pipeline...")
restart = (
    "cd /workspace/chhat-project && source .venv/bin/activate && "
    "CUDA_VISIBLE_DEVICES=0 nohup python /workspace/run_batch.py "
    "backend/uploads/12000_batch.csv > /workspace/pipeline_output.log 2>&1 &"
)
run_cmd(restart, wait=2)
print(run_cmd("echo PID=$!", wait=1))

# 5. Wait and verify
time.sleep(8)
print("Process check:")
print(run_cmd("ps aux | grep python | grep -v grep | head -3"))

# 6. Wait more and check rows
time.sleep(60)
print("Row count after 60s:")
print(run_cmd("wc -l /workspace/chhat-project/backend/uploads/12000_batch_results.csv"))

chan.sendall(b"exit\n")
time.sleep(2)
chan.close()
client.close()
print("Done.")
