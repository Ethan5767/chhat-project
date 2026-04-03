"""SSH into RunPod pod and check GPU process monitor."""
import os
import paramiko
import time

POD_HOST_ID = "cdygbp3b0bderb-64412070"

key = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser("~/.ssh/runpod_ed25519"))
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("ssh.runpod.io", port=22, username=POD_HOST_ID, pkey=key, timeout=30)
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


print("=== GPU SUMMARY ===")
print(run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv", 3))

print("=== GPU PMON (5 samples, ~20s) ===")
print(run("nvidia-smi pmon -c 5 -s mu 2>/dev/null", 20))

print("=== PROGRESS FILE ===")
print(run("cat /tmp/rfdetr_progress_1757a6a1-f680-4e0b-9842-9c866cc45378.json 2>/dev/null || echo NO_FILE", 3))

print("=== TRAINING PROCESSES ===")
print(run("ps aux | grep train.py | grep -v grep | head -3", 3))

shell.close()
client.close()
