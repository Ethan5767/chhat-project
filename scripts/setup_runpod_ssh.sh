#!/usr/bin/env bash
# Run on the API server as the SAME user that runs uvicorn (systemd User=).
# The backend looks for (in order): ~/.runpod/ssh/RunPod-Key-Go, ~/.ssh/id_ed25519, ~/.ssh/id_rsa
set -euo pipefail

RUNPOD_SSH_DIR="${HOME}/.runpod/ssh"
SSH_DIR="${HOME}/.ssh"
KEY_GO="${RUNPOD_SSH_DIR}/RunPod-Key-Go"
KEY_ED="${SSH_DIR}/id_ed25519"
KEY_RSA="${SSH_DIR}/id_rsa"

mkdir -p "${RUNPOD_SSH_DIR}" "${SSH_DIR}"
chmod 700 "${SSH_DIR}" "${RUNPOD_SSH_DIR}" 2>/dev/null || true

echo "=== RunPod SSH setup (home=${HOME}) ==="
echo ""

found=0
for p in "${KEY_GO}" "${KEY_ED}" "${KEY_RSA}"; do
  if [[ -f "${p}" ]]; then
    echo "Found private key: ${p}"
    found=1
  fi
done

if [[ "${found}" -eq 0 ]]; then
  echo "No key found. Generating ${KEY_ED} (empty passphrase, for server automation)."
  ssh-keygen -t ed25519 -f "${KEY_ED}" -N "" -C "runpod-api-$(hostname -s 2>/dev/null || echo host)"
  chmod 600 "${KEY_ED}"
  echo "Created ${KEY_ED}"
fi

echo ""
echo "Register this PUBLIC key in RunPod: https://www.runpod.io/console/user/settings"
echo "Section: SSH Public Keys (paste one line, save)."
echo ""
echo "----- BEGIN PUBLIC KEY -----"
_show_pub() {
  local priv="$1"
  if [[ -f "${priv}.pub" ]]; then
    cat "${priv}.pub"
  else
    ssh-keygen -y -f "${priv}"
  fi
}
if [[ -f "${KEY_GO}" ]]; then
  _show_pub "${KEY_GO}"
elif [[ -f "${KEY_ED}" ]]; then
  _show_pub "${KEY_ED}"
elif [[ -f "${KEY_RSA}" ]]; then
  _show_pub "${KEY_RSA}"
else
  echo "No private key to derive public key from."
  exit 1
fi
echo "----- END PUBLIC KEY -----"
echo ""
echo "Test from this server after a pod is up (replace host/port):"
echo "  ssh -i <path-to-private-key> -p <publicPort> -o StrictHostKeyChecking=no root@<ip> 'echo ok'"
echo ""
echo "Restart backend after keys are in place: sudo systemctl restart chhat-backend"
