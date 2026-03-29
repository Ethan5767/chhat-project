# From Windows: upload Roboflow COCO train/ folder to RunPod and start RF-DETR training there.
# Training runs ONLY on the pod (GPU). This script only packs, scp, and ssh.
#
# Prerequisites:
#   - OpenSSH (scp/ssh) in PATH
#   - RunPod pod running, TCP SSH exposed (note the port RunPod shows, often not 22)
#   - On the pod once: clone repo + bash runpod/bootstrap_training_pod.sh
#
# Example:
#   .\runpod\upload_coco_train_and_train.ps1 `
#     -SshTarget "root@YOUR_POD_IP" `
#     -SshPort 12345 `
#     -DatasetFolder "C:\Users\kimto\OneDrive\Desktop\RE-AI\rf-detr-cigarette\Cigarette pack brand.coco (5)"
#
param(
    [Parameter(Mandatory = $true)]
    [string] $SshTarget,
    [int] $SshPort = 22,
    [Parameter(Mandatory = $true)]
    [string] $DatasetFolder,
    [string] $RemoteDir = "/workspace/chhat-project",
    [string] $ArchiveName = "rfdetr_coco_train.tar.gz"
)

$ErrorActionPreference = "Stop"
$trainPath = Join-Path $DatasetFolder "train"
if (-not (Test-Path $trainPath)) {
    throw "Expected a Roboflow-style folder with a train\ subfolder. Not found: $trainPath"
}
$ann = Join-Path $trainPath "_annotations.coco.json"
if (-not (Test-Path $ann)) {
    throw "Missing COCO annotations: $ann"
}

$tarPath = Join-Path $env:TEMP $ArchiveName
if (Test-Path $tarPath) { Remove-Item $tarPath -Force }

Write-Host "Creating $tarPath (train/ only)..."
# Archive contents: train/... so extract -C datasets/cigarette_packs yields datasets/cigarette_packs/train/...
& tar.exe -czf $tarPath -C $DatasetFolder train
if ($LASTEXITCODE -ne 0) { throw "tar failed" }

$remoteTar = ($RemoteDir.TrimEnd("/") + "/" + $ArchiveName)
Write-Host "Uploading to ${SshTarget}:$remoteTar ..."
& scp.exe -P $SshPort $tarPath "${SshTarget}:$remoteTar"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

$remoteCmd = "cd $RemoteDir && bash runpod/train_rfdetr_only.sh $ArchiveName"

Write-Host "Starting training on RunPod (this uses the pod GPU, not your laptop)..."
& ssh.exe -p $SshPort $SshTarget $remoteCmd
if ($LASTEXITCODE -ne 0) { throw "ssh / training failed" }

Write-Host "Done. On your laptop, pull checkpoints back, e.g.:"
Write-Host "  scp -P $SshPort -r ${SshTarget}:$RemoteDir/runs <local-folder>"
