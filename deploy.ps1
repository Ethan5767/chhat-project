param(
    [string]$Server = "root@152.42.247.183",
    [string]$RemoteDir = "/opt/chhat-project",
    [switch]$IncludeAssets,
    [switch]$TrainClassifier
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Ensure-File($path, $hint) {
    if (-not (Test-Path $path)) {
        throw "Required file/folder missing: $path`n$hint"
    }
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Ensure-File "$Root\.env" "Create .env before deploying."

Step "Packaging git HEAD"
$codeTar = Join-Path $Root "deploy_code.tar"
if (Test-Path $codeTar) { Remove-Item $codeTar -Force }
git archive --format=tar --output "$codeTar" HEAD

Step "Uploading and extracting code on server"
scp "$codeTar" "${Server}:/tmp/chhat_deploy_code.tar"
ssh $Server "rm -rf $RemoteDir && mkdir -p $RemoteDir && tar -xf /tmp/chhat_deploy_code.tar -C $RemoteDir && rm -f /tmp/chhat_deploy_code.tar"
Remove-Item $codeTar -Force

Step "Syncing .env"
scp "$Root\.env" "${Server}:${RemoteDir}/.env"
ssh $Server "chmod 600 ${RemoteDir}/.env"

Step "Installing Python dependencies"
ssh $Server "cd $RemoteDir && python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install scikit-learn"

if ($IncludeAssets) {
    Step "Syncing model + references + classifier artifacts"
    Ensure-File "$Root\rf-detr-medium.pth" "Put RF-DETR weights at project root."
    Ensure-File "$Root\backend\references" "Put reference images in backend/references."
    Ensure-File "$Root\backend\classifier_model" "Run/keep classifier artifacts in backend/classifier_model."

    ssh $Server "mkdir -p $RemoteDir/backend/references $RemoteDir/backend/classifier_model"
    scp "$Root\rf-detr-medium.pth" "${Server}:${RemoteDir}/"

    $refTar = Join-Path $Root "references_sync.tar"
    if (Test-Path $refTar) { Remove-Item $refTar -Force }
    tar -cf "$refTar" -C "$Root\backend" "references"
    scp "$refTar" "${Server}:/tmp/references_sync.tar"
    ssh $Server "rm -rf $RemoteDir/backend/references && mkdir -p $RemoteDir/backend && tar -xf /tmp/references_sync.tar -C $RemoteDir/backend && rm -f /tmp/references_sync.tar"
    Remove-Item $refTar -Force

    scp "$Root\backend\classifier_model\*" "${Server}:${RemoteDir}/backend/classifier_model/"
}

if ($TrainClassifier) {
    Step "Training classifier on server"
    ssh $Server "cd $RemoteDir && . .venv/bin/activate && python brand_classifier.py"
}

Step "Restarting services"
ssh $Server "systemctl restart chhat-backend chhat-frontend nginx"

Step "Service status + health checks"
ssh $Server "systemctl is-active chhat-backend chhat-frontend nginx && curl -sS http://127.0.0.1:8000/health"

try {
    $status = (Invoke-WebRequest -UseBasicParsing "http://152.42.247.183" -TimeoutSec 15).StatusCode
    Write-Host "Public UI check: HTTP $status" -ForegroundColor Green
} catch {
    Write-Warning "Public UI check failed: $($_.Exception.Message)"
}

Write-Host "`nDeploy complete." -ForegroundColor Green
