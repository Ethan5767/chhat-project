param(
    [string]$Server = "root@152.42.247.183",
    [string]$RemoteDir = "/opt/chhat-project",
    # Persistent data (never deleted by normal deploy): references, classifiers, uploads, runs, JSON state
    [string]$DataRoot = "/var/lib/chhat-project",
    [switch]$IncludeAssets,
    [switch]$TrainClassifier,
    # Sync RF-DETR checkpoints: local runs\*.pth -> server $DataRoot/runs (skipped if no local .pth)
    [switch]$SyncRfdetrCheckpoints,
    # Destructive: deletes entire remote *code* dir before extract. Does NOT delete $DataRoot.
    [switch]$CleanRemote
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

Step "Server: persistent data dir + systemd (CHHAT_DATA_ROOT=$DataRoot)"
$d = $DataRoot
$r = $RemoteDir
ssh $Server "mkdir -p $d/references/pack $d/references/box $d/classifier_model/pack $d/classifier_model/box $d/uploads/results $d/runs && chmod 755 $d && install -d /etc/systemd/system/chhat-backend.service.d && printf '%s\n' '[Service]' 'Environment=CHHAT_DATA_ROOT=$d' > /etc/systemd/system/chhat-backend.service.d/chhat-data.conf && systemctl daemon-reload"
# One-time migration from pre-CHHAT_DATA_ROOT layout (under deploy tree) if persistent dir is still empty
ssh $Server "if [ -d $r/backend/references/pack ] && [ ! -d $d/references/pack ]; then rsync -a $r/backend/references/ $d/references/; fi"
ssh $Server "if [ -f $r/backend/batch_history.json ] && [ ! -f $d/batch_history.json ]; then cp -a $r/backend/batch_history.json $d/; fi"
ssh $Server "if [ -f $r/backend/training_history.json ] && [ ! -f $d/training_history.json ]; then cp -a $r/backend/training_history.json $d/; fi"
ssh $Server "if [ -f $r/backend/model_registry.json ] && [ ! -f $d/model_registry.json ]; then cp -a $r/backend/model_registry.json $d/; fi"
ssh $Server "if [ -f $r/backend/training_version_state.json ] && [ ! -f $d/training_version_state.json ]; then cp -a $r/backend/training_version_state.json $d/; fi"
ssh $Server "if [ -d $r/backend/classifier_model ] && [ ! -f $d/classifier_model/best_classifier.pth ] && [ ! -f $d/classifier_model/pack/best_classifier.pth ]; then rsync -a $r/backend/classifier_model/ $d/classifier_model/; fi"

Step "Packaging git HEAD"
$codeTar = Join-Path $Root "deploy_code.tar"
if (Test-Path $codeTar) { Remove-Item $codeTar -Force }
git archive --format=tar --output "$codeTar" HEAD

Step "Uploading and extracting code on server (data dir NOT wiped)"
scp "$codeTar" "${Server}:/tmp/chhat_deploy_code.tar"
if ($CleanRemote) {
    Step "CleanRemote: removing $RemoteDir (code only; $DataRoot is kept)"
    ssh $Server "rm -rf $RemoteDir && mkdir -p $RemoteDir"
} else {
    ssh $Server "mkdir -p $RemoteDir"
}
ssh $Server "tar -xf /tmp/chhat_deploy_code.tar -C $RemoteDir && rm -f /tmp/chhat_deploy_code.tar"
Remove-Item $codeTar -Force

Step "Syncing .env"
scp "$Root\.env" "${Server}:${RemoteDir}/.env"
ssh $Server "chmod 600 ${RemoteDir}/.env"

Step "Installing Python dependencies"
ssh $Server "cd $RemoteDir && python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install scikit-learn"

if ($IncludeAssets) {
    Step "Syncing model + references + classifier into $DataRoot"
    Ensure-File "$Root\rf-detr-medium.pth" "Put RF-DETR weights at project root."
    Ensure-File "$Root\backend\references" "Put reference images in backend/references."
    Ensure-File "$Root\backend\classifier_model" "Run/keep classifier artifacts in backend/classifier_model."

    scp "$Root\rf-detr-medium.pth" "${Server}:${RemoteDir}/"

    $refTar = Join-Path $Root "references_sync.tar"
    if (Test-Path $refTar) { Remove-Item $refTar -Force }
    tar -cf "$refTar" -C "$Root\backend" "references"
    scp "$refTar" "${Server}:/tmp/references_sync.tar"
    ssh $Server "rm -rf '$DataRoot/references' && mkdir -p '$DataRoot' && tar -xf /tmp/references_sync.tar -C '$DataRoot' && rm -f /tmp/references_sync.tar"
    Remove-Item $refTar -Force

    ssh $Server "mkdir -p '$DataRoot/classifier_model'"
    scp "$Root\backend\classifier_model\*" "${Server}:${DataRoot}/classifier_model/"
}

if ($SyncRfdetrCheckpoints) {
    Step "Sync RF-DETR run checkpoints to ${DataRoot}/runs"
    $runsLocal = Join-Path $Root "runs"
    if (-not (Test-Path $runsLocal)) {
        Write-Warning "No local runs/ folder; nothing to sync."
    } else {
        $pth = Get-ChildItem -Path $runsLocal -Recurse -Filter "*.pth" -ErrorAction SilentlyContinue
        if (-not $pth) {
            Write-Warning "No .pth files under runs/; skip SyncRfdetrCheckpoints."
        } else {
            ssh $Server "mkdir -p '$DataRoot/runs'"
            scp -r "$runsLocal\*" "${Server}:${DataRoot}/runs/"
        }
    }
}

if ($TrainClassifier) {
    Step "Training classifier on server"
    ssh $Server "cd $RemoteDir && . .venv/bin/activate && python brand_classifier.py"
}

Step "Restarting services"
ssh $Server "systemctl restart chhat-backend chhat-frontend nginx"

Step "Service status + health checks"
ssh $Server "sleep 4; systemctl is-active chhat-backend chhat-frontend nginx && curl -sS --max-time 15 http://127.0.0.1:8000/health"

try {
    $status = (Invoke-WebRequest -UseBasicParsing "http://152.42.247.183" -TimeoutSec 15).StatusCode
    Write-Host "Public UI check: HTTP $status" -ForegroundColor Green
} catch {
    Write-Warning "Public UI check failed: $($_.Exception.Message)"
}

Write-Host "`nDeploy complete. Persistent data: $DataRoot (unchanged unless -IncludeAssets or -SyncRfdetrCheckpoints)." -ForegroundColor Green
