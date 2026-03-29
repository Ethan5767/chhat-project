# Deploy application code to the server. Default: extracts git archive OVER existing tree — does NOT delete
# /opt/chhat-project. Gitignored data (references, runs/, uploads, etc.) remains unless you opt into destructive flags.
#
# DESTRUCTIVE FLAGS (use only when you intend to remove data):
#   -CleanRemote + -ConfirmFullRemoteWipe   Deletes entire $RemoteDir before unpack (same as old dangerous default).
#   -IncludeAssets + -ReplaceReferencesOnServer   Deletes server backend/references before unpacking the references tarball.
#   Normal -IncludeAssets merges/overwrites from tarball without deleting the references folder first.

param(
    [string]$Server = "root@152.42.247.183",
    [string]$RemoteDir = "/opt/chhat-project",
    [switch]$IncludeAssets,
    # Only with -IncludeAssets: rm -rf backend/references on server before extracting references tarball.
    [switch]$ReplaceReferencesOnServer,
    [switch]$TrainClassifier,
    [switch]$SyncRfdetrCheckpoints,
    # Deletes entire $RemoteDir — requires -ConfirmFullRemoteWipe to run (two-switch safety).
    [switch]$CleanRemote,
    [switch]$ConfirmFullRemoteWipe
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

# Remove obsolete systemd drop-in from older experiments only (single small conf file, not project data)
ssh $Server "rm -f /etc/systemd/system/chhat-backend.service.d/chhat-data.conf 2>/dev/null; systemctl daemon-reload 2>/dev/null; true"

if ($CleanRemote -and -not $ConfirmFullRemoteWipe) {
    throw "CleanRemote is blocked unless you also pass -ConfirmFullRemoteWipe. This would delete everything under $RemoteDir on the server."
}

if ($ReplaceReferencesOnServer -and -not $IncludeAssets) {
    throw "-ReplaceReferencesOnServer only applies with -IncludeAssets."
}

Step "Packaging git HEAD"
$codeTar = Join-Path $Root "deploy_code.tar"
if (Test-Path $codeTar) { Remove-Item $codeTar -Force }
git archive --format=tar --output "$codeTar" HEAD

Step "Uploading and extracting code (existing server files outside git are kept)"
scp "$codeTar" "${Server}:/tmp/chhat_deploy_code.tar"
if ($CleanRemote -and $ConfirmFullRemoteWipe) {
    Step "CleanRemote: removing entire $RemoteDir"
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
    Step "Syncing model + references + classifier into $RemoteDir"
    Ensure-File "$Root\rf-detr-medium.pth" "Put RF-DETR weights at project root."
    Ensure-File "$Root\backend\references" "Put reference images in backend/references."
    Ensure-File "$Root\backend\classifier_model" "Run/keep classifier artifacts in backend/classifier_model."

    ssh $Server "mkdir -p $RemoteDir/backend/references $RemoteDir/backend/classifier_model"
    scp "$Root\rf-detr-medium.pth" "${Server}:${RemoteDir}/"

    $refTar = Join-Path $Root "references_sync.tar"
    if (Test-Path $refTar) { Remove-Item $refTar -Force }
    tar -cf "$refTar" -C "$Root\backend" "references"
    scp "$refTar" "${Server}:/tmp/references_sync.tar"
    if ($ReplaceReferencesOnServer) {
        ssh $Server "rm -rf $RemoteDir/backend/references && mkdir -p $RemoteDir/backend && tar -xf /tmp/references_sync.tar -C $RemoteDir/backend && rm -f /tmp/references_sync.tar"
    } else {
        ssh $Server "mkdir -p $RemoteDir/backend && tar -xf /tmp/references_sync.tar -C $RemoteDir/backend && rm -f /tmp/references_sync.tar"
    }
    Remove-Item $refTar -Force

    scp "$Root\backend\classifier_model\*" "${Server}:${RemoteDir}/backend/classifier_model/"
}

if ($SyncRfdetrCheckpoints) {
    Step "Sync RF-DETR checkpoints to ${RemoteDir}/runs"
    $runsLocal = Join-Path $Root "runs"
    if (-not (Test-Path $runsLocal)) {
        Write-Warning "No local runs/ folder."
    } else {
        $pth = Get-ChildItem -Path $runsLocal -Recurse -Filter "*.pth" -ErrorAction SilentlyContinue
        if (-not $pth) {
            Write-Warning "No .pth under runs/."
        } else {
            ssh $Server "mkdir -p $RemoteDir/runs"
            scp -r "$runsLocal\*" "${Server}:${RemoteDir}/runs/"
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

Write-Host "`nDeploy complete. Server data under $RemoteDir is preserved unless you used -CleanRemote -ConfirmFullRemoteWipe or -IncludeAssets -ReplaceReferencesOnServer." -ForegroundColor Green
