# Sync application source to the droplet without uploading backend/references or large datasets.
# Run from PowerShell on a machine that can SSH (same network/VPN as when deploy.ps1 works).
#
# Usage:
#   .\sync_code_to_server.ps1
#   .\sync_code_to_server.ps1 -Restart

param(
    [string]$Server = "root@152.42.247.183",
    [string]$RemoteDir = "/opt/chhat-project",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Collecting files"
$files = New-Object System.Collections.Generic.List[string]

function Add-File($relativePath) {
    $full = Join-Path $Root $relativePath
    if (Test-Path -LiteralPath $full) {
        if (-not $files.Contains($relativePath)) { $files.Add($relativePath) }
    }
}

Get-ChildItem "$Root\backend" -File -Filter "*.py" | ForEach-Object { Add-File ("backend\" + $_.Name) }
Get-ChildItem "$Root\backend" -File -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object { Add-File ("backend\" + $_.Name) }
Get-ChildItem "$Root\frontend" -Recurse -File -Filter "*.py" -ErrorAction SilentlyContinue | ForEach-Object {
    $rel = $_.FullName.Substring($Root.Length).TrimStart('\')
    Add-File $rel
}
Get-ChildItem "$Root\runpod" -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @(".py", ".sh", ".ps1") } | ForEach-Object {
    Add-File ("runpod\" + $_.Name)
}
foreach ($name in @(
        "brand_classifier.py", "finetune_dinov2.py", "prepare_dataset.py", "train.py",
        "process_glass_view.py", "process_videos.py", "requirements.txt", "deploy.ps1"
    )) {
    Add-File $name
}

$listFile = Join-Path $env:TEMP "chhat_sync_files.txt"
$files | Sort-Object -Unique | ForEach-Object { $_ -replace '\\', '/' } | Set-Content -Path $listFile -Encoding utf8
Write-Host "Files to sync: $($files.Count)"

$tarOut = Join-Path $env:TEMP "chhat_code_sync.tar"
if (Test-Path $tarOut) { Remove-Item $tarOut -Force }

Step "Creating tarball"
tar -cf $tarOut -T $listFile
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$szMb = [math]::Round((Get-Item $tarOut).Length / 1MB, 2)
Write-Host "Archive: $szMb MB"

Step "Uploading to $Server"
scp $tarOut "${Server}:/tmp/chhat_code_sync.tar"

Step "Extracting into $RemoteDir"
ssh $Server "cd $RemoteDir && tar -xf /tmp/chhat_code_sync.tar && rm -f /tmp/chhat_code_sync.tar"

if ($Restart) {
    Step "Restarting services"
    ssh $Server "systemctl restart chhat-backend chhat-frontend && sleep 2 && systemctl is-active chhat-backend chhat-frontend"
}

Remove-Item $listFile -ErrorAction SilentlyContinue
Remove-Item $tarOut -ErrorAction SilentlyContinue

Write-Host "`nDone. Use -Restart next time to reload API + Streamlit." -ForegroundColor Green
