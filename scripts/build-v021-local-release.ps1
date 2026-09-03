param(
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$package = Get-Content -LiteralPath (Join-Path $projectRoot "package.json") -Raw | ConvertFrom-Json
$version = [string]$package.version
if ($version -ne "0.2.1") {
    throw "Expected v0.2.1, got $version"
}

$releaseRoot = Join-Path $projectRoot "release"
$localReleaseRoot = Join-Path $projectRoot "release-local"
$python = Join-Path $projectRoot ".venv-cuda312/Scripts/python.exe"
$electronBuilder = Join-Path $projectRoot "node_modules/.bin/electron-builder.cmd"
$setupPath = Join-Path $releaseRoot "一点筛图-$version-Windows-Setup.exe"
$volumePattern = "一点筛图-$version-Windows-CUDA.7z.*"
$singleInstallerPath = Join-Path $localReleaseRoot "一点筛图-$version-Windows-CUDA-完整离线版.exe"

$requiredPaths = @(
    $python,
    $electronBuilder,
    (Join-Path $projectRoot "backend/photocull/assets/rating_model_v1.json"),
    (Join-Path $projectRoot "lightroom/YidianPhotoCull.lrplugin/manifest.json"),
    (Join-Path $projectRoot "models/dinov2_small.onnx"),
    (Join-Path $projectRoot "models/scene_mobilenetv2.onnx"),
    (Join-Path $projectRoot "models/open-closed-eye.onnx"),
    (Join-Path $projectRoot "models/facial_expression_mobilefacenet.onnx"),
    (Join-Path $projectRoot "models/ediffiqa_tiny.onnx"),
    (Join-Path $projectRoot "models/yolov8n.onnx"),
    (Join-Path $projectRoot "models/person_detection_mediapipe.onnx"),
    (Join-Path $projectRoot "models/osnet_x0_25_msmt17.onnx"),
    (Join-Path $projectRoot "models/pose_landmarker_heavy.task"),
    (Join-Path $projectRoot "models/depth_anything_v2_vitl.onnx"),
    (Join-Path $projectRoot "models/buffalo_l/det_10g.onnx"),
    (Join-Path $projectRoot "models/buffalo_l/w600k_r50.onnx"),
    (Join-Path $projectRoot "models/buffalo_l/2d106det.onnx"),
    (Join-Path $projectRoot "models/buffalo_l/1k3d68.onnx"),
    (Join-Path $projectRoot "models/buffalo_l/meanshape_68.json")
)
foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required packaging resource is missing: $requiredPath"
    }
}

& $python -c "import PyInstaller, onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA build environment is not ready: $python"
}

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($projectRoot))
$requiredFreeBytes = 30GB
if ($drive.AvailableFreeSpace -lt $requiredFreeBytes) {
    throw "Packaging requires at least 30 GiB free; available: $([math]::Round($drive.AvailableFreeSpace / 1GB, 2)) GiB"
}
if (Test-Path -LiteralPath $singleInstallerPath -PathType Leaf) {
    throw "Final installer already exists and will not be overwritten: $singleInstallerPath"
}

Write-Output "Version: $version"
Write-Output "CUDA Python: $python"
Write-Output "Intermediate Setup: $setupPath"
Write-Output "Intermediate volumes: $(Join-Path $releaseRoot $volumePattern)"
Write-Output "Final single installer: $singleInstallerPath"
Write-Output "Required free space: 30 GiB"
Write-Output "Available free space: $([math]::Round($drive.AvailableFreeSpace / 1GB, 2)) GiB"

if ($PreflightOnly) {
    Write-Output "V021_PACKAGE_PREFLIGHT_OK"
    exit 0
}

Push-Location $projectRoot
try {
    Invoke-NativeChecked -FilePath "npm.cmd" -Arguments @("run", "build") -Label "Renderer build"
    Invoke-NativeChecked -FilePath "npm.cmd" -Arguments @("run", "build:backend:cuda") -Label "CUDA backend build"
    Invoke-NativeChecked -FilePath $electronBuilder -Arguments @("--win", "dir", "--publish", "never") -Label "Electron directory package"
    & (Join-Path $projectRoot "scripts/build-cuda-split-installer.ps1")
    & (Join-Path $projectRoot "scripts/verify-release.ps1") -RequireCuda
}
finally {
    Pop-Location
}

$volumes = @(Get-ChildItem -LiteralPath $releaseRoot -File -Filter $volumePattern | Sort-Object Name)
if (-not (Test-Path -LiteralPath $setupPath -PathType Leaf) -or $volumes.Count -lt 2) {
    throw "v0.2.1 intermediate payload is incomplete."
}

Write-Output "V021_INTERMEDIATE_RELEASE_OK"
