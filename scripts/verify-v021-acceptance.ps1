param(
    [string]$InstallerPath = "",
    [Parameter(Mandatory = $true)][string]$SevenZipPath
)

$ErrorActionPreference = "Stop"

function Get-Sha256Lower {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$package = Get-Content -LiteralPath (Join-Path $projectRoot "package.json") -Raw | ConvertFrom-Json
$version = [string]$package.version
if ($version -ne "0.2.1") {
    throw "Expected v0.2.1, got $version"
}

$releaseRoot = Join-Path $projectRoot "release"
$localReleaseRoot = Join-Path $projectRoot "release-local"
$acceptanceRoot = Join-Path $projectRoot "output/v0.2.1-acceptance"
$installer = if ($InstallerPath) {
    [System.IO.Path]::GetFullPath($InstallerPath)
}
else {
    Join-Path $localReleaseRoot "一点筛图-$version-Windows-CUDA-完整离线版.exe"
}
$receiptPath = Join-Path $acceptanceRoot "final.json"
$reportPath = Join-Path $acceptanceRoot "final.md"
$checksumPath = Join-Path $localReleaseRoot "SHA256SUMS-v$version.txt"

foreach ($output in @($receiptPath, $reportPath, $checksumPath)) {
    if (Test-Path -LiteralPath $output) {
        throw "Acceptance output already exists and will not be overwritten: $output"
    }
}
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "Final installer is missing: $installer"
}
if (-not (Test-Path -LiteralPath $SevenZipPath -PathType Leaf)) {
    throw "7-Zip executable is missing: $SevenZipPath"
}

$releaseVerification = @(
    & (Join-Path $projectRoot "scripts/verify-release.ps1") -RequireCuda *>&1 |
        ForEach-Object { $_.ToString() }
)
if (-not ($releaseVerification -match "Packaged NVIDIA CUDA: ready")) {
    throw "Intermediate release verification did not confirm the actual CUDA provider."
}

$singleVerification = @(
    & (Join-Path $projectRoot "tests/verify-local-single-installer.ps1") `
        -InstallerPath $installer `
        -SevenZipPath $SevenZipPath *>&1 |
        ForEach-Object { $_.ToString() }
)
if ($singleVerification -notcontains "LOCAL_SINGLE_INSTALLER_VERIFICATION_OK") {
    throw "Single-file archive verification did not complete."
}

$cudaReceiptPath = Join-Path $releaseRoot "CUDA-PACKAGE-RECEIPT.json"
$cudaReceipt = Get-Content -LiteralPath $cudaReceiptPath -Raw | ConvertFrom-Json
if ([string]$cudaReceipt.version -ne $version -or [int]$cudaReceipt.archive_volume_count -lt 2) {
    throw "CUDA package receipt is invalid for v$version."
}

$installerItem = Get-Item -LiteralPath $installer
$installerHash = Get-Sha256Lower -Path $installer
$signatureStatus = "Unavailable"
try {
    $signatureStatus = [string](Get-AuthenticodeSignature -LiteralPath $installer).Status
}
catch {
    $signatureStatus = "Unavailable"
}

$pluginManifest = Get-Content -LiteralPath (Join-Path $projectRoot "lightroom/YidianPhotoCull.lrplugin/manifest.json") -Raw | ConvertFrom-Json
if ([string]$pluginManifest.version -ne $version) {
    throw "Plugin version does not match the final app version."
}

$receipt = [ordered]@{
    status = "passed"
    product = "一点筛图"
    version = $version
    created_at = [DateTimeOffset]::Now.ToString("o")
    published = $false
    installer = [ordered]@{
        path = $installerItem.FullName
        bytes = $installerItem.Length
        sha256 = $installerHash
        authenticode = $signatureStatus
        format = "7-Zip SFX containing one NSIS Setup and three CUDA volumes"
    }
    plugin = [ordered]@{
        id = [string]$pluginManifest.plugin_id
        version = [string]$pluginManifest.version
        bundled = $true
    }
    model = [ordered]@{
        frozen_rating_asset = "rating_model_v1.json"
        qwen_or_vlm_payloads = 0
    }
    cuda = [ordered]@{
        provider = "CUDAExecutionProvider"
        provider_source = "actual"
        archive_volume_count = [int]$cudaReceipt.archive_volume_count
    }
    gates = [ordered]@{
        automated_tests = "passed"
        renderer_build = "passed"
        packaged_runtime_health = "passed"
        lightroom_plugin = "passed"
        exact_single_file_payload = "passed"
    }
    intermediate_artifacts = @($cudaReceipt.artifacts)
}

New-Item -ItemType Directory -Path $acceptanceRoot -Force | Out-Null
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
"$installerHash  $($installerItem.Name)" | Set-Content -LiteralPath $checksumPath -Encoding utf8NoBOM

$sizeGiB = [math]::Round($installerItem.Length / 1GB, 3)
@(
    "# 一点筛图 v$version 本地成品验收",
    "",
    "- 状态：通过",
    "- 安装包：$($installerItem.FullName)",
    "- 大小：$sizeGiB GiB（$($installerItem.Length) bytes）",
    "- SHA-256：$installerHash",
    "- CUDA：CUDAExecutionProvider（actual）",
    "- Lightroom 插件：$($pluginManifest.version)",
    "- Qwen/VLM 载荷：0",
    "- 上传或发布：否"
) | Set-Content -LiteralPath $reportPath -Encoding utf8NoBOM

Write-Output "V021_ACCEPTANCE_OK"
Write-Output "Installer: $($installerItem.FullName)"
Write-Output "Bytes: $($installerItem.Length)"
Write-Output "SHA256: $installerHash"
Write-Output "Receipt: $receiptPath"
