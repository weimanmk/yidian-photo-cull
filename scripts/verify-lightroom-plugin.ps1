param(
    [switch]$DryRun,
    [string]$SmokeRoot = "",
    [string]$CatalogPath = "",
    [string]$SourcePath = "",
    [string]$BridgeRoot = "",
    [string]$PluginPath = "",
    [string]$ReceiptPath = "",
    [int]$SampleCount = 5
)

$ErrorActionPreference = "Stop"

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $resolvedRoot = Get-FullPath $Root
    $resolvedCandidate = Get-FullPath $Candidate
    $prefix = $resolvedRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedCandidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must stay under the supplied smoke root: $resolvedCandidate"
    }
    return $resolvedCandidate
}

function Get-Sha256Text {
    param([Parameter(Mandatory = $true)][string]$Value)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
    }
    finally {
        $sha.Dispose()
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$defaultSmokeRoot = Join-Path $projectRoot "output/lightroom-smoke"
$resolvedSmokeRoot = Get-FullPath $(if ($SmokeRoot) { $SmokeRoot } else { $defaultSmokeRoot })
$resolvedCatalogPath = Assert-ChildPath -Root $resolvedSmokeRoot -Candidate $(if ($CatalogPath) { $CatalogPath } else { Join-Path $resolvedSmokeRoot "YidianSmoke.lrcat" }) -Label "Catalog path"
$resolvedSourcePath = Assert-ChildPath -Root $resolvedSmokeRoot -Candidate $(if ($SourcePath) { $SourcePath } else { Join-Path $resolvedSmokeRoot "source" }) -Label "Source path"
$resolvedBridgeRoot = Get-FullPath $(if ($BridgeRoot) { $BridgeRoot } else { Join-Path $env:APPDATA "Adobe/Lightroom/YidianPhotoCull/lightroom-bridge" })
$resolvedPluginPath = Get-FullPath $(if ($PluginPath) { $PluginPath } else { Join-Path $env:APPDATA "Adobe/Lightroom/Modules/YidianPhotoCull.lrplugin" })
$resolvedReceiptPath = Assert-ChildPath -Root $resolvedSmokeRoot -Candidate $(if ($ReceiptPath) { $ReceiptPath } else { Join-Path $resolvedSmokeRoot "receipt.json" }) -Label "Receipt path"

if ([System.IO.Path]::GetExtension($resolvedCatalogPath) -ne ".lrcat") {
    throw "Catalog path must end with .lrcat"
}
if ($SampleCount -ne 5) {
    throw "The disposable mixed-state smoke plan requires exactly 5 samples."
}
if (-not (Test-Path -LiteralPath (Join-Path $resolvedPluginPath "manifest.json") -PathType Leaf)) {
    throw "Plugin manifest is missing: $resolvedPluginPath"
}
$pluginManifestPath = Join-Path $resolvedPluginPath "manifest.json"
$pluginManifest = [System.IO.File]::ReadAllText($pluginManifestPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
if ($pluginManifest.plugin_id -ne "com.yidian.photocull.lightroom" -or $pluginManifest.version -ne "0.2.1") {
    throw "Plugin manifest must be com.yidian.photocull.lightroom version 0.2.1."
}

Write-Output "Smoke root: $resolvedSmokeRoot"
Write-Output "Catalog path: $resolvedCatalogPath"
Write-Output "Source path: $resolvedSourcePath"
Write-Output "Bridge root: $resolvedBridgeRoot"
Write-Output "Plugin path: $resolvedPluginPath"
Write-Output "Receipt path: $resolvedReceiptPath"
Write-Output "Sample count: $SampleCount"

if ($DryRun) {
    Write-Output "LIGHTROOM_DRY_RUN_OK"
    exit 0
}

if ($resolvedSmokeRoot -ne (Get-FullPath $defaultSmokeRoot)) {
    throw "Live verification is restricted to the project output/lightroom-smoke root."
}
if (-not (Test-Path -LiteralPath $resolvedSourcePath -PathType Container)) {
    throw "Generated source directory is missing: $resolvedSourcePath"
}
$samples = @(Get-ChildItem -LiteralPath $resolvedSourcePath -File | Where-Object { $_.Extension -match "^\.jpe?g$" })
if ($samples.Count -ne $SampleCount) {
    throw "Expected $SampleCount generated JPG samples, found $($samples.Count)."
}
if (-not (Test-Path -LiteralPath $resolvedReceiptPath -PathType Leaf)) {
    throw "Final receipt is missing: $resolvedReceiptPath"
}

$receipt = [System.IO.File]::ReadAllText($resolvedReceiptPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
if ($receipt.status -ne "complete") {
    throw "Final receipt status is not complete: $($receipt.status)"
}
if ([int]$receipt.counts.pending_rating -ne 0) {
    throw "Final receipt still has pending ratings."
}
if ([int]$receipt.counts.verified -ne ([int]$receipt.counts.new + [int]$receipt.counts.update)) {
    throw "Final receipt verification invariant failed."
}
$receiptJson = $receipt | ConvertTo-Json -Depth 20 -Compress
if ($receiptJson -match '"source_path"') {
    throw "Redacted receipt contains an absolute source path field."
}

$archiveRequest = Join-Path $resolvedBridgeRoot ("archive/" + [string]$receipt.request_id + ".json")
if (-not (Test-Path -LiteralPath $archiveRequest -PathType Leaf)) {
    throw "Archived execute request is missing: $archiveRequest"
}
$manifest = [System.IO.File]::ReadAllText($archiveRequest, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
if ($manifest.plan_hash -ne $receipt.plan_hash -or $manifest.operation_id -ne $receipt.operation_id) {
    throw "Archived request does not match the final receipt."
}
foreach ($item in $manifest.items) {
    $source = Assert-ChildPath -Root $resolvedSourcePath -Candidate ([string]$item.source_path) -Label "Manifest source"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Manifest source is missing: $source"
    }
    $file = Get-Item -LiteralPath $source
    # Windows PowerShell 5.1 lacks DateTimeOffset.UnixEpoch; use its fixed tick value.
    $unixEpochTicks = [int64]621355968000000000
    $modifiedNs = (([int64]([DateTimeOffset]$file.LastWriteTimeUtc).Ticks) - $unixEpochTicks) * 100
    if ([int64]$item.file_size -ne $file.Length -or [int64]$item.modified_ns -ne $modifiedNs) {
        throw "Source fingerprint drifted: $source"
    }
    $normalized = $source.ToLowerInvariant()
    if ([string]$item.path_hash -ne (Get-Sha256Text $normalized)) {
        throw "Source path hash mismatch: $source"
    }
}

Write-Output "LIGHTROOM_RECEIPT_OK"
