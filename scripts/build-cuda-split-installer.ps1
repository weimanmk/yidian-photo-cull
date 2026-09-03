param(
    [string]$AppDirectory = "",
    [string]$PackedArchive = "",
    [int]$VolumeSizeMiB = 1900,
    [ValidateRange(0, 9)][int]$CompressionLevel = 1,
    [switch]$ReuseVolumes
)

$ErrorActionPreference = "Stop"

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        return -join ($sha256.ComputeHash($stream) | ForEach-Object { $_.ToString("X2") })
    }
    finally {
        $stream.Dispose()
        if ($sha256) { $sha256.Dispose() }
    }
}

function Find-PackagingTool {
    param(
        [Parameter(Mandatory = $true)][string]$CacheRoot,
        [Parameter(Mandatory = $true)][string]$Filename
    )

    $tool = Get-ChildItem -LiteralPath $CacheRoot -Recurse -File -Filter $Filename -ErrorAction SilentlyContinue |
        Sort-Object FullName |
        Select-Object -First 1
    if (-not $tool) {
        throw "Packaging tool is missing: $Filename under $CacheRoot"
    }
    return $tool.FullName
}

function Split-PackedArchive {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$DestinationBase,
        [Parameter(Mandatory = $true)][long]$PartBytes
    )

    $sourceStream = [System.IO.File]::OpenRead($Source)
    try {
        $buffer = New-Object byte[] (8MB)
        $partNumber = 1
        while ($sourceStream.Position -lt $sourceStream.Length) {
            $partPath = "{0}.{1:D3}" -f $DestinationBase, $partNumber
            $partStream = [System.IO.File]::Create($partPath)
            try {
                $written = 0L
                while ($written -lt $PartBytes -and $sourceStream.Position -lt $sourceStream.Length) {
                    $remaining = [Math]::Min([long]$buffer.Length, $PartBytes - $written)
                    $read = $sourceStream.Read($buffer, 0, [int]$remaining)
                    if ($read -le 0) { break }
                    $partStream.Write($buffer, 0, $read)
                    $written += $read
                }
            }
            finally {
                $partStream.Dispose()
            }
            $partNumber += 1
        }
    }
    finally {
        $sourceStream.Dispose()
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseRoot = Join-Path $projectRoot "release"
$appRoot = if ($AppDirectory) { (Resolve-Path -LiteralPath $AppDirectory).Path } else { Join-Path $releaseRoot "win-unpacked" }
$package = Get-Content -LiteralPath (Join-Path $projectRoot "package.json") -Raw | ConvertFrom-Json
$version = [string]$package.version
$fileVersion = if (($version -split '\.').Count -eq 3) { "$version.0" } else { $version }
$archiveBaseName = "一点筛图-$version-Windows-CUDA.7z"
$archiveBase = Join-Path $releaseRoot $archiveBaseName
$installer = Join-Path $releaseRoot "一点筛图-$version-Windows-Setup.exe"
$installerIcon = Join-Path $releaseRoot ".icon-ico/icon.ico"
$cacheRoot = Join-Path $env:LOCALAPPDATA "electron-builder/Cache"
$sevenZip = Find-PackagingTool -CacheRoot $cacheRoot -Filename "7za.exe"
$makeNsis = Find-PackagingTool -CacheRoot $cacheRoot -Filename "makensis.exe"

if (-not (Test-Path -LiteralPath (Join-Path $appRoot "一点筛图.exe") -PathType Leaf)) {
    throw "Packaged application is missing from $appRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $appRoot "resources/backend/_internal/onnxruntime/capi/onnxruntime_providers_cuda.dll") -PathType Leaf)) {
    throw "Packaged application does not contain the CUDA Execution Provider."
}
if (-not (Test-Path -LiteralPath $installerIcon -PathType Leaf)) {
    throw "Installer icon is missing: $installerIcon"
}
if ($VolumeSizeMiB -le 0 -or $VolumeSizeMiB -ge 1908) {
    throw "VolumeSizeMiB must be between 1 and 1907 so every release asset stays below 2 GB."
}
if ($ReuseVolumes -and $PackedArchive) {
    throw "ReuseVolumes and PackedArchive cannot be used together."
}

if (-not $ReuseVolumes) {
    Get-ChildItem -LiteralPath $releaseRoot -File -Filter "$archiveBaseName.*" -ErrorAction SilentlyContinue |
        Remove-Item -Force

    if ($PackedArchive) {
        $sourceArchive = (Resolve-Path -LiteralPath $PackedArchive).Path
        Split-PackedArchive -Source $sourceArchive -DestinationBase $archiveBase -PartBytes ([long]$VolumeSizeMiB * 1MB)
    }
    else {
        Push-Location $appRoot
        try {
            & $sevenZip a -t7z "-mx=$CompressionLevel" -mmt=on -ms=off "-v${VolumeSizeMiB}m" $archiveBase ".\*"
            if ($LASTEXITCODE -ne 0) {
                throw "7-Zip failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }
}

$volumes = @(Get-ChildItem -LiteralPath $releaseRoot -File -Filter "$archiveBaseName.*" | Sort-Object Name)
if ($volumes.Count -lt 2) {
    throw "Expected at least two CUDA archive volumes, found $($volumes.Count)."
}
foreach ($volume in $volumes) {
    if ($volume.Length -ge 2GB) {
        throw "Archive volume exceeds the 2 GB release-asset limit: $($volume.Name)"
    }
}

& $sevenZip t $volumes[0].FullName
if ($LASTEXITCODE -ne 0) {
    throw "CUDA split archive validation failed with exit code $LASTEXITCODE"
}

$nsisScript = Join-Path $PSScriptRoot "cuda-split-installer.nsi"
& $makeNsis `
    "/INPUTCHARSET" `
    "UTF8" `
    "/DSEVENZIP_EXE=$sevenZip" `
    "/DOUTPUT_FILE=$installer" `
    "/DARCHIVE_BASE=$archiveBaseName" `
    "/DARCHIVE_VOLUME_COUNT=$($volumes.Count)" `
    "/DAPP_VERSION=$version" `
    "/DFILE_VERSION=$fileVersion" `
    "/DICON_FILE=$installerIcon" `
    $nsisScript
if ($LASTEXITCODE -ne 0) {
    throw "NSIS split installer failed with exit code $LASTEXITCODE"
}

$artifacts = @((Get-Item -LiteralPath $installer)) + $volumes
$manifest = foreach ($artifact in $artifacts) {
    [ordered]@{
        name = $artifact.Name
        bytes = $artifact.Length
        sha256 = (Get-Sha256 -Path $artifact.FullName).ToLowerInvariant()
    }
}
$manifestPath = Join-Path $releaseRoot "SHA256SUMS-CUDA.txt"
$manifest | ForEach-Object { "$($_.sha256)  $($_.name)" } |
    Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM

$receipt = [ordered]@{
    product = "一点筛图"
    version = $version
    format = "NSIS bootstrapper with local split 7z payload"
    app_directory = $appRoot
    volume_size_mib = $VolumeSizeMiB
    compression_level = $CompressionLevel
    reused_existing_volumes = [bool]$ReuseVolumes
    archive_volume_count = $volumes.Count
    artifacts = $manifest
}
$receipt | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $releaseRoot "CUDA-PACKAGE-RECEIPT.json") -Encoding utf8NoBOM

Write-Host "CUDA split installer ready: $installer"
foreach ($artifact in $artifacts) {
    Write-Host "  $($artifact.Name): $($artifact.Length) bytes"
}
Write-Host "Checksums: $manifestPath"
