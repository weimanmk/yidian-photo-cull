param(
    [switch]$AcceptNonCommercialLicense,
    [string]$SourceDirectory = ""
)

$ErrorActionPreference = "Stop"

if (-not $AcceptNonCommercialLicense) {
    throw "InsightFace buffalo_l models are limited to non-commercial research use. Re-run with -AcceptNonCommercialLicense only after accepting those terms."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelRoot = Join-Path $projectRoot "models"
$targetDirectory = Join-Path $modelRoot "buffalo_l"
$downloadUrl = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
$required = @(
    @{ Name = "det_10g.onnx"; Bytes = 16923827; Sha256 = "5838F7FE053675B1C7A08B633DF49E7AF5495CEE0493C7DCF6697200B85B5B91" },
    @{ Name = "w600k_r50.onnx"; Bytes = 174383860; Sha256 = "4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43" },
    @{ Name = "2d106det.onnx"; Bytes = 5030888; Sha256 = "F001B856447C413801EF5C42091ED0CD516FCD21F2D6B79635B1E733A7109DBF" },
    @{ Name = "1k3d68.onnx"; Bytes = 143607619; Sha256 = "DF5C06B8A0C12E422B2ED8947B8869FAA4105387F199C477AF038AA01F9A45CC" }
)

function Test-RequiredModels {
    param([Parameter(Mandatory = $true)][string]$Directory)

    foreach ($model in $required) {
        $path = Join-Path $Directory $model.Name
        if (-not (Test-Path -LiteralPath $path)) { return $false }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -ne $model.Bytes) { return $false }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $model.Sha256) { return $false }
    }
    return $true
}

if (Test-RequiredModels -Directory $targetDirectory) {
    Write-Host "Verified existing InsightFace model subset: $targetDirectory"
    exit 0
}

$source = $null
if ($SourceDirectory) {
    $candidate = Resolve-Path -LiteralPath $SourceDirectory
    $nested = Join-Path $candidate.Path "buffalo_l"
    $source = if (Test-Path -LiteralPath $nested) { $nested } else { $candidate.Path }
    if (-not (Test-RequiredModels -Directory $source)) {
        throw "Source directory does not contain the verified buffalo_l files: $source"
    }
}

$temporaryRoot = $null
if (-not $source) {
    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("yidian-face-models-" + [guid]::NewGuid().ToString("N"))
    $archive = Join-Path $temporaryRoot "buffalo_l.zip"
    $expanded = Join-Path $temporaryRoot "expanded"
    New-Item -ItemType Directory -Force -Path $expanded | Out-Null
    Write-Host "Downloading official InsightFace buffalo_l model pack..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $expanded
    $candidateDirectories = @((Get-Item -LiteralPath $expanded)) + @(Get-ChildItem -LiteralPath $expanded -Directory -Recurse)
    $candidate = $candidateDirectories | Where-Object { Test-RequiredModels -Directory $_.FullName } | Select-Object -First 1
    if (-not $candidate) {
        throw "Downloaded archive does not contain the expected verified buffalo_l files."
    }
    $source = $candidate.FullName
}

New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
foreach ($model in $required) {
    Copy-Item -LiteralPath (Join-Path $source $model.Name) -Destination (Join-Path $targetDirectory $model.Name) -Force
}

if (-not (Test-RequiredModels -Directory $targetDirectory)) {
    throw "Copied InsightFace models failed integrity verification."
}

Write-Host "Installed and verified InsightFace model subset: $targetDirectory"
foreach ($model in $required) {
    Write-Host "$($model.Name): $($model.Sha256)"
}

if ($temporaryRoot -and (Test-Path -LiteralPath $temporaryRoot)) {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
}
