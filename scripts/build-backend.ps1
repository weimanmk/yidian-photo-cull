param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$configuredPython = if ($PythonPath) { $PythonPath } elseif ($env:PHOTOCULL_BUILD_PYTHON) { $env:PHOTOCULL_BUILD_PYTHON } else { ".venv/Scripts/python.exe" }
$python = if ([System.IO.Path]::IsPathRooted($configuredPython)) { $configuredPython } else { Join-Path $projectRoot $configuredPython }
$spec = Join-Path $projectRoot "backend/photocull-backend.spec"
$dist = Join-Path $projectRoot "build/backend"
$work = Join-Path $projectRoot "build/pyinstaller"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python 3.12 build environment not found: $python"
}

& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is not installed. Run: uv pip install --python ".venv/Scripts/python.exe" "pyinstaller==6.22.2"'
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --log-level WARN `
    --distpath $dist `
    --workpath $work `
    $spec

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$backendExecutable = Join-Path $dist "photocull-backend/photocull-backend.exe"
if (-not (Test-Path -LiteralPath $backendExecutable)) {
    throw "Backend executable was not generated: $backendExecutable"
}

$item = Get-Item -LiteralPath $backendExecutable
$stream = [System.IO.File]::OpenRead($backendExecutable)
try {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    $hash = -join ($sha256.ComputeHash($stream) | ForEach-Object { $_.ToString("X2") })
}
finally {
    $stream.Dispose()
    if ($sha256) { $sha256.Dispose() }
}
Write-Host "Backend ready: $($item.FullName)"
Write-Host "Backend bytes: $($item.Length)"
Write-Host "Backend SHA256: $hash"
