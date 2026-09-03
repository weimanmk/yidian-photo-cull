param(
    [Parameter(Mandatory = $true)][string]$PayloadDirectory,
    [Parameter(Mandatory = $true)][string]$OutputPath,
    [Parameter(Mandatory = $true)][string]$RunProgram,
    [Parameter(Mandatory = $true)][string]$SfxModulePath,
    [Parameter(Mandatory = $true)][string]$SevenZipPath,
    [ValidateRange(0, 9)][int]$CompressionLevel = 0
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description is missing: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

if (-not (Test-Path -LiteralPath $PayloadDirectory -PathType Container)) {
    throw "Payload directory is missing: $PayloadDirectory"
}
if ([string]::IsNullOrWhiteSpace($RunProgram) -or $RunProgram.IndexOfAny(@('"', "`r", "`n")) -ge 0) {
    throw "RunProgram must be a non-empty relative file name without quotes or line breaks."
}
if ([System.IO.Path]::IsPathRooted($RunProgram)) {
    throw "RunProgram must be relative to the payload directory."
}

$payloadRoot = (Resolve-Path -LiteralPath $PayloadDirectory).Path
$launcherPath = [System.IO.Path]::GetFullPath((Join-Path $payloadRoot $RunProgram))
if (-not $launcherPath.StartsWith($payloadRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not (Test-Path -LiteralPath $launcherPath -PathType Leaf)) {
    throw "RunProgram is not a payload file: $RunProgram"
}

$sfxModule = Resolve-RequiredFile -Path $SfxModulePath -Description "SFX module"
$sevenZip = Resolve-RequiredFile -Path $SevenZipPath -Description "7-Zip executable"
$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $outputFullPath
if (-not $outputDirectory) {
    throw "OutputPath must include a parent directory."
}
if (Test-Path -LiteralPath $outputFullPath) {
    throw "Output already exists: $outputFullPath"
}

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$workRoot = Join-Path $outputDirectory (".sfx-work-" + [guid]::NewGuid().ToString("N"))
$archivePath = Join-Path $workRoot "payload.7z"
$configPath = Join-Path $workRoot "config.txt"
$stagedOutput = Join-Path $workRoot "installer.exe"

try {
    New-Item -ItemType Directory -Path $workRoot -Force | Out-Null

    Push-Location $payloadRoot
    try {
        & $sevenZip a -t7z "-mx=$CompressionLevel" -mmt=on -ms=off $archivePath ".\*"
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

    $escapedRunProgram = $RunProgram.Replace("\", "\\")
    $config = [string]::Join("`r`n", @(
        ";!@Install@!UTF-8!",
        "Title=`"一点筛图 CUDA 完整离线安装器`"",
        "Progress=`"yes`"",
        "RunProgram=`"$escapedRunProgram`"",
        ";!@InstallEnd@!",
        ""
    ))
    [System.IO.File]::WriteAllText($configPath, $config, [System.Text.UTF8Encoding]::new($false))

    $outputStream = [System.IO.File]::Create($stagedOutput)
    try {
        foreach ($part in @($sfxModule, $configPath, $archivePath)) {
            $inputStream = [System.IO.File]::OpenRead($part)
            try {
                $inputStream.CopyTo($outputStream, 8MB)
            }
            finally {
                $inputStream.Dispose()
            }
        }
    }
    finally {
        $outputStream.Dispose()
    }

    Move-Item -LiteralPath $stagedOutput -Destination $outputFullPath
    Write-Output $outputFullPath
}
finally {
    $resolvedWorkRoot = [System.IO.Path]::GetFullPath($workRoot)
    $resolvedOutputDirectory = [System.IO.Path]::GetFullPath($outputDirectory)
    if ($resolvedWorkRoot.StartsWith($resolvedOutputDirectory + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -and
        $resolvedWorkRoot -ne $resolvedOutputDirectory -and
        (Test-Path -LiteralPath $resolvedWorkRoot)) {
        Remove-Item -LiteralPath $resolvedWorkRoot -Recurse -Force
    }
}
