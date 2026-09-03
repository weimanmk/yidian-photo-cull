param(
    [string]$ReleaseDirectory = "",
    [switch]$RequireCuda
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

$projectRoot = Split-Path -Parent $PSScriptRoot
$package = Get-Content -LiteralPath (Join-Path $projectRoot "package.json") -Raw | ConvertFrom-Json
$version = [string]$package.version
if ($version -ne "0.2.1") {
    throw "Expected v0.2.1, got $version"
}
$releaseRoot = if ($ReleaseDirectory) { Resolve-Path -LiteralPath $ReleaseDirectory } else { Join-Path $projectRoot "release" }
$unpacked = Join-Path $releaseRoot "win-unpacked"
$rendererEntry = Join-Path $projectRoot "dist/index.html"

if (-not (Test-Path -LiteralPath $rendererEntry)) {
    throw "Renderer entry is missing: $rendererEntry"
}
$rendererHtml = Get-Content -LiteralPath $rendererEntry -Raw
if ($rendererHtml -match '(?i)(?:src|href)="/') {
    throw "Renderer contains root-relative assets that cannot load through Electron file URLs."
}

$installerPath = Join-Path $releaseRoot "一点筛图-$version-Windows-Setup.exe"
$installer = Get-Item -LiteralPath $installerPath -ErrorAction SilentlyContinue

if (-not $installer) {
    throw "Windows installer not found in $releaseRoot"
}

$appExecutable = Get-ChildItem -LiteralPath $unpacked -Filter "*.exe" -File |
    Where-Object { $_.Name -notmatch "(?i)(uninstall|elevate)" } |
    Select-Object -First 1
if (-not $appExecutable) {
    throw "Packaged application executable not found in $unpacked"
}

$requiredPaths = @(
    "resources/backend/photocull-backend.exe",
    "resources/models/dinov2_small.onnx",
    "resources/models/scene_mobilenetv2.onnx",
    "resources/models/open-closed-eye.onnx",
    "resources/models/facial_expression_mobilefacenet.onnx",
    "resources/models/ediffiqa_tiny.onnx",
    "resources/models/yolov8n.onnx",
    "resources/models/person_detection_mediapipe.onnx",
    "resources/models/osnet_x0_25_msmt17.onnx",
    "resources/models/pose_landmarker_heavy.task",
    "resources/models/depth_anything_v2_vitl.onnx",
    "resources/models/buffalo_l/det_10g.onnx",
    "resources/models/buffalo_l/w600k_r50.onnx",
    "resources/models/buffalo_l/2d106det.onnx",
    "resources/models/buffalo_l/1k3d68.onnx",
    "resources/models/buffalo_l/meanshape_68.json",
    "resources/models/INSIGHTFACE_MODEL_NOTICE.txt",
    "resources/THIRD_PARTY_MODELS.md",
    "resources/lightroom/YidianPhotoCull.lrplugin/Info.lua",
    "resources/lightroom/YidianPhotoCull.lrplugin/manifest.json",
    "resources/backend/_internal/photocull/assets/rating_model_v1.json"
)

foreach ($relativePath in $requiredPaths) {
    $path = Join-Path $unpacked $relativePath
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required release file is missing: $relativePath"
    }
}

$pluginManifestPath = Join-Path $unpacked "resources/lightroom/YidianPhotoCull.lrplugin/manifest.json"
$pluginManifest = Get-Content -LiteralPath $pluginManifestPath -Raw | ConvertFrom-Json
if ($pluginManifest.plugin_id -ne "com.yidian.photocull.lightroom" -or [string]$pluginManifest.version -ne $version) {
    throw "Packaged Lightroom plugin does not match app version $version."
}
$pluginSource = Get-ChildItem -LiteralPath (Split-Path -Parent $pluginManifestPath) -File -Filter "*.lua" |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }
foreach ($forbiddenPluginToken in @(
    "LrHttp",
    "createCollection",
    "createCollectionSet",
    "executeCommand",
    "powershell",
    "cmd.exe",
    "moveToTrash"
)) {
    if (($pluginSource -join "`n").Contains($forbiddenPluginToken)) {
        throw "Packaged Lightroom plugin contains forbidden capability: $forbiddenPluginToken"
    }
}

if ($RequireCuda) {
    $requiredCudaPaths = @(
        "resources/backend/_internal/nvidia/cublas/bin/cublas64_12.dll",
        "resources/backend/_internal/nvidia/cuda_runtime/bin/cudart64_12.dll",
        "resources/backend/_internal/nvidia/cudnn/bin/cudnn64_9.dll",
        "resources/backend/_internal/nvidia/cudnn/bin/cudnn_engines_tensor_ir64_9.dll"
    )
    foreach ($relativePath in $requiredCudaPaths) {
        if (-not (Test-Path -LiteralPath (Join-Path $unpacked $relativePath) -PathType Leaf)) {
            throw "Required CUDA release file is missing: $relativePath"
        }
    }
}

$expectedModelHashes = @{
    "resources/models/dinov2_small.onnx" = "83141175EC78B4FF9A2BB58A4C7C264BA0054D1C2E122E5A8114B79A8D4179EA"
    "resources/models/scene_mobilenetv2.onnx" = "C0C3F76D93FA3FD6580652A45618618A220FCED18BABF65774ED169DE0432AD5"
    "resources/models/open-closed-eye.onnx" = "4DAA100034482525A26C9AFB9297C16580A531189E66E3D2B2AC7D32BECFD593"
    "resources/models/facial_expression_mobilefacenet.onnx" = "4F61307602FC089CE20488A31D4E4614E3C9753A7D6C41578C854858B183E1A9"
    "resources/models/ediffiqa_tiny.onnx" = "9426C899CC0F01665240CB7D9E7F98E18E24E456C178326C771A43DA289BFC6A"
    "resources/models/yolov8n.onnx" = "65158DAD735BE799C2466FA15E260C09558080BD530B42A8D0C3D1B419AFD8B5"
    "resources/models/person_detection_mediapipe.onnx" = "47FD5599D6FA17608F03E0EB0AE230BAA6E597D7E8A2C8199FE00ABEA55A701F"
    "resources/models/osnet_x0_25_msmt17.onnx" = "E78604F4CCDA49B8F41CD0F8F7303800CE75D2361895EBB0729513C1BF53D277"
    "resources/models/pose_landmarker_heavy.task" = "64437AF838A65D18E5BA7A0D39B465540069BC8AAE8308DE3E318AAD31FCBC7B"
    "resources/models/depth_anything_v2_vitl.onnx" = "36AB02FFA2094C74E00EC3FD85CA6641811A91AC4F494A5812D193A3764A359B"
    "resources/models/buffalo_l/det_10g.onnx" = "5838F7FE053675B1C7A08B633DF49E7AF5495CEE0493C7DCF6697200B85B5B91"
    "resources/models/buffalo_l/w600k_r50.onnx" = "4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43"
    "resources/models/buffalo_l/2d106det.onnx" = "F001B856447C413801EF5C42091ED0CD516FCD21F2D6B79635B1E733A7109DBF"
    "resources/models/buffalo_l/1k3d68.onnx" = "DF5C06B8A0C12E422B2ED8947B8869FAA4105387F199C477AF038AA01F9A45CC"
    "resources/models/buffalo_l/meanshape_68.json" = "FB89D44A02F280E2CF07BA899FE340E09F4BDFE1D3C94E503A534D8D429EB71E"
}

foreach ($entry in $expectedModelHashes.GetEnumerator()) {
    $path = Join-Path $unpacked $entry.Key
    $actual = Get-Sha256 -Path $path
    if ($actual -ne $entry.Value) {
        throw "Model checksum mismatch: $($entry.Key). Expected $($entry.Value), got $actual"
    }
}

$normalizedUnpacked = ([System.IO.Path]::GetFullPath($unpacked)).TrimEnd('\', '/')
$forbidden = Get-ChildItem -LiteralPath $unpacked -File -Recurse | Where-Object {
    $relative = $_.FullName.Substring($normalizedUnpacked.Length).TrimStart('\', '/').Replace('\', '/')
    $_.Name -match "(?i)(qwen|mmproj|llama-server|\.gguf$|\.safetensors$)" -or
        $relative -match "(?i)(^|/)(torch|musiq|qualiclip)(/|$)"
}
if ($forbidden) {
    throw "Forbidden model or dataset files found: $($forbidden.FullName -join ', ')"
}

$cudaVolumes = @()
if ($RequireCuda) {
    $installerStem = [System.IO.Path]::GetFileNameWithoutExtension($installer.Name)
    $releaseAssetPrefix = $installerStem -replace "-Windows-Setup$", ""
    if ($releaseAssetPrefix -eq $installerStem) {
        throw "CUDA installer name does not match the expected *-Windows-Setup.exe format: $($installer.Name)"
    }
    $cudaVolumePattern = "$releaseAssetPrefix-Windows-CUDA.7z.*"
    $cudaVolumes = @(
        Get-ChildItem -LiteralPath $releaseRoot -File -Filter $cudaVolumePattern |
            Sort-Object Name
    )
}
if ($cudaVolumes.Count -gt 0) {
    if ($installer.Length -lt 400KB) {
        throw "CUDA bootstrap installer is unexpectedly small: $($installer.Length) bytes"
    }
    if ($cudaVolumes.Count -lt 2) {
        throw "CUDA split package is incomplete: expected at least two volumes."
    }
    foreach ($volume in $cudaVolumes) {
        if ($volume.Length -ge 2GB) {
            throw "CUDA volume exceeds 2 GB: $($volume.Name)"
        }
    }
}
elseif ($installer.Length -lt 50MB) {
    throw "Installer is unexpectedly small: $($installer.Length) bytes"
}

$smokeRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("yidian-release-smoke-" + [guid]::NewGuid().ToString("N"))
$backendExecutable = Join-Path $unpacked "resources/backend/photocull-backend.exe"
$modelDirectory = Join-Path $unpacked "resources/models"
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$smokePort = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()
$smokeToken = [guid]::NewGuid().ToString("N")
$backendProcess = $null

try {
    New-Item -ItemType Directory -Force -Path $smokeRoot | Out-Null
    $env:PHOTOCULL_PORT = [string]$smokePort
    $env:PHOTOCULL_API_TOKEN = $smokeToken
    $env:PHOTOCULL_DISABLE_VLM = "1"
    $env:PHOTOCULL_MODEL_DIR = $modelDirectory
    $env:PHOTOCULL_DATA_DIR = Join-Path $smokeRoot "data"
    $stdout = Join-Path $smokeRoot "stdout.log"
    $stderr = Join-Path $smokeRoot "stderr.log"
    $backendProcess = Start-Process -FilePath $backendExecutable -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $health = $null
    for ($attempt = 0; $attempt -lt 60; $attempt += 1) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$smokePort/api/health" -Headers @{ "X-PhotoCull-Token" = $smokeToken } -TimeoutSec 2
            break
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $health) {
        throw "Packaged backend did not become healthy. stderr: $(Get-Content -Raw $stderr -ErrorAction SilentlyContinue)"
    }
    if (
        -not $health.offline `
        -or -not $health.scene_ai.available `
        -or -not $health.face_ai.available `
        -or -not $health.face_ai.eye_model.available `
        -or -not $health.face_ai.expression_model.available `
        -or -not $health.face_ai.landmark_3d_model.available `
        -or -not $health.face_ai.face_quality_model.available `
        -or -not $health.body_ai.detector.available `
        -or -not $health.body_ai.reid_model.available `
        -or -not $health.pose_ai.available `
        -or -not $health.pose_ai.world_coordinates `
        -or $health.pose_ai.telemetry `
        -or -not $health.depth_ai.available `
        -or -not $health.depth_ai.relative_depth `
        -or -not $health.depth_ai.local_only
    ) {
        throw "Packaged AI health check failed: $($health | ConvertTo-Json -Depth 8 -Compress)"
    }
    if ($health.vlm_ai.enabled -or $health.vlm_ai.model_id -ne "disabled") {
        throw "Packaged build did not fully disable VLM: $($health.vlm_ai | ConvertTo-Json -Depth 5 -Compress)"
    }
    if ($RequireCuda -and (
        $health.version -ne $version `
        -or $health.depth_ai.backend -ne "CUDAExecutionProvider" `
        -or $health.depth_ai.provider_source -ne "actual" `
        -or $health.depth_ai.cuda_preload_error
    )) {
        throw "Packaged CUDA health check failed: $($health | ConvertTo-Json -Depth 8 -Compress)"
    }
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$smokePort/api/shutdown" -Headers @{ "X-PhotoCull-Token" = $smokeToken } -TimeoutSec 3 | Out-Null
    $backendProcess.WaitForExit(5000) | Out-Null
    if (-not $backendProcess.HasExited) {
        throw "Packaged backend did not exit after the shutdown request."
    }
}
finally {
    if ($backendProcess -and -not $backendProcess.HasExited) {
        $backendProcess.Kill()
        $backendProcess.WaitForExit()
    }
    if (Test-Path -LiteralPath $smokeRoot) {
        Remove-Item -LiteralPath $smokeRoot -Recurse -Force
    }
}

$hash = Get-Sha256 -Path $installer.FullName
$checksumPath = Join-Path $releaseRoot $(if ($cudaVolumes.Count -gt 0) { "SHA256SUMS-CUDA.txt" } else { "SHA256SUMS.txt" })
(@($installer) + $cudaVolumes) | ForEach-Object {
    "$((Get-Sha256 -Path $_.FullName).ToLowerInvariant())  $($_.Name)"
} | Set-Content -LiteralPath $checksumPath -Encoding utf8NoBOM

$signatureStatus = "Unavailable"
try {
    $signatureStatus = (Get-AuthenticodeSignature -LiteralPath $installer.FullName).Status
}
catch {
    Write-Warning "Authenticode status could not be read: $($_.Exception.Message)"
}
Write-Host "Release verified: $($installer.FullName)"
Write-Host "Installer bytes: $($installer.Length)"
Write-Host "Installer SHA256: $hash"
Write-Host "Authenticode status: $signatureStatus"
Write-Host "Qwen/GGUF/mmproj/llama-server payloads: none"
Write-Host "Packaged scene AI: ready"
Write-Host "Packaged face AI: ready"
Write-Host "Packaged body AI: ready"
Write-Host "Packaged 3D pose AI: ready (raw LiteRT, telemetry disabled)"
Write-Host "Packaged depth AI: ready (Depth Anything V2 Large, local only)"
if ($RequireCuda) {
    Write-Host "Packaged NVIDIA CUDA: ready (CUDA 12 + cuDNN 9, actual depth inference session)"
    if ($cudaVolumes.Count -gt 0) {
        Write-Host "CUDA split payloads: $($cudaVolumes.Count) volumes, each below 2 GB"
    }
}
