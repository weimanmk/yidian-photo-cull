param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory,
    [string]$UnpackedDirectory = "",
    [string]$OutputDirectory = "",
    [switch]$RequirePose,
    [switch]$RequireCuda,
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $UnpackedDirectory) {
    $UnpackedDirectory = Join-Path $projectRoot "release/win-unpacked"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot "output/package-inference-smoke"
}

$source = (Resolve-Path -LiteralPath $SourceDirectory).Path
$unpacked = (Resolve-Path -LiteralPath $UnpackedDirectory).Path
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Smoke source must be a directory: $source"
}

$backendExecutable = Join-Path $unpacked "resources/backend/photocull-backend.exe"
$modelDirectory = Join-Path $unpacked "resources/models"
if (-not (Test-Path -LiteralPath $backendExecutable -PathType Leaf)) {
    throw "Packaged backend is missing: $backendExecutable"
}
if (-not (Test-Path -LiteralPath $modelDirectory -PathType Container)) {
    throw "Packaged model directory is missing: $modelDirectory"
}

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$runDirectory = Join-Path $OutputDirectory $runId
$dataDirectory = Join-Path $runDirectory "data"
New-Item -ItemType Directory -Force -Path $dataDirectory | Out-Null

$stdoutPath = Join-Path $runDirectory "backend.stdout.log"
$stderrPath = Join-Path $runDirectory "backend.stderr.log"
$receiptPath = Join-Path $runDirectory "receipt.json"
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$smokePort = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()
$smokeToken = [guid]::NewGuid().ToString("N")
$headers = @{ "X-PhotoCull-Token" = $smokeToken }
$backendProcess = $null
$shutdownSent = $false

try {
    $env:PHOTOCULL_PORT = [string]$smokePort
    $env:PHOTOCULL_API_TOKEN = $smokeToken
    $env:PHOTOCULL_DISABLE_VLM = "1"
    $env:PHOTOCULL_MODEL_DIR = $modelDirectory
    $env:PHOTOCULL_DATA_DIR = $dataDirectory

    $backendProcess = Start-Process `
        -FilePath $backendExecutable `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $health = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $health = Invoke-RestMethod `
                -Uri "http://127.0.0.1:$smokePort/api/health" `
                -Headers $headers `
                -TimeoutSec 2
            break
        }
        catch {
            if ($backendProcess.HasExited) {
                throw "Packaged backend exited before health check. See $stderrPath"
            }
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $health) {
        throw "Packaged backend did not become healthy before timeout. See $stderrPath"
    }

    $requestBody = @{
        folder = $source
        grouping_preset = "balanced"
        keep_per_group = 1
        recursive = $false
        coverage_enabled = $true
        coverage_window_minutes = 15
    } | ConvertTo-Json
    Invoke-RestMethod `
        -Method Post `
        -Uri "http://127.0.0.1:$smokePort/api/scan/start" `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $requestBody | Out-Null

    $status = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $status = Invoke-RestMethod `
                -Uri "http://127.0.0.1:$smokePort/api/scan/status" `
                -Headers $headers `
                -TimeoutSec 3
        }
        catch {
            # RAW 首次推理会短时占用计算线程；单次轮询超时不代表后端或模型失败。
            if ($backendProcess.HasExited) {
                throw "Packaged backend exited during inference. See $stderrPath"
            }
            Start-Sleep -Milliseconds 750
            continue
        }
        if ($status.status -in @("completed", "failed", "cancelled")) {
            break
        }
        Start-Sleep -Milliseconds 750
    }
    if (-not $status -or $status.status -ne "completed") {
        $statusJson = if ($status) { $status | ConvertTo-Json -Depth 6 -Compress } else { "no status" }
        throw "Packaged inference did not complete: $statusJson. See $stderrPath"
    }

    $results = Invoke-RestMethod `
        -Uri "http://127.0.0.1:$smokePort/api/scan/results" `
        -Headers $headers `
        -TimeoutSec 20
    $photos = @($results.photos)
    $depthPhotos = @($photos | Where-Object { $_.depth -and $_.depth.model })
    $posePhotos = @($photos | Where-Object { @($_.poses).Count -gt 0 })
    $foregroundPoses = @(
        $photos |
            ForEach-Object { @($_.poses) } |
            Where-Object { $null -ne $_.foreground_score }
    )

    if ($photos.Count -lt 1) {
        throw "Packaged inference did not return any photos."
    }
    if ($depthPhotos.Count -ne $photos.Count) {
        throw "Depth inference is missing for $($photos.Count - $depthPhotos.Count) photo(s)."
    }
    if ($RequirePose -and $posePhotos.Count -lt 1) {
        throw "The selected real-photo fixture did not produce a packaged 3D pose observation."
    }
    if (-not $results.coverage -or -not $results.coverage.enabled) {
        throw "Person-stage coverage guard was not enabled in packaged inference results."
    }
    if ($results.engine.version -ne "0.9.0") {
        throw "Unexpected packaged engine version: $($results.engine.version)"
    }
    $modelBackends = [ordered]@{
        scene = $results.engine.scene_ai.backend
        face = $results.engine.face_ai.backend
        body = $results.engine.body_ai.backend
        depth = $results.engine.depth_ai.backend
    }
    if ($RequireCuda) {
        $nonCuda = @($modelBackends.GetEnumerator() | Where-Object { $_.Value -ne "CUDAExecutionProvider" })
        if ($nonCuda.Count -gt 0) {
            throw "Packaged inference fell back from CUDA: $($modelBackends | ConvertTo-Json -Compress)"
        }
        if ($results.engine.face_ai.provider_source -ne "actual" -or $results.engine.depth_ai.provider_source -ne "actual") {
            throw "Packaged CUDA provider was not confirmed by actual model sessions."
        }
    }
    $coverageRequiredCells = [int]$results.coverage.required_cells
    $coverageAlreadyCoveredCells = [int]$results.coverage.already_covered_cells
    $coverageProtectedCells = [int]$results.coverage.protected_cells
    $coverageProtectedPhotos = [int]$results.coverage.protected_photos
    $coverageUnresolvedCells = [int]$results.coverage.unresolved_cells
    if ($coverageRequiredCells -ne ($coverageAlreadyCoveredCells + $coverageProtectedCells + $coverageUnresolvedCells)) {
        throw "Coverage report is inconsistent: required=$coverageRequiredCells, already=$coverageAlreadyCoveredCells, protected=$coverageProtectedCells, unresolved=$coverageUnresolvedCells"
    }

    $receipt = [ordered]@{
        verified_at = [DateTimeOffset]::Now.ToString("o")
        source_directory = $source
        backend_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $backendExecutable).Hash
        status = $status.status
        total = $results.summary.total
        groups = $results.summary.groups
        selected = $results.summary.selected
        depth_photos = $depthPhotos.Count
        pose_photos = $posePhotos.Count
        foreground_poses = $foregroundPoses.Count
        coverage_enabled = $results.coverage.enabled
        coverage_stages = @($results.coverage.stages).Count
        coverage_required_cells = $coverageRequiredCells
        coverage_already_covered_cells = $coverageAlreadyCoveredCells
        coverage_protected_cells = $coverageProtectedCells
        coverage_protected_photos = $coverageProtectedPhotos
        coverage_unresolved_cells = $coverageUnresolvedCells
        scene_backend = $modelBackends.scene
        face_backend = $modelBackends.face
        body_backend = $modelBackends.body
        depth_backend = $modelBackends.depth
        depth_model = $health.depth_ai.model
        pose_model = $health.pose_ai.model
        engine_version = $results.engine.version
        feature_pipeline_signature = $results.engine.feature_cache.pipeline_signature
        vlm_enabled = $health.vlm_ai.enabled
    }
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding utf8NoBOM
    $receipt | ConvertTo-Json -Depth 8 -Compress

    Invoke-RestMethod `
        -Method Post `
        -Uri "http://127.0.0.1:$smokePort/api/shutdown" `
        -Headers $headers `
        -TimeoutSec 10 | Out-Null
    $shutdownSent = $true
    $backendProcess.WaitForExit(8000) | Out-Null
}
catch {
    Write-Error "$($_.Exception.Message) Diagnostic run preserved at: $runDirectory"
}
finally {
    if ($backendProcess -and -not $backendProcess.HasExited) {
        if (-not $shutdownSent) {
            try {
                Invoke-RestMethod `
                    -Method Post `
                    -Uri "http://127.0.0.1:$smokePort/api/shutdown" `
                    -Headers $headers `
                    -TimeoutSec 3 | Out-Null
                $backendProcess.WaitForExit(5000) | Out-Null
            }
            catch {
                # 只停止本脚本创建的子进程，不枚举或影响其他后端实例。
            }
        }
        if (-not $backendProcess.HasExited) {
            $backendProcess.Kill()
            $backendProcess.WaitForExit()
        }
    }
}
