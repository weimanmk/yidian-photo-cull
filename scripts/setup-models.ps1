$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $projectRoot "models"
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null

function Install-VerifiedModel {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256,
        [Parameter(Mandatory = $true)][long]$MinimumBytes
    )

    $target = Join-Path $modelDirectory $Name
    if (Test-Path -LiteralPath $target) {
        $existingHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($existingHash -eq $ExpectedSha256) {
            Write-Host "Verified existing model: $target"
            return
        }
        throw "Existing model checksum mismatch: $target"
    }

    $temporary = "$target.download"
    try {
        Write-Host "Downloading model: $Name"
        Invoke-WebRequest -Uri $Source -OutFile $temporary
        $item = Get-Item -LiteralPath $temporary
        if ($item.Length -lt $MinimumBytes) {
            throw "Downloaded model is unexpectedly small: $($item.Length) bytes"
        }
        $hash = Get-FileHash -LiteralPath $temporary -Algorithm SHA256
        if ($hash.Hash -ne $ExpectedSha256) {
            throw "Model checksum mismatch. Expected $ExpectedSha256, got $($hash.Hash)"
        }
        Move-Item -LiteralPath $temporary -Destination $target
        Write-Host "Installed: $target"
        Write-Host "SHA256: $($hash.Hash)"
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

Install-VerifiedModel `
    -Name "dinov2_small.onnx" `
    -Source "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model.onnx?download=true" `
    -ExpectedSha256 "83141175EC78B4FF9A2BB58A4C7C264BA0054D1C2E122E5A8114B79A8D4179EA" `
    -MinimumBytes 80000000

Install-VerifiedModel `
    -Name "scene_mobilenetv2.onnx" `
    -Source "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx" `
    -ExpectedSha256 "C0C3F76D93FA3FD6580652A45618618A220FCED18BABF65774ED169DE0432AD5" `
    -MinimumBytes 10000000

Install-VerifiedModel `
    -Name "open-closed-eye.onnx" `
    -Source "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/open-closed-eye-0001/open_closed_eye.onnx" `
    -ExpectedSha256 "4DAA100034482525A26C9AFB9297C16580A531189E66E3D2B2AC7D32BECFD593" `
    -MinimumBytes 45000

Install-VerifiedModel `
    -Name "facial_expression_mobilefacenet.onnx" `
    -Source "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx" `
    -ExpectedSha256 "4F61307602FC089CE20488A31D4E4614E3C9753A7D6C41578C854858B183E1A9" `
    -MinimumBytes 4700000

Install-VerifiedModel `
    -Name "ediffiqa_tiny.onnx" `
    -Source "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_image_quality_assessment_ediffiqa/ediffiqa_tiny_jun2024.onnx" `
    -ExpectedSha256 "9426C899CC0F01665240CB7D9E7F98E18E24E456C178326C771A43DA289BFC6A" `
    -MinimumBytes 7000000

Install-VerifiedModel `
    -Name "yolov8n.onnx" `
    -Source "https://huggingface.co/Kalray/yolov8/resolve/9e0af089be9c2f172e4fd9b724805f8b6514854e/yolov8n.onnx?download=true" `
    -ExpectedSha256 "65158DAD735BE799C2466FA15E260C09558080BD530B42A8D0C3D1B419AFD8B5" `
    -MinimumBytes 12000000

Install-VerifiedModel `
    -Name "person_detection_mediapipe.onnx" `
    -Source "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx" `
    -ExpectedSha256 "47FD5599D6FA17608F03E0EB0AE230BAA6E597D7E8A2C8199FE00ABEA55A701F" `
    -MinimumBytes 11000000

Install-VerifiedModel `
    -Name "osnet_x0_25_msmt17.onnx" `
    -Source "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/1e22b925c70caa5591e9fab3f5540af8484bcad8/osnet_x0_25_msmt17.onnx?download=true" `
    -ExpectedSha256 "E78604F4CCDA49B8F41CD0F8F7303800CE75D2361895EBB0729513C1BF53D277" `
    -MinimumBytes 850000

Install-VerifiedModel `
    -Name "pose_landmarker_heavy.task" `
    -Source "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" `
    -ExpectedSha256 "64437AF838A65D18E5BA7A0D39B465540069BC8AAE8308DE3E318AAD31FCBC7B" `
    -MinimumBytes 30000000

Install-VerifiedModel `
    -Name "depth_anything_v2_vitl.onnx" `
    -Source "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitl.onnx" `
    -ExpectedSha256 "36AB02FFA2094C74E00EC3FD85CA6641811A91AC4F494A5812D193A3764A359B" `
    -MinimumBytes 1300000000
