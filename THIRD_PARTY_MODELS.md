# 一点筛图 v0.2.1 第三方模型与许可

本文对应 `v0.2.1` Windows 发行包。实际打包清单以 `package.json` 中的 `build.extraResources` 为准；模型下载和校验由 `scripts/setup-models.ps1`、`scripts/setup-face-models.ps1` 与 `scripts/verify-release.ps1` 共同约束。

第三方模型权重的条款可能与模型代码库的开源许可不同，始终以上游最新的模型卡、许可文件和使用条款为准。下列内容是项目的技术盘点，不构成法律意见。

## 发行包摘要

| 模型 | 发行状态 | 主要条款 |
|---|---|---|
| DINOv2 Small | 已打包 | Apache-2.0 |
| OpenCV Zoo MobileNetV2 | 已打包 | Apache-2.0 |
| OpenVINO open-closed-eye-0001 | 已打包 | Apache-2.0 |
| OpenCV Zoo MobileFaceNet FER | 已打包 | Apache-2.0 |
| OpenCV Zoo eDifFIQA-Tiny | 已打包 | CC-BY-4.0 |
| Ultralytics YOLOv8n | 已打包 | AGPL-3.0；商业条款另行授权 |
| OpenCV Zoo MediaPipe person detector | 已打包 | Apache-2.0 |
| OSNet x0.25 MSMT17 | 已打包 | MIT |
| MediaPipe BlazePose GHUM Heavy | 已打包 | 代码/SDK 为 Apache-2.0；模型包按 Google 上游条款 |
| Depth Anything V2 Large | 已打包 | 权重 CC-BY-NC-4.0；转换代码 Apache-2.0 |
| InsightFace buffalo_l 选定权重 | 已打包 | 公开预训练模型仅限非商业学术研究 |
| Qwen / GGUF / mmproj / llama.cpp | 未打包 | 不属于标准发行版 |

## DINOv2 Small

- 文件：`models/dinov2_small.onnx`
- 基础模型：[`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small)
- ONNX 转换：[`Xenova/dinov2-small/onnx/model.onnx`](https://huggingface.co/Xenova/dinov2-small)
- 上游许可：[Apache License 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE)
- SHA-256：`83141175EC78B4FF9A2BB58A4C7C264BA0054D1C2E122E5A8114B79A8D4179EA`
- 用途：场景、连拍和近重复分组的主图像向量。
- 回退：加载失败时依次使用 MobileNetV2 和手工感知特征。

## OpenCV Zoo MobileNetV2

- 文件：`models/scene_mobilenetv2.onnx`
- 上游：[`opencv/opencv_zoo/models/image_classification_mobilenet`](https://github.com/opencv/opencv_zoo/tree/main/models/image_classification_mobilenet)
- 许可：[Apache License 2.0](https://github.com/opencv/opencv_zoo/blob/main/LICENSE)
- SHA-256：`C0C3F76D93FA3FD6580652A45618618A220FCED18BABF65774ED169DE0432AD5`
- 用途：DINOv2 不可用时的本地场景语义描述子。

## OpenVINO open-closed-eye-0001

- 文件：`models/open-closed-eye.onnx`
- 上游：[`openvinotoolkit/open_model_zoo/models/public/open-closed-eye-0001`](https://github.com/openvinotoolkit/open_model_zoo/tree/2022.1.0/models/public/open-closed-eye-0001)
- 上游仓库许可：[Apache License 2.0](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/LICENSE)
- SHA-256：`4DAA100034482525A26C9AFB9297C16580A531189E66E3D2B2AC7D32BECFD593`
- 用途：与 InsightFace 106 点眼部几何融合，判断睁闭眼并用于组内排序。
- 回退：用户另行安装的 `eye_state_mobilenet.onnx`，最后回退到关键点几何。

## OpenCV Zoo MobileFaceNet 表情识别

- 文件：`models/facial_expression_mobilefacenet.onnx`
- 上游：[`opencv/opencv_zoo/models/facial_expression_recognition`](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition)
- 许可：[Apache License 2.0](https://github.com/opencv/opencv_zoo/blob/main/LICENSE)
- SHA-256：`4F61307602FC089CE20488A31D4E4614E3C9753A7D6C41578C854858B183E1A9`
- 用途：七类人脸表情概率，与嘴部关键点和眼睛状态融合，只用于组内优选。

## OpenCV Zoo eDifFIQA-Tiny

- 文件：`models/ediffiqa_tiny.onnx`
- 上游：[`opencv/opencv_zoo/models/face_image_quality_assessment_ediffiqa`](https://github.com/opencv/opencv_zoo/tree/main/models/face_image_quality_assessment_ediffiqa)
- 模型许可：[Creative Commons Attribution 4.0 International](https://github.com/opencv/opencv_zoo/blob/main/models/face_image_quality_assessment_ediffiqa/LICENSE)
- SHA-256：`9426C899CC0F01665240CB7D9E7F98E18E24E456C178326C771A43DA289BFC6A`
- 用途：对齐人脸的感知质量。评分与原分辨率人脸/眼区清晰度、眼睛状态、可见性和 3D 姿态融合。
- 回退：不可用时恢复确定性人脸质量公式；不使用固定分数直接淘汰。

## Ultralytics YOLOv8n

- 文件：`models/yolov8n.onnx`
- 基础模型：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)；项目使用可复现 ONNX 导出镜像 `Kalray/yolov8`
- 上游条款：[AGPL-3.0 或 Ultralytics 企业授权](https://docs.ultralytics.com/help/licensing/)
- SHA-256：`65158DAD735BE799C2466FA15E260C09558080BD530B42A8D0C3D1B419AFD8B5`
- 用途：远景、侧身和背身人物的主检测器。ONNX 镜像来源不改变基础模型的上游条款。

## OpenCV Zoo MediaPipe person detector

- 文件：`models/person_detection_mediapipe.onnx`
- 上游：[`opencv/opencv_zoo/models/person_detection_mediapipe`](https://github.com/opencv/opencv_zoo/tree/main/models/person_detection_mediapipe)
- 模型目录许可：[Apache License 2.0](https://github.com/opencv/opencv_zoo/blob/main/models/person_detection_mediapipe/LICENSE)
- SHA-256：`47FD5599D6FA17608F03E0EB0AE230BAA6E597D7E8A2C8199FE00ABEA55A701F`
- 用途：YOLOv8n 无法加载时的人体检测回退。

## OSNet x0.25 MSMT17

- 文件：`models/osnet_x0_25_msmt17.onnx`
- 架构：[`KaiyangZhou/deep-person-reid`](https://github.com/KaiyangZhou/deep-person-reid) 的 OSNet x0.25
- ONNX 转换：[`anriha/osnet_x0_25_msmt17`](https://huggingface.co/anriha/osnet_x0_25_msmt17)
- 仓库/模型卡标示许可：MIT
- SHA-256：`E78604F4CCDA49B8F41CD0F8F7303800CE75D2361895EBB0729513C1BF53D277`
- 用途：可靠人脸缺失时的服装/人体外观软证据，永不覆盖可靠 ArcFace 身份冲突。

## MediaPipe BlazePose GHUM Heavy

- 文件：`models/pose_landmarker_heavy.task`
- 上游：[Google MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- 代码与 SDK 许可：[Apache License 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
- SHA-256：`64437AF838A65D18E5BA7A0D39B465540069BC8AAE8308DE3E318AAD31FCBC7B`
- 用途：33 点图像骨架和 GHUM 世界坐标，用于本地 3D 动作描述子；不用于人脸身份。
- 运行方式：应用使用 `ai-edge-litert` 直接读取 task 包内的检测与关键点 TFLite，不初始化 MediaPipe Tasks SDK。
- 分发提示：Apache-2.0 明确覆盖上游代码/SDK；重新分发 task 模型包前仍应核对 Google 当前模型条款。

## Depth Anything V2 Large

- 文件：`models/depth_anything_v2_vitl.onnx`
- 基础模型：[`DepthAnything/Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) 的 ViT-L 相对深度权重
- ONNX 转换：[`fabio-sim/Depth-Anything-ONNX` v2.0.0](https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0)
- 许可：Base/Large/Giant 权重为 [CC-BY-NC-4.0](https://github.com/DepthAnything/Depth-Anything-V2#license)；ONNX 转换代码为 Apache-2.0
- SHA-256：`36AB02FFA2094C74E00EC3FD85CA6641811A91AC4F494A5812D193A3764A359B`
- 用途：前后景几何、主体合焦辅助、背景分离，以及抑制海报人物/舞台图案的姿态误检。
- 运行方式：NVIDIA 版优先 ONNX Runtime CUDA，其他 Windows 环境可用 DirectML，最终回退 CPU。输出是相对场景几何，并非光学合焦结论。

## InsightFace buffalo_l 选定权重

`v0.2.1` Windows 发行包包含 InsightFace 官方 `buffalo_l` 模型包中的四个 ONNX 文件，以及用于 3D68 姿态转换的均值脸数据。

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `models/buffalo_l/det_10g.onnx` | SCRFD 人脸检测 | `5838F7FE053675B1C7A08B633DF49E7AF5495CEE0493C7DCF6697200B85B5B91` |
| `models/buffalo_l/w600k_r50.onnx` | ArcFace 512 维身份向量 | `4C06341C33C2CA1F86781DAB0E829F88AD5B64BE9FBA56E56BC9EBDEFC619E43` |
| `models/buffalo_l/2d106det.onnx` | 106 点二维人脸关键点 | `F001B856447C413801EF5C42091ED0CD516FCD21F2D6B79635B1E733A7109DBF` |
| `models/buffalo_l/1k3d68.onnx` | 68 点三维人脸与头部姿态 | `DF5C06B8A0C12E422B2ED8947B8869FAA4105387F199C477AF038AA01F9A45CC` |
| `models/buffalo_l/meanshape_68.json` | 3D68 均值脸变换数据 | `FB89D44A02F280E2CF07BA899FE340E09F4BDFE1D3C94E503A534D8D429EB71E` |

InsightFace 的[官方授权说明](https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md)将公开预训练模型限定为非商业学术研究用途，商业使用需单独授权。源码仓库不提交这些 ONNX 二进制；建模和发行脚本只在显式传入 `-AcceptNonCommercialLicense` 后下载并按 SHA-256 验证。安装资源中同时附带 `models/INSIGHTFACE_MODEL_NOTICE.txt`。

## 未随标准发行版分发

`v0.2.1` 不包含 Qwen 权重、GGUF、mmproj、llama.cpp 或其他视觉大模型运行时，打包应用也会强制关闭该集成。源码中保留的本地 VLM 接口只用于连接用户自行部署的 `127.0.0.1` / `localhost` 兼容服务；用户应自行核对所选模型和运行时的许可。
