# 本地模型目录

运行时不会自动联网下载模型。

- `scene_mobilenetv2.onnx`：同场景语义特征，来自 OpenCV Zoo 的 MobileNetV2，Apache-2.0。
- `open-closed-eye.onnx`：睁闭眼分类，来自 OpenVINO Open Model Zoo，Apache-2.0。
- `facial_expression_mobilefacenet.onnx`：七类面部表情识别，来自 OpenCV Zoo，Apache-2.0。
- `ediffiqa_tiny.onnx`：对齐人脸感知质量评分，来自 OpenCV Zoo eDifFIQA(T)，CC-BY-4.0。
- `yolov8n.onnx`：远景/背脸人体检测，Ultralytics YOLOv8n，AGPL-3.0。
- `person_detection_mediapipe.onnx`：人体检测回退模型，来自 OpenCV Zoo，Apache-2.0。
- `osnet_x0_25_msmt17.onnx`：人体外观 ReID 软证据，不作为硬身份约束。
- `pose_landmarker_heavy.task`：MediaPipe BlazePose GHUM Heavy，输出 33 个二维关键点与米制世界坐标；通过原始 LiteRT 运行，不使用 Tasks SDK 遥测路径。
- `depth_anything_v2_vitl.onnx`：Depth Anything V2 Large 相对景深；用于前景真人过滤、主体合焦与背景分离，不参与人物身份判定。
- `buffalo_l/`：SCRFD 人脸检测、ArcFace 人物向量、106 点关键点与 3D68 姿态模型；仅限非商业研究使用。
- 标准发行版不包含 Qwen、GGUF、mmproj 或 llama.cpp。

执行 `scripts/setup-models.ps1` 安装场景、睁眼、表情、人脸质量、人体、3D 姿态和景深模型；明确接受 InsightFace 非商业许可后，执行 `scripts/setup-face-models.ps1 -AcceptNonCommercialLicense` 安装人脸模型。
