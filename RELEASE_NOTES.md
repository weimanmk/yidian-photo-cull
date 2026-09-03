# 一点筛图 v0.2.0

Windows 本地 AI 筛图 CUDA 增强版。

## 引擎 0.9.0 增量

- NVIDIA 设备优先使用 `CUDAExecutionProvider`，CUDA 成品随包携带 CUDA 12 与 cuDNN 9 运行库；不可用时保留 DirectML/CPU 代码回退。
- 修复 cuDNN 9 引擎子库只预加载入口 DLL、首次卷积仍回退 CPU 的问题。
- 设置页显示模型实际 Provider，并区分首次推理前的配置状态，避免把静默 CPU 回退误报成 GPU。
- 场景、人脸、人体与景深 ONNX 模型均可使用 CUDA；BlazePose GHUM Heavy 继续使用 LiteRT XNNPACK CPU。
- CUDA 完整资源超过传统 NSIS 单文件容量后，改为“小型安装器 + 多个本地分卷”；所有文件放在同一目录即可离线安装。
- 修复极端姿态假阳性 ROI 触发 Pillow 巨幅画布保护的问题，并把特征缓存签名升级到 0.9.0，避免复用修复前结果。

- 同场景、连拍与近重复图片聚类
- SCRFD/ArcFace 人脸身份 + YOLOv8n/OSNet 远景与背脸人物软匹配
- OpenVINO 睁闭眼 + OpenCV FER 七类表情 + eDifFIQA 人脸质量 + 原图清晰度复核
- InsightFace 3D68 精细头部姿态，并修正 106 点关键点的官方 0–255 输入预处理
- BlazePose GHUM Heavy 33 点人体 3D 动作描述；直接使用 LiteRT，本地运行且不经过 Tasks 遥测路径
- Depth Anything V2 Large 相对景深，辅助过滤背景海报假人体、判断主体合焦与背景分离
- 可选“人物环节保底”：优先按一级子目录、否则按 EXIF 时间分段，确保可靠已识别人物不会在某环节被全部当废片过滤
- 保底片会随精选导出，但保留闭眼/模糊/曝光警告，结果页可用“保底”标签单独复核
- 清晰度、闭眼、表情、曝光、构图和动作综合评分
- 无登录、无会员、无额度、无遥测
- 照片与人脸向量仅在本机处理
- 不包含 Qwen3.8-27B、GGUF、mmproj、llama.cpp、训练集或测试照片

注意：随私有成品附带的 InsightFace `buffalo_l` 人脸权重仅限非商业研究使用。商业发布前需取得相应授权或更换模型。

人体检测默认使用 AGPL-3.0 的 Ultralytics YOLOv8n；MediaPipe Apache-2.0 模型作为回退。若未来商业分发，需要按实际分发方式重新进行许可证评估。

本版本尚未使用商业代码签名证书，首次运行时 Windows 可能显示 SmartScreen 提示。
