# 本地视觉大模型接入

## 为什么不让大模型直接筛全库

大模型擅长判断表情、动作、互动、构图和瞬间，但不应取代人脸身份、原图焦点、闭眼、曝光和相似度测量。PhotoCull 的固定顺序是：

```text
专业小模型提取事实
  → 同场景/同人物分组
  → 硬问题过滤
  → 通用技术排序
  → 只将难分胜负的少数组交给 VLM
  → 严格校验后更新软排序
```

VLM 不可用、超时、返回非法 JSON、使用未知照片 ID 或置信度低于门槛时，原技术排序保持不变。

## 本机推荐配置

已检测的目标机器为 RTX 4070 Laptop 8GB、约 48GB 内存。Qwen3.8-27B 官方 GGUF 仓库中，`Q8_0` 主模型约 28.6GB，加上 mmproj、KV cache、系统和应用内存后不适合该机器。默认建议：

```text
Model: Qwen3.8-27B Q4_K_M（约 19GB）
mmproj: 与 Qwen3.8-27B 严格匹配的 BF16/F16 或 Q8_0
Context: 8192
GPU layers: 16
Parallel requests: 1
Host: 127.0.0.1
Port: 8768
Max candidates per group: 4
Max reviewed groups per run: 12
```

不能将其他 Qwen 型号的 mmproj 与 Qwen3.8-27B 主模型混用。

这两个限制来自目标机器上的真实推理结果：Q4_K_M、16 层 GPU offload、4 张候选图时，首轮 5 个疑难组平均约 230 秒/组。默认 12 组约需 46 分钟；需要更大批次时可手动提高，但不建议直接恢复到 80 组（理论耗时约 5.1 小时）。相同候选与 Prompt 版本会命中 SQLite 缓存，不会重复推理。

## 配置方式

1. 用户自行安装支持 CUDA 和 multimodal 的新版 `llama-server.exe`。
2. 用户自行下载 Qwen3.8-27B GGUF 主模型与匹配 mmproj。
3. 在“AI 设置 → 视觉大模型组内复核”中填入三个路径。
4. 保持服务地址为 `http://127.0.0.1:8768`，启用复核并保存设置。

如果已在 LM Studio 或其他本地 OpenAI 兼容服务中载入了视觉模型，可仅填本地服务地址和模型 ID，将三个托管路径留空。后端不会停止由其他程序启动的外部本地服务。

官方参考：[Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[ggml-org GGUF 与 mmproj](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)、[llama.cpp 多模态文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)。

## 输入与安全约束

- 候选照片先合成一张联系表，每格显示照片 ID、技术分和最多 3 张主人脸裁剪。
- 不向 VLM 提供人名，也不允许 VLM 猜测身份。
- 只允许 `http://127.0.0.1` 或 `http://localhost` 服务地址；`0.0.0.0`、局域网和公网地址会被设置校验拒绝。
- 输出由 JSON schema 约束，应用边界再校验 ID 白名单、完整排名、唯一名次和最佳照片一致性。
- 大模型决策连同模型、量化、Prompt 版本和技术事实一起写入 SQLite 缓存。

## 验收

模型安装后不应直接宣称准确率提升。应先对 `活动A` 和 `活动C` 重跑，并与现有人工“已修”对照：

- 组内 Top-1 / Top-2 人工命中；
- 大模型改选的组数与真实改善率；
- 闭眼、失焦、缺人的错误翻盘数，目标必须为 0；
- 平均单组耗时、总耗时、显存峰值与失败回退数。
