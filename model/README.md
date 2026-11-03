# 模型目录说明

本目录用于放置离线 ASR 运行所需的本地模型。模型权重不会提交到 Git，但会参与 Docker 构建并复制到镜像内的 `/app/model/`。

`/v1.1.8/seacraft_asr` 已下线，本目录不再包含其专用的 SeACo-Paraformer、PUNC 与 `faster-whisper-large-v3`。

## 必需目录

以下三个目录必须完整存在，目录名不可修改：

| 目录 | 用途 | 配置文件项 | 生效开关 |
|------|------|------------|----------|
| `speech_fsmn_vad_zh-cn-16k-common-pytorch/` | 语音端点检测（VAD） | `[model_paths].vad_model_dir` | 常驻 |
| `speech_campplus_sv_zh_en_16k-common_advanced/` | CAM++ 说话人特征与说话人分离 | `[model_paths].spk_model_dir` | `[features].open_v2_role = true` |
| `emotion2vec_plus_large/` | emotion2vec 情绪识别 | `[model_paths].emotion_model_dir` | `[features].open_v2_emotion = true` |

`open_v2_role` 与 `open_v2_emotion` 是 v3 配置中保留的兼容键，互相独立：任一为 `false` 时对应模型不会加载，也不占用显存。

## Qwen3-ASR 模型不在本目录

Qwen3-ASR 由**独立容器**提供，不进本镜像、不需要 API Key。本服务通过「IP + 8806」访问该容器；`[qwen_asr].model` 留空时从 `/v1/models` 自动发现模型名。

## 放置要求

1. 将每个模型的全部文件复制到对应目录，包括配置、词表、`.pt` 权重及模型运行所需的附属文件。
2. 不要只复制 `model.pt`，也不要额外增加一层同名子目录。
3. `config.toml` 使用 `./model/<目录名>`，容器配置使用 `/app/model/<目录名>`。
4. 模型文件被 `.gitignore` 忽略；只有本说明文件会纳入版本控制。
5. Docker 构建会检查上述三个目录是否存在且非空。缺少任一目录时构建失败并指出缺失目录名。

## 模型清单

```text
model/
├── README.md
├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── speech_campplus_sv_zh_en_16k-common_advanced/
└── emotion2vec_plus_large/
```

## 可选 TensorRT 验证产物

CAM++ 和 emotion2vec 的 FP32、FP16 engine 分别保存在原始模型目录的 `tensorrt/<precision>/engine.plan`，同目录附带 `manifest.json`。原始 PyTorch 权重保留；服务已支持显式配置启用，默认开关为 false。

本批产物针对 RTX 4090 / SM89、TensorRT 10.7.0.post1、CUDA 12.4。早期模型独立实验见 [原始实验交付说明](tensorrt-validation-20261102/README.md)，服务集成验收见下方报告。

## CAM++ / emotion2vec TensorRT 资产

两个模型可在各自目录的 `tensorrt/{fp32,fp16}/` 提供 `engine.plan` 与正式 `manifest.json`，
保留原始 PyTorch 权重。`tensorrt/sources/` 保留导出来源及哈希，加载时检查资产与实际实例环境。
四个配置项、NPU 路由、依赖隔离、输入范围、状态与回滚参见 [后端说明](../docs/TENSORRT.md)。
现有候选验证环境为 RTX 4090 / SM89、TensorRT 10.7.0.post1、CUDA 12.4；不可直接用于 NPU。

八门完整课堂性能、聚类及情绪边界结果见 [服务集成验证报告](../docs/cam-emo-tensorrt-backends-validation-20261102.md)。
FP16 在本批共同输出段的角色划分一致，但情绪存在少量边界标签翻转；FP32 的高等传热学
聚类与 PyTorch FP32 一致、与当前 PyTorch FP16 不同。不能把高 embedding 相似度或小规模标签一致当作无损保证。
