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

`open_v2_role` 与 `open_v2_emotion` 互相独立：任一为 `false` 时对应模型不会加载，也不占用显存。

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
