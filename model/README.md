# 模型目录说明

本目录用于放置离线 ASR 运行所需的模型。模型权重不会提交到 Git，但会参与 Docker 构建并复制到镜像内的 `/app/model/`。

## 必需目录

以下五个目录必须完整存在，目录名不可修改：

| 目录 | 用途 | 配置文件 |
|------|------|----------|
| `speech_fsmn_vad_zh-cn-16k-common-pytorch/` | 语音端点检测（VAD） | `config.toml`、`config.docker.toml` |
| `punc_ct-transformer_cn-en-common-vocab471067-large/` | 中英文标点恢复 | 同上 |
| `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/` | Paraformer 中文语音识别 | 同上 |
| `speech_campplus_sv_zh_en_16k-common_advanced/` | CAM++ 说话人特征与说话人分离 | 同上 |
| `emotion2vec_plus_large/` | emotion2vec 情绪识别 | 同上 |

## 放置要求

1. 将每个模型的全部文件复制到对应目录，包括配置、词表、`.pt` 权重及模型运行所需的附属文件。
2. 不要只复制 `model.pt`，也不要额外增加一层同名子目录。
3. `config.toml` 使用 `./model/<目录名>`，容器配置使用 `/app/model/<目录名>`。
4. 模型文件被 `.gitignore` 忽略；只有本说明文件会纳入版本控制。
5. Docker 构建会检查五个目录是否存在且非空。缺少任一目录时构建失败。

## 模型清单

```text
model/
├── README.md
├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── punc_ct-transformer_cn-en-common-vocab471067-large/
├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
├── speech_campplus_sv_zh_en_16k-common_advanced/
└── emotion2vec_plus_large/
```
