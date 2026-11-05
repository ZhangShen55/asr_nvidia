# Proposal

## Why

当前 v2 链路使用 `speech_fsmn_vad_zh-cn-16k-common-pytorch`，模型名称和训练定位偏向中文场景。FireRedVAD 声明支持 100 多种语言，因此需要先在本项目真实课堂音频上与现有 VAD 做可重复的分段、稳定性和性能对比，再决定是否迁移。

本 change 只负责验证、报告和迁移决策依据，不直接把 FireRedVAD 切换为生产默认后端。

## What Changes

- 在 conda `asr` 环境中验证 FireRedVAD 的依赖、模型加载、CPU/CUDA 推理和长音频处理能力。
- 递归测试 `/home/zhangs/workspace/asr_nvidia/test/audio_file/音频文件/` 下的全部 WAV 文件；当前基线清单为 45 个 16 kHz、单声道、16 bit WAV，总时长约 37.9 小时。
- 使用相同音频、相同预处理和相同并发条件，对比现有 FunASR FSMN-VAD 与 FireRedVAD 的分段结果。
- 记录分段数量、语音覆盖率、起止边界、过短段、空音频和长音频分块行为。
- 记录单请求和并发场景的 VAD P50/P95、实时率、队列等待、CPU/显存占用及 worker 稳定性。
- 抽样把两套 VAD 结果接入现有 Qwen-ASR v2 链路，验证时间戳、Qwen 请求数量、CAM++/emotion2vec 调度和端到端结果是否受影响。
- 产出对比测试报告，明确 FireRedVAD 是否达到迁移条件；若达到，给出后续独立迁移 change 的配置、兼容、回滚和分阶段发布方案。
- 不修改当前生产默认 VAD，不改变 `/v2/seacraft_asr` 响应协议，不重新设计 CAM++/emotion2vec 并发闸门。

## Capabilities

### New Capabilities

无。本 change 是验证和决策资料，不新增生产能力。

### Modified Capabilities

无。生产行为和现有 OpenSpec capability 不在本 change 内变更。

## Impact

- 影响测试脚本、测试产物和文档报告，主要涉及 VAD worker、VAD 结果解析及 v2 链路观测数据。
- 需要评估 FireRedVAD Python 包、模型权重目录和当前 `asr` 环境（Python 3.11、PyTorch/FunASR）的兼容性，但不在本 change 中强制替换运行时依赖。
- 测试可能产生临时音频分段、日志和指标文件；报告必须记录测试版本、配置、样本清单和失败样本，确保结果可复现。
- 当前已完成的 `2026-10-27-cam-emo-independent-concurrency-gates` 作为架构前置背景，不能因本验证改变其验收结论。
