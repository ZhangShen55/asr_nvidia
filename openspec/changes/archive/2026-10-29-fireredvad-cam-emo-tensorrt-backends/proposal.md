# Proposal

## Why

CAM++ 与 emotion2vec 的 FP32/FP16 TensorRT 独立实验已证明模型层面的初步可行性，但项目仍只有 PyTorch 推理路径，尚不能通过配置启用这些 engine。本 change 的核心是加入四个独立的 TensorRT 配置项，并按现有 `[model_device_pool]` 的实例设备接入实际推理后端，同时保留后续迁移 NPU 的边界。

## What Changes

- 在 `[model_device_pool]` 增加以下四项，两个模型分别启用和回滚；精度支持 `fp16`、`fp32`，只选择对应 TensorRT engine：

  ```toml
  speech_campplus_use_tensorrt = false
  speech_campplus_tensorrt_precision = "fp16"
  emotion2vec_use_tensorrt = false
  emotion2vec_tensorrt_precision = "fp16"
  ```

- 保留 `speech_campplus_devices`、`emotion2vec_devices` 的数组语义，每个元素对应一个实例；CUDA+对应开关 true 使用 TensorRT，CUDA+false 使用 PyTorch CUDA，NPU 固定使用 `torch_npu`，CPU 使用 PyTorch CPU。
- 通过模型运行器隔离设备上下文、AMP 和推理实现，复用既有独立模型池、懒加载、能力开关、FIFO 租约及等待统计；每个 TensorRT 实例独立管理执行上下文、stream 和缓冲区。
- 复用 `model/<原始模型目录>/tensorrt/{fp32,fp16}/` 下的候选产物，补齐可复现导出、manifest 校验和运行适配。先以 FP32 为数值基准，再验收 FP16；不纳入 INT8。
- 将 engine 失败、实际后端、实际精度及兼容信息纳入状态与日志。启用 TensorRT 后不得因加载失败静默切到 PyTorch；关闭对应开关是显式回滚方式。
- 以 `2026-11-02-split-audio-concurrency-gates` 完成后的实现为基线，保留其 A/B 音频准入、能力错误和取消收尾规则；推理真正完成后才允许归还模型及整音频租约。
- 在固定 VAD、Qwen 负载、音频并发、B 上限、段并发和实例数组下，验证 embedding、角色聚类、完整情绪输出及端到端性能。
- 必测 CAM++ 3 实例、emotion2vec 1 实例、`showSpk=true`、`showEmotion=true` 的八路完整音频场景：从 `test/audio_file/音频文件/` 选择 8 门不同课程，每门使用一份完整 WAV；测试时总音频/B 上限均为 8，对比 PyTorch、TensorRT FP32/FP16，并交付逐音频耗时和端到端 P50/P95。

本 change 不包含 VAD 后端替换、环境音过滤、Qwen 调优或新的并发闸门；现有目录名沿用历史名称，实际范围以本提案及规格为准。

## Capabilities

### New Capabilities

- `local-model-inference-backends`: 定义 CAM++ 和 emotion2vec 按实例设备选择 PyTorch/TensorRT/torch_npu 的规则、engine 生命周期、实际后端观测及独立回滚。

### Modified Capabilities

- `qwen-asr-v2-adapter`: 在既有并行增强链路中接入配置选择的运行器，明确 TensorRT 精度可选，保留 A/B 准入、能力校验与响应契约。
- `container-model-packaging`: 定义可选 TensorRT engine、manifest、来源与兼容信息的交付方式，保留现有 PyTorch 权重。
- `service-operations`: 明确四个参数在 `[model_device_pool]` 下的位置、默认值、设备优先级、配置校验和实际后端状态。

## Impact

- 后续实现主要涉及 `core/config.py`、`core/models.py`、新增模型运行器、`api/routes/asr_qwen_common.py`、状态接口、配置模板、模型说明、依赖与相关测试；本轮仅更新规划。
- TensorRT 运行依赖仅用于相应 CUDA 路径，ONNX 导出依赖与服务运行依赖分开；NPU 路径隔离 CUDA/TensorRT 导入，适配与 CANN 匹配的 PyTorch/torch_npu 环境。
- 已有 RTX 4090 / SM89、TensorRT 10.7.0.post1、CUDA 12.4 产物仅是已验证环境下的候选；其他目标环境需重新核验兼容性或重建。
- 初步实验报告位于 `model/tensorrt-validation-20261102/evidence/REPORT.md`。已有实验不代表服务接入、完整课堂回归或同卡多实例验收完成，也不自动勾选本 change 的实现任务。
