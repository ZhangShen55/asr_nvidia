# Design

## Context

当前 VAD worker 在 [core/vad_pool.py](/home/zhangs/workspace/asr_nvidia/core/vad_pool.py) 中通过 FunASR `AutoModel.generate()` 执行一次完整音频切分，[core/vad.py](/home/zhangs/workspace/asr_nvidia/core/vad.py) 将 FunASR 返回值转换为 `VadSegment(start_s, end_s)`。下游 Qwen-ASR、CAM++ 和 emotion2vec 只应消费统一的 `VadSegment`，不应感知 VAD 后端。

测试目录下目标范围包含 45 个 WAV，均为 16 kHz、单声道、16 bit PCM，总时长约 136,420 秒。音频以中文课堂录音为主，不能仅凭公开多语言基准推断 FireRedVAD 在本项目上的效果。

## Goals / Non-Goals

**Goals:**

- 在 `asr` 环境确认 FireRedVAD 模型、依赖、CPU/CUDA 加载和长音频处理可行。
- 对目标目录下全部 WAV 建立可复现的 VAD 对比清单和结果数据。
- 对比分段数量、语音覆盖、边界误差、空音频、过短段和异常退出。
- 在代表性样本上验证两套分段接入现有 v2 链路后的时间戳、Qwen 请求数、本地模型等待和端到端延迟。
- 形成明确的迁移门槛、配置方式、灰度步骤和回滚方案。

**Non-Goals:**

- 本 change 不把 FireRedVAD 接入生产默认路径。
- 本 change 不修改 CAM++/emotion2vec 模型池或 `segment_concurrency` 语义。
- 本 change 不使用 AAC 文件，不把现有 JSON 自动当作人工 VAD 标注。
- 本 change 不在没有实测证据时修改 FireRedVAD 阈值或承诺多语言效果。

## Decisions

### 1. 使用独立对比驱动，保持生产 VAD 不变

测试驱动分别调用现有 FunASR VAD 和 FireRedVAD，并将结果规范化为同一结构：`[(start_s, end_s)]`。这样可以隔离模型差异，避免为了测试临时改变线上 worker。直接修改生产 `core/vad_pool.py` 会把依赖安装、后端适配和业务回归混在一起，保留到后续迁移 change。

### 2. FireRedVAD 采用非流式模型作为首轮对照

当前接口处理完整离线音频，首轮使用 FireRedVAD 的非流式 `FireRedVad.detect()`。其返回的秒级时间戳统一转换为 `VadSegment` 的秒级边界后再比较。流式模型、AED 事件识别和在线增量接口不在本次范围内。

### 3. 全量 VAD、分层全链路

45 个 WAV 全部执行两套 VAD，确保报告覆盖完整目标目录。全链路测试按时长和内容分层抽取代表样本，至少覆盖短/中/长音频、连续讲话、长静音和噪声场景；全链路只用于验证下游影响，不用其结果替代全量 VAD 结论。

### 4. 先比较默认参数，再单独记录调参结果

第一轮使用 FireRedVAD 官方建议默认值，和当前 FunASR 的现行配置进行公平比较。若默认值不满足中文课堂样本，再进行独立阈值 sweep，并在报告中区分“默认结果”和“调参结果”，避免把测试集调参结果误认为泛化能力。

### 5. 以人工标注子集建立绝对指标，以全量建立相对指标

从目标 WAV 按时长、课程和音频特征抽取代表片段，人工标注语音起止区间，计算 speech precision/recall/F1、误报率、漏报率和边界误差。对全量 WAV 计算两套 VAD 的段数、语音总时长、覆盖率、边界差异和过短段比例。现有 JSON 仅作为可选的下游回归参考，不直接作为 VAD 真值。

### 6. 使用可复现的运行和结果清单

测试开始前记录代码提交、模型来源与版本、依赖版本、配置、GPU、CPU、CUDA、文件 SHA256、音频时长和采样格式。每个文件保存单独的 VAD 输出、耗时和错误状态，汇总报告只从这些结果生成。

## Risks / Trade-offs

- [Python 3.11 兼容性未知] → 先在当前 `asr` 环境做导入和最小推理；若官方依赖只支持 Python 3.10，记录阻断原因，不修改生产环境，另行评估独立环境或兼容补丁。
- [FireRedVAD 与当前 VAD 的分段边界不同，可能增加 Qwen/CAM++/emotion2vec 请求数量] → 统计段数、总语音时长和模型池等待，并用同一批样本做端到端对比。
- [没有现成 VAD 真值] → 使用人工标注子集计算绝对指标，同时将全量结果明确标为相对对比，避免虚构准确率。
- [长音频测试耗时较长] → VAD 全量执行但支持断点续跑；每个文件独立落盘结果，单文件失败不影响其他文件。
- [FireRedVAD 模型或依赖下载失败] → 在报告中记录模型来源、校验和与失败日志；验证阶段不替换当前 FunASR 模型。
- [CPU/GPU 实测结论相反] → 分别报告 CPU 和 CUDA 结果，迁移建议以目标部署设备的实测数据为准。

## Migration Plan

本 change 完成后，如果 FireRedVAD 在人工标注子集和全量回归中达到门槛，再创建独立迁移 change：

1. 增加显式 `vad_backend` 选择，默认保持现有 FunASR；复用现有 `vad_model_dir`、`vad_device` 和 VAD worker 池配置。
2. 增加 FireRedVAD 适配器，将其输出转换为统一的 `VadSegment`，不改下游 Qwen/CAM++/emotion2vec 接口。
3. 补充模型目录、依赖锁定、启动检查和状态观测；禁止后端加载失败时静默切换模型。
4. 先以 FireRedVAD 非默认配置进行单机灰度，比较错误率、分段质量、P50/P95 和 GPU/CPU 资源。
5. 灰度通过后再评估是否把默认后端切换为 FireRedVAD；任何异常都可将 `vad_backend` 改回 FunASR 并重启回滚。

## Open Questions

- 人工标注子集的最终数量和标注人员需要在执行前确定；这只影响评估置信度，不改变对比架构。
- FireRedVAD 是否在当前 `asr` Python 3.11 环境长期可用，需要依赖安装和最小推理实测确认。
