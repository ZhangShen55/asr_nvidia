# Proposal

## Why

`/v2/seacraft_asr` 的并发瓶颈不在 Qwen 服务侧，而在本服务的 VAD 阶段：压测中 VAD 占端到端耗时的 83.2%，c4→c32 并发扩大 8 倍时纯转写吞吐从 26.1 段/s 反而略降到 24.5 段/s。

专项实测定位到根因是 **Python GIL 而非算力**：同进程内把 VAD 实例从 1 个加到 16 个，总吞吐始终横在 176~236× 实时、单请求耗时却线性放大 21 倍；而多进程 8 worker 可达 1079× 实时，是同进程的 5.4 倍。同时实测证明 **CPU 与 GPU 的 VAD 分段边界逐段完全一致**（三份完整音频分别切出 753 / 947 / 703 段，全部逐段吻合），说明 VAD 可以安全下沉到 CPU，并把腾出的 GPU 全部留给 Qwen、CAM++ 与 emotion2vec。

## What Changes

**VAD 设备可配置，默认下沉到 CPU**

- 新增 `[qwen_asr].vad_device`，取值为 `"cpu"`（默认）或 `"cuda:0"` 形式的设备串；VAD 的落卡不再与 CAM++ / emotion2vec 绑定。
- 移除 `[model_device].vad`（其职责由 `vad_device` 接管）；`[model_device].emotion2vec` 与 `[model_device].speech_campplus` 保持不变，默认 GPU 0。
- `core/models.py` 的 `_resolve_device()` 在 CUDA 不可用时直接抛错，VAD 走 CPU 需要一条绕过该校验的设备解析路径。

**VAD 由独立进程池承载**

- 新增常驻 `vad_pool_size`（默认 8）个 VAD worker 进程，每个进程预热持有一份 FSMN-VAD 实例，各自拥有独立 GIL。
- 一次音频的 VAD 调用整体交给一个 worker 执行；无空闲 worker 时请求进入等待队列，不阻塞服务主事件循环。
- worker 之间不共享模型状态；每次调用使用独立 cache。

**按排队深度自适应扩缩容**

- 等待队列深度超过 `vad_queue_watermark`（默认 4）时临时拉起 worker，最多到 `vad_pool_max`（默认 16）。
- 空闲超过 `vad_collapse_timeout_s`（默认 120 秒）的 worker 被回收，常驻规模回落到 `vad_pool_size`。
- 触发判据是排队深度而非显存：实测 FSMN-VAD 权重仅 1.6 MB、每增一个实例显存增量约 15 MB，显存从来不是并发能力的约束。

**分段临时文件生命周期治理**

- 段级 clip 由「一次性全量落盘、请求结束统一删除」改为「按需惰性创建、转写完成即时删除」。
- 峰值临时文件占用由约 219 MB/请求（943 段 × 238 KB）降至 MB 量级，32 路并发下由约 6.8 GB 降至可忽略量级。
- worker 进程只读主进程已落盘的音频临时文件路径，全程零拷贝、不新增文件；主进程是唯一的文件写入方，清理责任单一。
- VAD 排队超时（默认 `vad_queue_timeout_s` = 120 秒）返回新增错误码 4014。

**进程命名与显存归属**

- VAD worker 进程使用独立进程名 `asr-offline-vad`，便于在进程列表中定位。
- `vad_device = "cpu"` 时 worker 不触碰 CUDA，不出现在 `nvidia-smi` 的 GPU 进程列表中；主进程 `asr-offline-model` 仍承载 CAM++ 与 emotion2vec。
- `vad_device` 指向 GPU 时，每个 worker 各自持有一份 CUDA 上下文，显存随 worker 数量增长，关系显式可预期。

**容器 CPU 配额契约**

- 实测单个 VAD worker 处理 60 分钟音频约占用 2.2~2.5 核（线程数调节无效，并行来自算子层）。
- `vad_pool_size = 8` 时 VAD 聚合约 20 核，主进程预留 4~6 核，容器建议配额 24~32 核；`vad_pool_size = 4` 时建议配额 16 核。

**不在本次范围**

- 不做「整文件切块并行 VAD」：切块位置会影响分段边界，等价性需要另行验证，待进程池上线后若 VAD 仍是瓶颈再单独立项。
- 不调整 Qwen 服务的容器参数（`--api-server-count` 等）：VAD 改造后 Qwen 引擎才会成为新瓶颈，届时另开变更。
- 不构建镜像、不部署上线：本变更只做后端运行、测试与效果比对。

## Capabilities

### New Capabilities

- `vad-segmentation-runtime`: VAD 切分的执行承载与并发治理——设备选择（CPU / GPU）、进程池隔离、等待队列与按排队深度自适应扩缩容、分段临时文件的生命周期与峰值占用。

### Modified Capabilities

- `local-model-runtime`: VAD 不再与 CAM++ / emotion2vec 同进程承载，进程命名与 `nvidia-smi` 可见性规则相应调整。
- `qwen-asr-v2-adapter`: VAD 的落卡配置从 `[model_device].vad` 改为 `[qwen_asr].vad_device`，默认 CPU；`vad_concurrency` 语义由「同进程实例池大小」改为「进程池相关参数」的组成部分。

## Impact

- **代码**：`core/vad.py`（由同进程实例池改为进程池客户端）、新增 `core/vad_pool.py`（worker 进程生命周期与队列）、`core/config.py`（新增 5 个配置项、移除 `model_device_vad`）、`core/models.py`（设备解析绕过 CUDA 校验）、`api/routes/asr_qwen_common.py`（段级 clip 惰性创建与即时删除）、`api/routes/asr_common.py`（清理逻辑）、`main.py`（worker 进程命名）。
- **配置**：`config.toml` / `config.docker.toml` 新增 `vad_device` / `vad_pool_size` / `vad_pool_max` / `vad_queue_watermark` / `vad_collapse_timeout_s` / `vad_queue_timeout_s`，移除 `[model_device].vad`。
- **接口**：`/v2/seacraft_asr` 的响应结构与字段语义不变；新增错误码 4014（VAD 排队超时）。
- **运行环境**：VAD 默认不再占用 GPU 显存；CPU 占用随 `vad_pool_size` 线性增长，容器 CPU 配额需相应放大。
- **部署**：容器需提供足够的 CPU 配额（8 worker 建议 24~32 核），否则 VAD 阶段会出现调度排队。
