# Design

## Context

### 现有 VAD 承载方式

`core/vad.py` 目前的实现是一个同进程实例池：

- 模块级维护 `_vad_instances`（已加载的 FSMN-VAD 实例列表）、`_vad_busy`（占用标记）与 `_vad_pool_lock`；
- `_vad_semaphore = asyncio.Semaphore(settings.qwen_vad_concurrency)`，默认 4，即**同时最多 4 个 VAD 调用**；
- `_acquire_vad_model()` 在池内找不到空闲实例时直接 `_create_vad_model()` 新建一个实例（实例池是懒创建的，`vad_concurrency` 限制的是并发调用数而非实例数）；
- `detect_segments()` 通过 `asyncio.to_thread(model.generate, ...)` 调用，把整个音频文件（45~65 分钟）交给一次 VAD 调用处理；
- 每个调用显式传入独立的 `cache={}`，这是 funasr 的 `FsmnVADStreaming.inference` 使用可变默认参数、必须逐次隔离的内部状态。

设备由 `core/models.py` 的 `_resolve_device(index)` 解析，该函数在 CUDA 不可用时直接抛 `RuntimeError`，因此当前实现**只支持 GPU**。配置项 `[model_device].vad`（默认 0）决定 VAD 落在哪张卡。

### 压测暴露的事实

来自 `docs/qwen-asr-v2-throughput-benchmark-20261023.md`：

| 事实 | 数据 |
|---|---|
| VAD 占端到端耗时 | 83.2%（纯转写模式） |
| c4→c32 并发扩大 8 倍 | 吞吐 26.1 → 24.5 段/s（不涨反跌） |
| 单文件 VAD 实测耗时 | 44~123 s |
| Qwen 引擎能力（4 前端） | 138 段/s，KV 峰值 3.76% |

### 决定设计形状的三组专项实测

**（1）瓶颈是 GIL，不是算力**（`scripts/bench_vad_process.py`，300 s 片段，同机 40 逻辑核）

| 并发 | 同进程线程：总吞吐 | 多进程：总吞吐 |
|---|---|---|
| 1 | 236× 实时 | 267× 实时 |
| 4 | 229× | 857× |
| 8 | 198× | **1079×** |
| 16 | 176× | 1058×（饱和） |

同进程内把实例从 1 加到 16，总吞吐始终横盘，而单请求耗时从 1.27 s 放大到 26.50 s。多进程 8 worker 是同进程方案的 **5.4 倍**。

**（2）CPU 与 GPU 的 VAD 结果完全等价**（`scripts/bench_vad_cpu_vs_gpu.py`，三份完整音频）

| 音频 | 段数 | CPU 墙钟 | GPU 墙钟 | 分段边界 |
|---|---|---|---|---|
| 储能学术英语实践.wav（60 min） | 753 | 11.25 s | 12.03 s | 逐段完全一致 |
| 传播学_马道全.aac | 947 | 12.95 s | 10.47 s | 逐段完全一致 |
| 形式语言与自动机理论_康达周.aac | 703 | 10.46 s | 8.58 s | 逐段完全一致 |

**（3）显存从来不是约束**

FSMN-VAD 权重约 1.6 MB；实测并发加载 1→8 个 GPU 实例，显存从 25,751 MB 涨到 25,857 MB，每增一个实例约 15 MB。扩到 16 个实例增量约 180 MB，相对 48 GiB 的 GPU0 可忽略。

### 每实例 CPU 核数

单个 VAD worker 处理 60 分钟音频：CPU 时间 23~29 s / 墙钟 10.4~13.0 s → **2.2~2.4 核**；8 worker 并发时 2.53 核/实例。线程数调节无效（`OMP_NUM_THREADS=1` → 2.42 核、interop=1 → 2.08 核），并行来自算子层。读音频只占 0.55 s，其余在特征提取与编码器前向。

### 段级临时文件的现状

`api/routes/asr_qwen_common.py` 用一次 `asyncio.gather` 把整份音频的**全部**段 clip 一次性落盘（`extract_audio_clip_from_tensor`，16 kHz 单声道 float32 WAV，实测 3.8 s 段 = 238 KB），直到请求结束才在 `api/routes/asr_v2.py` 的 `finally` 里统一 `cleanup_temp_files`：

| 场景 | 峰值临时文件 |
|---|---|
| 单个 60 分钟请求（943 段） | 219 MB |
| 8 路并发 | 1.7 GB |
| 32 路并发 | 6.8 GB |

## Goals / Non-Goals

**Goals:**

- 把 VAD 从「同进程实例池」迁移到「独立 worker 进程池」，解除 GIL 串行约束。
- 默认让 VAD 运行在 CPU，把 GPU 完整留给 Qwen、CAM++ 与 emotion2vec；同时保留 GPU 运行 VAD 的配置能力。
- 让 VAD 的并发能力具备可预期的 CPU 成本，并给出部署侧的配额契约。
- 把段级临时文件的峰值占用从「与分段总数成正比」降到「与在途分段数成正比」。
- 保持 `/v2/seacraft_asr` 的响应结构、分段语义与时间轴不变。

**Non-Goals:**

- 不做整文件切块并行 VAD（切块边界会影响分段结果，等价性需另行验证；见「Risks」）。
- 不调整 Qwen 服务的容器参数与并发档位（`--api-server-count`、`--max-num-seqs` 等），那是部署侧的另一件事。
- 不改动 VAD 的切分参数语义（`max_single_segment_time`、`max_end_silence_time`、最短段时长过滤）。
- 不构建镜像、不部署上线；本变更只做到后端运行、测试与效果比对。

## Decisions

### 决策 1：VAD 进程池用「常驻 + 按需扩容」的独立进程，而非线程池

**选择**：常驻 `vad_pool_size` 个 worker 进程，每进程持有一份 VAD 实例；等待队列深度超过 `vad_queue_watermark` 时临时拉起额外 worker，上限 `vad_pool_max`；空闲超过 `vad_collapse_timeout_s` 回收。

**理由**：实测表明同进程内无论加多少实例，总吞吐都不变（决策依据见 Context）。真正的扩展只能来自进程级隔离。常驻池保证稳态延迟，弹性扩缩容吸收突发流量。

**替代方案与取舍**：

- *同进程多实例*：零改造成本，但实测吞吐零增长，否决。
- *纯动态创建/销毁*：进程启动需加载模型（约 0.02 s CPU 加载 + 首次推理预热），突发流量下延迟抖动明显；常驻池 + 弹性扩容是折中。
- *固定池不做弹性*：突发流量下排队延迟随并发线性上升。弹性扩容的复杂度主要在「回收时机」这一处逻辑，收益明确，采纳。

### 决策 2：扩容判据是等待队列深度，不是显存占用

**选择**：以「当前等待 VAD 的请求数」作为扩容信号。

**理由**：FSMN-VAD 实例权重仅 1.6 MB，每实例显存增量约 15 MB，显存与并发能力完全解耦。若以显存为判据，触发条件几乎永不成立（等于没有自适应）。队列深度直接对应用户可感知的等待时间。

### 决策 3：VAD 默认在 CPU 上运行，GPU 通过配置切换

**选择**：新增 `[qwen_asr].vad_device`，默认 `"cpu"`；取 `"cuda:0"` 形式时切换到 GPU。移除 `[model_device].vad`。

**理由**：

1. **结果等价**：三份完整音频实测 CPU 与 GPU 的分段边界逐段完全一致，不存在精度差异。
2. **速度相当**：CPU 1.24 s vs GPU 1.29 s（300 s 片段）；整文件下 GPU 快 15%~20%，但两者都受制于 CPU 侧的 Python 开销（GPU 侧仍占 1.7~1.8 核）。
3. **多进程下 GPU 是负资产**：每个 worker 进程各自建立 CUDA context（数百 MB），8 个 worker 就是 8 份 context，且在 GPU 上串行执行；CPU 上则完全无此开销，还把 GPU 腾给 Qwen。
4. **GPU 选项保留**：个别部署方可能希望 VAD 与 Qwen 同卡，或需要该能力用于其他链路，配置项保留即可。

**附带影响**：VAD 不再由主进程承载，`core/models.py` 的 `_resolve_device()` 需要一条不要求 CUDA 可用的设备解析路径；`[model_device]` 保留给 CAM++ 与 emotion2vec。

### 决策 4：worker 与主进程之间只传文件路径，不传音频数据

**选择**：主进程负责把上传音频落盘为临时文件，把路径交给 worker；worker 直接对该路径做 VAD。主进程是唯一的文件写入方。

**理由**：主进程本来就要落盘（`write_audio_bytes_to_temp_file` + `load_audio_tensor`），VAD worker 复用同一份文件即可做到零拷贝、零重复转码。若改成传 PCM 数组，需要额外的序列化与内存拷贝，与「零拷贝」的初衷相反。

**代价**：临时文件的存在期被拉长到整个请求（VAD 阶段 + 全部转写阶段）。这正是决策 5 要解决的问题。

### 决策 5：段级 clip 改为「惰性创建 + 用完即删」

**选择**：段 clip 不再一次性全量落盘，而是在该段即将发起转写时才创建；该段转写返回后立即删除。

**理由**：当前 219 MB/请求的峰值占用来自「全量落盘 + 请求结束统一删除」，与音频分段总数成正比。改成惰性创建后，同时存在的段文件数等于在途转写段数（受 `request_concurrency` 约束，量级为数十），峰值占用降为 MB 量级。

**可验证性**：role / emotion 走的是 numpy 数组（`extract_audio_clip_array`），不依赖段文件，因此删除时机与 role / emotion 的并行计算不冲突——文件只在 Qwen 转写这条路径上被使用。

**兜底**：即使惰性创建，异常路径仍需清理已创建的文件。现有 `api/routes/asr_v2.py` 的 `finally: cleanup_temp_files(ctx.tmp_paths)` 继续兜底；`ctx.tmp_paths` 需要在文件删除后从中摘除，避免重复删除（`cleanup_temp_files` 已吞 `FileNotFoundError`，重复删除无害但日志会变脏）。

### 决策 6：worker 进程与主进程的边界

**选择**：新增 `core/vad_pool.py`，负责 worker 进程的创建、请求分发、结果回收、扩容与回收；`core/vad.py` 保留分段解析、过滤与日志，改为调用进程池客户端。

**理由**：`core/vad.py` 当前混了三件事——实例池管理、funasr 调用、分段解析。进程池化后实例管理逻辑整体搬走，`core/vad.py` 收敛为纯数据处理，边界清晰。

**worker 通信**：主进程与 worker 之间用 `multiprocessing` 的队列传递「音频路径 → 分段结果」。分段结果是毫秒级区间列表，序列化开销远小于音频本身。

### 决策 7：排队超时用独立错误码 4014

**选择**：新增 `vad_queue_timeout_s`（默认 120 秒）与错误码 4014。

**理由**：现有错误码 4001~4013 已占满（4001~4009 为音频与转写类、4010 排队超时、4011 Qwen 不可用、4012 调用失败、4013 队列满）。VAD 排队超时与 Qwen 排队超时的成因不同，需要能被上游区分。

**为什么需要超时**：虽然 `audio_concurrency`（默认 32）已经限制了同时处理的音频数，但在 `vad_pool_size` 较小、音频很长时，VAD 阶段的等待时间仍可能超出上游的耐心。没有超时会让请求无限期挂起。

## Risks / Trade-offs

- **[CPU 配额不足导致调度排队]** → 池规模与容器配额需匹配：8 worker 建议 24~32 核、4 worker 建议 16 核。在配置文档与 `/get_status` 中暴露当前池规模与 CPU 占用，便于运维判断；部署前用真实音频跑一次并发验证。
- **[多进程下模型加载耗时叠加]** → worker 在启动阶段预热模型，且首个请求前服务不进入就绪状态，避免把加载延迟摊给业务请求。
- **[worker 崩溃导致请求悬挂]** → 每个 worker 的调用设置超时，超时或异常时该 worker 被销毁重建，受影响请求返回失败而不是无限等待。
- **[段 clip 惰性创建后仍存在异常残留]** → 惰性创建路径必须把文件登记进 `ctx.tmp_paths`，并保留 `finally` 兜底清理；对已被即时删除的文件从列表中摘除，保证清理日志可读。
- **[整文件切块并行 VAD 未纳入本次]** → 若进程池上线后 VAD 仍是高并发下的主要瓶颈，届时需研究 FSMN-VAD 流式模型在分块边界上的等价性（`cache` 跨块有状态），单独开变更。
- **[规格与上一个变更的冲突]** → 本变更修改了 `local-model-runtime` 中「本地模型由单一服务进程承载」的要求（VAD 改由独立进程承载）。为使该修改在归档时可应用，已完成的 `2026-10-23-retire-v118-and-refine-v2-config` 已先行归档并同步主规格。
- **[段级临时文件清理时机的回归风险]** → 「用完即删」改变了文件生命周期，需覆盖「转写成功」「转写失败」「整体异常」「ITN 关闭」四条路径的测试。

## Migration Plan

1. **配置先行**：在 `config.toml` / `config.docker.toml` 中新增 `vad_device`、`vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s`，并移除 `[model_device].vad`。默认值保证与变更前行为接近（`vad_pool_size = 8` 对应原先的 4 路并发上限，属于放宽）。
2. **代码切换**：`core/vad.py` 切换到进程池客户端；新增 `core/vad_pool.py`；`core/config.py` 增加配置读取；`core/models.py` 增加 VAD 设备解析；`api/routes/asr_qwen_common.py` 改为段 clip 惰性创建与即时删除。
3. **后端运行与验证**：本地起服务，跑单元测试 + 真实音频的并发与效果验证，确认分段边界与变更前逐段一致（回归基线即三份完整音频的 753 / 947 / 703 段）。
4. **回滚策略**：配置项 `vad_pool_size = 1` 即退化为单 worker；把 `vad_device` 设为 `"cuda:0"` 可切回 GPU。若进程池出现异常，可临时把 `core/vad.py` 切回同进程实例池实现（保留旧代码路径一个版本）。
5. **不在本次范围**：镜像构建与部署上线另行确认后执行。

## Open Questions

- 部署侧最终给容器分配多少 CPU 核尚未确定，这直接决定 `vad_pool_size` 的取值。计划在部署前用真实音频压测 4 / 8 / 16 三档 worker 数，据此定档；规格不绑定具体数值，配置默认值 8。
- VAD 腾出的 GPU 容量是否足够支撑把 Qwen 的 `--api-server-count` 从 4 提到 8，需要在 VAD 改造上线后实测决定，属于后续变更。
