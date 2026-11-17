# Design

## Context

迁移前的链路是「FastAPI 服务 + 外部 vLLM 容器」：主服务内跑 CAM++ / emotion2vec，FSMN-VAD 由独立 spawn 固定 worker 进程池承载，其中 CAM++ 与 emotion2vec 用 fp16 TensorRT engine 加速；Qwen3-ASR 由独立容器以 OpenAI 兼容接口提供，服务通过 `POST /v1/audio/transcriptions` 上传内存 WAV 字节。

目标硬件是华为昇腾 910B（aarch64，8 卡 × 64GB，CANN 8.1.RC1，npu-smi 25.0.rc1.1）。迁移涉及两处推理后端的替换，以及随之而来的设备语义、并发闸门、状态口径、配置项和镜像交付方式的整体变化。

本设计同时记录历史实机结果、当前只读文件/安装元数据核验和待实施的方案；后两者不代表本轮推理或性能验收通过。验证环境为 `/home/zhangs/conda-envs/asr`（conda 环境，Python 3.10）+ 昇腾 CANN 8.1.RC1 容器底座。

## Goals / Non-Goals

**Goals:**

- Qwen3-ASR 改为进程内 `transformers` 推理，且对外请求/响应字段、错误码、分段语义与迁移前完全一致。
- CAM++ / emotion2vec 从 TensorRT 收敛为 `torch_npu`，数值语义不变。
- 产出一套可复现的昇腾 910B 运行时基线（依赖矩阵、容器绑定、dtype 与算子约束），后续人不再重复踩同样的坑。
- `config.toml` 与状态接口全面改为 NPU 口径，不残留 NVIDIA 专属配置。

**Non-Goals:**

- 不将多套生产后端长期并存作为目标；vLLM-Ascend 在隔离环境做候选验证，交付架构改变前先更新本 change 的决策和规格。
- 本轮完成原生并发标定与候选评估，不以追平 vLLM 单段延迟为验收条件；实际镜像构建和上线在服务稳定之后后置，与本轮硬件验收区分。
- 不改 `/v3/seacraft_asr` 的请求与响应字段；`/get_status` 允许按 NPU 口径调整字段语义。
- 不在本 change 内重做 VAD 切分策略与段级过滤阈值。

## Decisions

增量归档按既有 Requirement/Scenario 标识匹配，MODIFIED 中保留历史场景标题；其中 CUDA/GPU/engine 字样仅为标识，正文定义 NPU 行为，不恢复旧后端。v3 的 gpu_time_ms 同样按兼容契约保留。

### 决策 1：原生 Qwen 作为正确性基线，生产资格经比较确定

首个实现使用 `transformers 5.13.0` 的 `Qwen3ASRForConditionalGeneration`，以原生 eager 建立可验证基线，不宣称其性能最佳。原生微批、匹配运行时的图编译与锁定历史版 vLLM-Ascend 列入受控候选。候选未通过时如实记录失败条件和维护成本，不伪报支持；候选胜出也须先同步本 change 的主链及依赖规格再替换，正式交付仅保留选定执行路径。

**证据修订**：官方 main 标注 v0.30.0 起停止维护，但 v0.23.0 教程明确支持单张 910B 64GB BF16 和 A2 镜像；当前 vLLM 上游已有 HF 权重 mapper，官方 SDK 的 `LLM.generate()` 可读取 raw 输出。历史 0.23.0 与本机 HF checkpoint 不匹配及 HTTP 输出后处理是特定版本/接口问题，不构成永久排除整条路线的理由。候选需独立环境/镜像锁定 vLLM、插件、CANN、torch、驱动、模型 revision 和必要补丁，不污染 conda asr。

**语言与输入**：processor 接收内存 numpy 音频；仅浮点输入对齐模型 dtype，整数/布尔张量保持 dtype。最新 HF 模型卡描述 assistant prefill，而本机安装的 5.13.0 processor:495–540 为 system hint；必须锁定实际代码并覆盖 auto、具体语种、非法语种及 raw 前缀测试，不能仅依赖 latest 文档，也不能把人为预填语言当独立识别结果。

**历史单段记录（本轮未重跑）**：torch/torch_npu 2.5.1、BF16、权重加载 0.9s；4.1s 英语音频推理 13.11s，raw 含 `language English<asr_text>`，allocated 3.80 GiB。该记录不证明完整服务通过，也不代表 warm 稳态上限。

**小模型格式选择**：CAM++/emotion2vec 优先原始 checkpoint + torch_npu，先数值回归，再评估混合精度、TorchAir 和调度；ONNX→ATC→OM 仅在收益及动态输入/算子/精度/维护成本可控时纳入。不以 checkpoint 当前不是 ONNX 为排除依据。

参考链接与版本化证据见 [migration-review.md](migration-review.md)。

### 决策 2：本地推理客户端复用 `QwenAsrClient` 的对外契约，替换其传输层

`core/qwen_asr_client.py` 保留类名、`transcribe_bytes()` 方法签名与四个错误类（`4011` / `4012` / `4009`），仅将内部实现从 HTTP multipart 改为本地推理。这样 `api/routes/asr_qwen_common.py` 的调用点、既有单元测试的替身契约、README 的错误码表都不需要改动。

**并发闸门复用**：既有 `core/qwen_asr_queue.py` 的三层闸门（音频准入 / 段级信号量 / 全局请求信号量）语义不变，只是「发往 Qwen 服务的请求数」变为「同时执行的本地推理任务数」。段级与全局信号量直接复用，无需新建。

**新增约束**：同一模型实例在一次推理完成前不得被再次调度。这需要在模型池侧增加一个实例级信号量——既有的 CAM++ / emotion2vec 池已有此机制（实例 busy/waiting 计数），转写模型实例需要同等处理。

**替代方案**：新建独立的 `LocalQwenAsrClient` 并让路由层按配置二选一。放弃原因是会让「本地推理」变成第二条链路，两套并发闸门、两套错误处理都要维护，且未来若要回退到 HTTP 形态会更混乱。

### 决策 3：CAM++ `seg_pooling` 以等价算子序列替换

NPU 运行时不支持 CAM++ `seg_pooling` 使用的 `AvgPoolV2` 大 stride 形态。替换为 reshape→sum/max→expand→slice 的等价序列，保持输出 192 维 embedding 的语义与数值分布。

**为什么不改模型结构**：CAM++ 的 `seg_pooling` 是训练时确定的结构，改动会改变 embedding 分布，导致既有的 `speaker_similarity_threshold = 0.3` 标定失效。本轮优先采用等价序列替换，先验证数值语义，再复验既有阈值。

**验证要求**：覆盖 T=99/100/101/199/200/201 和负值，平均尾段按有效元素计数，最大尾段不得零填充改变负值；适配放项目内，不修改 site-packages。预先确定容差，使用可追溯 CUDA golden 或 CPU FP32 算子参考做同段比对，并复验聚类；没有 CUDA 对照时不得宣称 CUDA/NPU 一致已验证。

### 决策 4：容器设备绑定采用 `--privileged --runtime=ascend --ipc=host` + `ASCEND_VISIBLE_DEVICES`

**实测踩坑记录（均为实机验证所得）**：

| 问题 | 现象 | 根因 | 处理 |
|---|---|---|---|
| 设备号越界 | `Invalid device ID ... Expected value: [0, 0)` | 容器未映射 `/dev/davinci*` 设备节点 | 加 `--privileged --runtime=ascend --ipc=host` |
| 设备号越界（第二形态） | `open device 7 failed ... invalid device=7, input visible devices:7` | `ASCEND_VISIBLE_DEVICES=7` 后容器内**重新编号为 0** | 容器内一律用 `npu:0` |
| 编译器初始化失败 | `AclSetCompileopt(ACL_PRECISION_MODE) error code is 500001` + `ModuleNotFoundError: No module named 'attr'` | CANN 算子编译器子进程依赖 `attrs` | conda 环境补装 `attrs` |
| torchvision 导入崩溃 | `RuntimeError: operator torchvision::nms does not exist` | 镜像系统 site-packages 的 torchvision 与 torch 2.5.1 ABI 不兼容 | 用 conda 环境自带解释器，不做 PYTHONPATH 叠加 |
| dtype 不匹配 | `Input type (float) and bias type (c10::BFloat16) should be the same` | processor 产出 float32 特征，权重为 bf16 | 推理前显式 `.to(device, model.dtype)` |
| set_env 冲突 | `PYTHONPATH：未绑定的变量` | `set -u` 与 CANN `set_env.sh` 冲突 | source 前 `set +u`，source 后恢复 |

**设备编号规则**：宿主机物理卡号通过环境变量传入容器，**容器内设备标识从 0 开始重新映射**。配置文件中写 `npu:0` 指的是容器内可见的第一张卡。这条规则必须写进配置注释，否则运维会按宿主机 `npu-smi` 的编号去配 `npu:7`，得到「设备号越界」。

### 决策 5：运行时基线固定为独立 conda 环境 `asr`

依赖组合：`torch 2.5.1` + `torch_npu 2.5.1` + `transformers 5.13.0` + `numpy 1.26.4` + `funasr 1.2.6` + `av 14.2.0` + `torchvision 0.20.1` + `attrs`，环境前缀 `/home/zhangs/conda-envs/asr`。

**为什么不用生产容器的系统 Python**：生产镜像的 site-packages 存在 torch 2.2.0 残留（两版 libgomp / libopenblas / libarm_compute 混存），实测触发 `ld.so: dl-tls.c: _dl_add_to_slotinfo: Assertion 'idx == 0' failed`。宿主 conda 环境导入完全正常（`AutoProcessor OK | tf 5.13.0`）。该污染属于历史遗留，不作为目标环境。

**为什么 `av` 用 conda 装 14.2.0**：pip 的 `av==14.4.0` 需要 ffmpeg 7，而 conda 的 ffmpeg 是 9.0.2，编译失败（`Package libavformat was not found`）。conda-forge 的 `av=14.2.0` 有 linux-aarch64 构建，且与生产容器版本一致。

### 决策 6：清理 NVIDIA 执行路径，保留 v3 历史字段与业务契约

**删除**：`speech_campplus_use_tensorrt`、`speech_campplus_tensorrt_precision`、`emotion2vec_use_tensorrt`、`emotion2vec_tensorrt_precision`、`core/tensorrt_runner.py`、`core/model_assets.py`。

**新增**：`[qwen_asr].local_model_dir`（本地权重目录）、`[qwen_asr].local_device`（推理设备标识）。`base_url` 字段保留但本地模式下不发起 HTTP 调用——保留它是为了让旧配置文件不因缺键而启动失败，读取时忽略即可，并在日志中说明迁移。

**设备标识收敛**：`_validate_model_device()` 从接受 `cpu`/`cuda:N`/`npu:N` 改为只接受 `cpu`/`npu:N`。`cuda:N` 在就绪前明确报错并给迁移提示，不静默改成其他卡或 CPU。废弃 TRT 键可按迁移策略忽略并告警，二者行为不能混淆。

### 决策 7：`/get_status` 字段语义调整的边界

按用户要求，`/v3/seacraft_asr` 的请求与响应字段不动；`/get_status` 可改。本次调整限于：

- `deployment` 段：`cuda_tensorrt` 布尔位改为 NPU profile 标识与设备信息。
- `localModels`：显存字段语义改为 NPU HBM 占用。
- 新增转写模型的加载状态与设备标识（放在 `localModels` 同级，不改动既有键名）。

**不改动**：既有 `localModels` / `vadPool` / `resourceDiagnostics` / 音频准入相关键名，避免破坏监控与告警脚本。

### 决策 8：固定开发设备、资产状态和证据等级

宿主开发使用 `npu:7`、`/home/zhangs/conda-envs/asr/bin/python`。不得因机器有八卡就自动扩展到其他卡；卡 7 已有服务，共享干扰必须记录。仅映射物理卡 7 的容器按实际可见设备核对逻辑编号。

规划初期仅核对三个小模型文件；apply 阶段已完成原始权重加载及四模型 NPU 前向，详见迁移验证报告。Qwen 使用本地 HF 权重，文件 SHA 可追溯，上游 revision 未提供，保持未知。52 WAV、2 MP3 共 5,144,064,354 字节，52 WAV 文件头合计 43.6597 小时且为 16kHz/单声道/16位。目录存在 GBK 字节，建立递归 manifest、稳定样本 ID 和编码映射，不覆盖原始音频；文件头读取不等于完整解码，文件存在不等于推理成功。

### 决策 9：批执行、进程和资源所有权

“一个段一次转写”限定逻辑任务和结果归属，不限定一次物理 forward 只能一个样本。微批不得拼接音频，须验证语言、长度/mask、输出排序、错误和取消归属；批大小、等待窗口及实例容量单独配置和观测。每实例同时只执行一个独占执行单元（单段或批次），设备工作未完成前不归还租约。Python 超时和线程返回不等于 NPU 工作已经取消或完成。

保留 VAD 的 spawn 静态 worker/FIFO，不迁移为主进程内 VAD；验证 FunASR 实际落卡。先以一个完整服务进程建立基线，instance_count 增加会复制 Qwen 和增强模型；不得把实例数与请求信号量混同。设备级 synchronize 是待测开销点，只能在 stream/event 所有权正确后优化，不能直接删除同步。

### 决策 10：依赖、构建和清理按实际用途处理

统一 requirements 与 conda 重建入口，锁定 torch/torch-npu/torchaudio 2.5.1、transformers 5.13.0、numpy 1.26.4、funasr 1.2.6 的目标组合及来源；av 按当前 conda 14.2.0，保留 attrs、ITN 与必要音频依赖，细节及 metadata 版本见补充审计。候选环境隔离。依赖一致性、导入、前向、端到端分别验收。

保留 Cython 交付，锁构建工具，更新 NPU/模型版本元数据并验证 aarch64/Python ABI。纯 TRT runner/asset/engine 工装可退役；通用脚本和失败隔离/取消/准入测试改写保留，删除后检查 imports 闭包。归档 OpenSpec、原始模型、音频和可复算标定证据保留，不授权删除运行镜像、日志或缓存。

设备映射/runtime/IPC 参数放 Compose 或宿主启动入口；docker/start.sh 是容器内 entrypoint，只负责解释器、预检和进程。准备交付文件属于本轮，实际镜像构建/上线待服务稳定后执行。

### 决策 11：v3 序列化与计时不变

`bg`/`ed` 保持两位小数字符串，load_audio_time_ms/gpu_time_ms 保持字符串；gpu_time_ms 保留迁移前主链路计时起止及统计口径，不改名、不替换为设备 kernel 时间。增强开关默认 false，错误码、文本规则、时间轴、speed_info/language_ratio 保持契约。NPU 内部排队/编译/执行遥测另记，不能混入旧字段。

## 测试策略与验收门槛

具体测试矩阵见 [test-plan.md](test-plan.md)，执行顺序为冻结 v3 基准、逻辑回归、四模型真实 NPU 前向、短样本与课堂长录音、故障/取消及稳态资源、最后并发与后端性能比较。tasks 第 11 节将现有测试项关联到用例和证据，补充安全故障注入、加载竞争、混合长度微批和连续请求资源检查；不重复执行已有覆盖，也不把规划、包版本或文件存在算作测试通过。

测试不得修改原始模型与音频、耗尽共享卡或停止既有服务。质量和资源/性能预算先登记；缺乏人工参考时不宣称准确率达标，候选路径不采用需说明依据，原生交付的必需验收缺项不得归档。

按用户要求，测试批次完成或中止后必须关闭本次测试启动的服务，释放接口监听端口。启动时登记实例、进程树、容器和端口，成功/失败/取消均进入统一收尾；在途设备工作结束后关闭本次服务、代理、候选引擎和 worker，核验进程退出及测试端口不再监听，保留日志和收尾证据。批次内部连续阶段可以复用；批次结束不得保留后台接口或被测试自身重启机制重新拉起。关闭超时仅对已确认测试归属的资源升级处置，收尾失败必须报告并阻止验收完成；服务器原有共享服务保持原状。关闭过程自身遭遇取消或重复取消时，仍须持有收尾任务至完成或按有界超时策略报告失败，再传播取消，不能中途遗弃 teardown。

## Risks / Trade-offs

- **[转写吞吐显著下降]** → 实测 4.1 秒音频单段推理 13.11 秒（RTF ≈ 3.2），该单样本不构成同条件 vLLM 对比，不能据此计算后端加速比。缓解：迁移期把段级并发控制在实测可承载范围，`segment_concurrency` / `request_concurrency` 需按 NPU 实测重新标定；不沿用 NVIDIA 时期的 16/128。

- **[CAM++ 等价替换引入数值偏差]** → 缓解：同一段音频对比 NPU 与可追溯 CUDA golden；缺失时用 CPU FP32 参考并注明未验证 CUDA 对比，并复验说话人聚类簇数与 `speaker_similarity_threshold=0.3` 下的聚类结果；偏差超出容差先定位尾段、dtype、算子或 checkpoint 差异，不用改变模型结构或业务阈值掩盖错误。

- **[既有阈值在 NPU 精度下失效]** → `speaker_similarity_threshold`、复读阈值等均基于 NVIDIA 下的数值分布。缓解：迁移完成后用真实课堂音频复验阈值，产出新报告后回写 `config.toml` 与 `README.md`。

- **[NPU 实例失败时禁止静默回退]** → 缓解：NPU 实例加载失败时报告该模型不可用（`4015`/`4016`），不静默改走 CPU——静默回退会让「实际后端」与「配置后端」不一致，破坏既有可观测性契约。

- **[删除 NVIDIA 历史文档丢失决策依据]** → 缓解：归档的 18 个 openspec change 是决策的权威来源，不受文档清理影响；现行部署指导需改写；历史标定与原始测量保留，不仅凭关键词删除报告。

- **[容器内设备编号与宿主机物理编号不一致导致运维误配]** → 缓解：在 `config.toml` 的设备字段旁写明「容器内从 0 重新编号」，并在 `AGENT.md` 的常见坑表中新增该条。

## Migration Plan

1. **环境基线**：`asr` 虚拟环境建成并通过导入校验（torch / torch_npu / transformers / funasr / av 版本固定）。
2. **推理验证**：Qwen3-ASR-1.7B 在 NPU 上完成真实前向，输出格式与既有解析契约一致。
3. **本地推理后端**：替换 `core/qwen_asr_client.py` 传输层，保留对外错误码与并发闸门语义。
4. **本地模型后端**：CAM++ / emotion2vec 走 `torch_npu`，完成 `seg_pooling` 等价替换与数值比对。
5. **配置体系**：`config.toml` 与 `core/config.py` 改为 NPU 口径，删除 TensorRT 配置项。
6. **状态与诊断**：`/get_status` 与资源诊断改为 NPU 口径。
7. **单元测试**：`pytest tests/` 全绿；TensorRT/CUDA 强绑定的用例改写为 NPU 形态。
8. **文档与清理**：按清单清理纯 NVIDIA 代码/engine 工装并改写通用脚本（不删除运行镜像），更新 `AGENT.md` 与 `README.md`。
9. **实机验收**：真实课堂音频端到端回归 + 并发档位标定，产出报告。
10. **归档**：change 归档，delta 同步进 `openspec/specs/`。

**回滚策略**：本分支尚无生产部署，迁移期以 git 分支为单位回退。运行时依赖层（conda 环境、容器镜像）不与代码耦合，可独立回滚。部署阶段（服务稳定后）若需回退，保留旧 NVIDIA 镜像与旧配置文件副本即可。

## Open Questions

- **转写模型实例数与 NPU 卡位的最优分配**：需要在实机压测后确定（机器有 8 卡，但本次仅使用卡 7；Qwen 加载 allocator 差值约 3.8 GiB，仅为估计，冷并发时可能混入同进程其他模型分配）。这会影响 `config.toml` 默认值，但不影响本 change 的规格与任务拆分。
- **VAD 的 NPU 设备占比**：`vad_device` 数组中 NPU 与 CPU 的配比需实测决定（FSMN-VAD 在 CPU 上本就不慢，可能维持多数 CPU + 少量 NPU）。
- **转写模型是否需要多实例**：本次仅比较卡 7 内单实例、多实例与微批，其他卡不在授权范围。暖短片 2/2 闸门单实例 P50/P95 为 14.14/23.73 秒，双实例为 14.45/22.90 秒，未见足够收益，因此正式配置保留单实例；微批有文本差异不采用。完整八路档位比较另见最终验收。

这些档位在功能稳定后标定，最终值和生产后端选择必须在本 change 实机验收完成前记录；不得用“稍后标定”提前归档。

## Apply 实机校正（2026-11-17）

- FP32 权重/输入本身不足以保证内部计算精度；默认 HF32 卷积使 CAM embedding 最大误差约 0.0033。固定每进程一次禁用 conv/matmul HF32 后降至 5.4e-6，未放宽预登记门槛。
- emotion2vec 原 GELU 的 none/erf 定义在当前 NPU 路径被 tanh 近似。逐层 hook 定位首 GELU 输入误差 1.4e-6、输出误差 4.74e-4；仅对所持 emo 实例绑定 npu_gelu(approximate=none)，CPU 保持原 F.gelu，7 类概率误差降至 4.9e-6。无需改权重、安装包或 attention。
- 原生 batch=2 短样本吞吐提升 56.4%，但 1/6 片段出现 7 轮稳定的实质文本差异，缺人工参考，不能判质量通过或失败。本轮正式交付保留逐段 eager，微批仅评估工装，未引入生产批次排队或容量语义。
- 音频标准化使用已锁定 PyAV，解决目标缺 ffmpeg CLI 的 MP3 入口；合规 PCM16 原字节直通，不新增段级落盘。
- conda clone 可能重新链接 NumPy 2，而 pip metadata 仍显示 1.26.4；必须锁定重建并核验实际 import，不能以 clone/pip list 代替前向验证。
- 关闭失败聚合上报且保留失败实例所有权。VAD 先排空结果再关闭 Queue/join_thread/Process。统一 main.py 入口将 TERM 重放转为 Python SystemExit，正常只通知自有服务父进程，保证 lifespan 和 Python atexit；当前 NPU 初始化后 INT 未能触发关闭，不以该信号作为可靠退出机制。超时升级自有组 TERM/KILL 且判失败。


## 最终实测决策（2026-11-18）

相同八条完整SHA去重WAV、4B+4A、Qwen单副本/3CAM++1emo/固定VAD池，在npu:7共享卡上g1=1/1的P50/P95为4104.338199445512/4781.66245310544秒，g2=2/2为4304.155876739882/5002.106922458205秒；各8/8成功，3109个最终输出分段逐项相同，无人工准确率参考。正式默认采用保守1/1+单Qwen，覆盖短片暖标定曾选2/2的阶段结论，不宣称独占因果加速或全局最优。

最终源版本449项CPU回归/146子断言通过；独立新30扩展stage仅重编core.config，其余29扩展与main字节保留。新stage三轮全B预热+16轮A/全B NPU稳态通过，allocated/reserved增长0、RSS+5701632bytes、线程/FD增长0，cache/lease清空。两完整批及最终steady的主进程/4worker均正常退出，无强制/孤儿，端口18083释放并观察5秒；既有服务保留。

Qwen仍原生eager BF16默认SDPA，CAM++/emo仍原生FP32；候选选择、资源/质量边界、源码/产物指纹和原日志SHA详见docs/npu-910b-migration-validation-20261117.md及其配套证据。speaker阈值0.3、复读40与原VAD实际参数/过滤行为保留，不以无人工标注声称召回达标。实际Docker构建/容器承载/上线未执行。
