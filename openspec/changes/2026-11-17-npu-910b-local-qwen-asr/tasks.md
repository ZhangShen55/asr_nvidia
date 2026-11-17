# Tasks

## 0. 迁移前置条件与 v3 契约冻结（补充）

- [x] 0.1 固化 `/v3/seacraft_asr` 的请求、响应、错误码和默认值基准；覆盖 `bg`/`ed` 两位小数字符串、`load_audio_time_ms`/`gpu_time_ms` 字符串、增强功能默认 false，以及 `speed_info`、`language_ratio`；保持 `gpu_time_ms` 原计时起止与统计口径，不改成纯 NPU kernel 时间。
- [x] 0.2 核验用户已补充的 emotion2vec_plus_large、speech_fsmn_vad_zh-cn-16k-common-pytorch、speech_campplus_sv_zh_en_16k-common_advanced 三个目录的权重、配置、标签和 checkpoint/revision；另行核对 Qwen 权重，分别记录缺失项。文件到位后仍须通过实际加载与前向验证，才能勾选对应验收任务。
- [x] 0.3 建立测试音频递归清单：已发现 52 WAV、2 MP3，共 5,144,064,354 字节；WAV 文件头合计约 43.66 小时，均为 16 kHz/单声道/16 位。记录 SHA256、音频格式、时长及样本用途，区分文件头检查、完整解码和人工标注的证据等级。
- [x] 0.4 处理测试目录 GBK 原始文件名字节与 UTF-8 路径不匹配；扫描器支持现有文件，上传使用稳定样本 ID；若规范化名称，先记录原始字节到新路径的映射并检查重名，禁止覆盖原样本。
- [x] 0.5 固定本次宿主开发为 `xjtu910b7`、`npu:7`、`/home/zhangs/conda-envs/asr/bin/python`；记录卡 7 上既有服务占用，不调度其他卡、不终止既有服务，不把共享卡结果称为独占性能。

> 0.x 是前置检查；10.x 按相应原任务的依赖关系执行，不是等到第 9 节全部完成后再开始。所有复选框代表实施/验收，规划写入不勾选。

## 1. 运行时基线与依赖锁定

- [x] 1.1 固化 `/home/zhangs/conda-envs/asr` 的目标依赖与来源；确认 torch/torch-npu/torchaudio 2.5.1、transformers 5.13.0、numpy 1.26.4、funasr 1.2.6、av 14.2.0、torchvision 0.20.1、attrs 的兼容组合，分别记录 metadata、依赖一致性和导入检查；在宿主 npu:7 验证设备可用，不以 metadata 代替前向通过。
- [x] 1.2 编写并固化环境重建脚本（参考 `/data/asr-bootstrap/rebuild_env.sh`），确保在空机器上可一次性重建出完全一致的 `asr` 环境；验证脚本在临时前缀下执行一次 `pip list` 快照，与目标环境的包版本逐项比对无差异。
- [x] 1.3 把实机验证踩坑固化为可执行的启动前检查脚本（`set_env.sh` 的 `set +u` 包裹、conda 环境解释器优先、`attrs` 存在性、`ASCEND_VISIBLE_DEVICES` 与容器内编号映射、dtype 转换）；验证脚本在容器内对缺少 `attrs` 的环境能给出明确诊断而非静默失败。

## 2. Qwen3-ASR 本地推理后端

- [x] 2.1 新增本地转写推理模块：从配置目录加载 `AutoProcessor` 与 `Qwen3ASRForConditionalGeneration`，权重以 `bfloat16` 加载并放置到目标 NPU 设备；验证加载后 `model.dtype` 为 `torch.bfloat16` 且权重放置无异常。
- [x] 2.2 实现内存音频直传：用 `processor.apply_transcription_request(audio=numpy_float32, language=...)` 处理段音频，推理前把浮点输入显式 `.to(device, model.dtype)` 而整型索引张量保持原 dtype；验证 4.1 秒样本输出 `language English<asr_text>...` 且整型张量数值未变。
- [x] 2.3 实现原始输出解析：按既有 `language <语种><asr_text>` 前缀切分文本与语种，空文本或空语种判定为无有效语音；验证既有 `utils/qwen_response.py` 解析器无需修改即可正确剥离前缀并得到语种。
- [x] 2.4 实现语种参数对接：`auto`/空串/未提交时不指定语种，具体语种原样传入处理器（支持全名与 ISO 码），处理器拒绝时映射为 `4009`；验证三种输入路径各自的行为符合规格。
- [x] 2.5 实现模型实例级串行调度：同一转写模型实例在一次推理完成前不被再次调度，实例 busy/waiting 计入状态；验证并发提交同一实例时实际执行数始终为 1，排队请求在完成后依次执行。
- [x] 2.6 实现懒加载与失败映射：模型未触发时不加载权重；目录缺失/权重不可读/设备不可用映射为 `4011`，推理异常或超时映射为 `4012`；验证三种失败路径各自返回既有错误码，且失败不被误判为 `4008`。
- [x] 2.7 改造 `core/qwen_asr_client.py`：保留类名、`transcribe_bytes()` 签名与 `QwenAsrError`/`QwenAsrUnavailableError`/`QwenAsrCallError`/`QwenAsrUnsupportedLanguageError` 四个错误类，替换内部传输层为本地推理，`base_url` 读取后不再发起 HTTP 调用；验证 `api/routes/asr_qwen_common.py` 的调用点无需改动即可工作。
- [x] 2.8 复用 `core/qwen_asr_queue.py` 的三层闸门：段级与全局信号量的语义从「发往服务的请求数」改为「同时执行的本地推理任务数」，准入、排队、超时与错误码逻辑不变；验证 `segment_concurrency` 与 `request_concurrency` 各自独立生效。

## 3. 本地模型后端迁移到 torch_npu

- [x] 3.1 收敛 `core/local_inference.py` 的后端映射：保留 `pytorch_cpu` 与 `torch_npu`，移除 `tensorrt` 与 `pytorch_cuda` 分支；验证 `backend_for_device()` 对 `cpu`/`npu:N` 返回正确后端，对 `cuda:N` 给出迁移提示。
- [x] 3.2 修复 `TorchRunner.generate()` 的 CUDA 硬编码：`torch.amp.autocast("cuda", ...)` 与 `torch.cuda.synchronize` 改为按设备分派，NPU 路径使用 `torch.npu` 的精度与同步接口；验证 NPU 实例调用 `generate()` 不触发 CUDA API。
- [x] 3.3 为 CAM++ 实现 `seg_pooling` 的 NPU 等价算子替换：用 reshape→sum/max→expand→slice 序列替代 NPU 不支持的 `AvgPoolV2` 大 stride 形态；验证输出仍为 192 维 embedding 且维度与 CUDA 路径一致。
- [x] 3.4 预先确定 CAM++ 数值容差，用同段音频比较 NPU 与可追溯 CUDA golden；没有 CUDA 基准时先用 CPU FP32 验证算子/网络参考并明确证据限制，不冒充 CUDA 对比通过；复验 0.3 阈值下聚类，记录余弦相似度、最大误差和簇数。
- [x] 3.5 验证 emotion2vec 在 NPU 上的完整分类输出（9 类中取 7 个有效类），确认标签顺序与业务映射一致、`tokens.txt` 的 `unuse` 位置校验仍生效；验证 1～10 秒之外的片段仍按既有规则记为「平淡」。
- [x] 3.6 删除 `core/tensorrt_runner.py` 与 `core/model_assets.py`，清理 `core/models.py`、`core/deployment.py` 中的 TensorRT 引用；验证 `python -m pytest tests/` 中不再有 TensorRT 相关导入错误。

## 4. 配置体系改为 NPU 口径

- [x] 4.1 改造 `core/config.py`：新增 `[qwen_asr].local_model_dir` 与 `[qwen_asr].local_device` 的读取与校验，`base_url` 保留读取但标记为迁移遗留且不生效；验证缺少 `local_model_dir` 时给出明确默认路径或校验错误。
- [x] 4.2 移除四个 TensorRT 配置项的读取与校验（`speech_campplus_use_tensorrt`、`speech_campplus_tensorrt_precision`、`emotion2vec_use_tensorrt`、`emotion2vec_tensorrt_precision`），旧配置中的这些键被忽略并记录迁移提示；验证 `validate_model_backends()` 不再要求 fp16 engine。
- [x] 4.3 收敛设备标识校验：`[model_device_pool]` 的设备数组只接受 `cpu` 与 `npu:N`，`cuda:N` 触发明确的迁移报错；验证非法设备标识在就绪前报出完整字段路径与合法取值。
- [x] 4.4 改写 `config.toml`：`[vad].vad_device` 改为 NPU 形态，`[model_device_pool]` 两个设备数组改为 `npu:N`，删除全部 TensorRT 与 `cuda:` 配置项，新增本地转写模型目录与设备配置；验证 TOML 可解析且默认值与 `core/config.py` 一致。
- [x] 4.5 在 `config.toml` 的设备字段旁写明「容器内设备编号从 0 重新映射，`ASCEND_VISIBLE_DEVICES` 指定的是宿主机物理卡号」；验证注释存在且与 `AGENT.md` 的常见坑表述一致。
- [x] 4.6 更新 `tests/test_config_contract.py`：覆盖新增字段、删除字段的忽略行为、设备标识合法性与非法值报错；验证 `python -m pytest tests/test_config_contract.py` 通过。

## 5. 状态接口与资源诊断改为 NPU 口径

- [x] 5.1 修改 `core/status.py` 的 `deployment` 段：移除 `cuda_tensorrt` 布尔位，改为报告 NPU profile 与设备标识；验证既有 `localModels`/`vadPool`/`resourceDiagnostics`/音频准入相关键名保持不变。
- [x] 5.2 在状态中新增转写模型的加载状态、设备标识与 NPU HBM 占用（放在 `localModels` 同级，不改动既有键名）；验证查询状态不触发模型加载，未加载时占用为 0。
- [x] 5.3 把 `core/models.py` 与 `core/local_inference.py` 的资源统计从 NVIDIA 显存口径改为 NPU HBM 口径，非 PyTorch 分配不得因 PyTorch 计数为零而被报告为无占用；验证状态数值与 `npu-smi info` 的进程占用可核对。
- [x] 5.4 修改 `core/resource_diagnostics.py`：以 `npu-smi` 采集 NPU 资源水位，GPU 显存安全水位改为 NPU HBM 水位；验证采样失败时按既有口径标记不可用而不影响业务字段。
- [x] 5.5 更新 `tests/test_resource_diagnostics.py` 与状态相关测试，覆盖 NPU 口径字段与旧 NVIDIA 字段的移除；验证 `python -m pytest tests/test_resource_diagnostics.py` 与状态接口测试通过。

## 6. 单元测试适配

- [x] 6.1 改写 `tests/test_tensorrt_config.py`：删除 TensorRT 参数校验用例，改为验证四个键被忽略并记录迁移提示、设备标识非法时报错；验证该文件单独运行通过。
- [x] 6.2 改写 tests/test_inference_backends.py 与 tests/test_local_model_runtime.py 的后端替身和设备契约，保留 NPU 不调用 CUDA、CPU 隔离、失败池及取消收尾场景，不能因含 TRT 字样整份删除；验证相关测试通过。
- [x] 6.3 改写 `tests/test_docker_deployment.py`：镜像标签规则保留，CUDA profile 断言改为昇腾 profile 与设备绑定参数断言；验证该文件单独运行通过。
- [x] 6.4 更新 `tests/test_qwen_v2.py` 与 `tests/test_memory_clip_pipeline.py` 的客户端替身：改为本地推理接口契约（`transcribe_bytes` 返回带前缀文本），并补充 `4011`/`4012`/`4009` 三种失败的映射用例；验证相关测试通过。
- [x] 6.5 运行完整单元测试 `python -m pytest tests/`，确认全部通过并记录结果到报告；验证无 import 错误、无 TensorRT 相关残留断言失败。

## 7. NVIDIA 资产清理

- [x] 7.1 逐项审计 docs 中现行 NVIDIA 部署指导、历史压测及调研报告；改写现行指导，保留可复算标定证据和归档 OpenSpec。对纯过时且确需移除的文档列清单并记录结论承接，不按 cam/cuda 等关键词批量删除。
- [x] 7.2 退役纯 TRT 导出、资产检查、验证与压测脚本；通用延迟、聚类、阈值和 VAD 工具改写保留。移除 validate_backend_admission 的 TRT 工装前核对准入/容量/取消断言已保留；验证保留脚本无已删模块断引用。
- [x] 7.3 审计 test/ 下四份历史 VAD 实验脚本的实际用途；仅移除无调用者且已被替代的实验工装，保留可复用基准与音频样本，不按实验名称自动判定 NVIDIA 专属；test/ 与 tests/ 职责保持清晰。
- [x] 7.4 移除三份 RTX Dockerfile 及 build.sh 中 GPU_PROFILE/engine 打包；准备昇腾 Dockerfile、compose 和宿主启动说明。NPU runtime/设备/IPC 参数放 compose 或宿主入口，docker/start.sh 只管容器内环境/预检/进程。验证交付文件契约，实际镜像构建与上线后置。
- [x] 7.5 删除 `requirements-tensorrt.txt`、`requirements-tensorrt-export.txt`，更新 `requirements*.txt` 为昇腾依赖栈；验证 `pip install -r` 的依赖清单不含 CUDA/TensorRT 包且包含 `torch==2.5.1`、`torch-npu==2.5.1`、`transformers==5.13.0`。
- [x] 7.6 审计其他明确废弃的 NVIDIA open change 并按项目流程处理；保留当前 NPU change 至实现、验收和归档，保留历史归档，不要求实施期间 openspec list 为空。

## 8. 文档回写

- [x] 8.1 更新 `AGENT.md`：速查表补 NPU 设备绑定与 `asr` 环境说明，代码地图把 `core/qwen_asr_client.py` 改为本地推理客户端、移除 `core/tensorrt_runner.py` 行，配置体系与构建部署章节改为昇腾口径，常识坑表新增「容器内设备编号从 0 重新映射」与「CANN 编译器依赖 attrs」两条；验证文中不再出现 CUDA/TensorRT 口径的现行描述。
- [x] 8.2 更新 `README.md`：配置章节改为 NPU 设备标识与新增配置项，错误码表保持 `4011`/`4012`/`4009` 语义并补充本地推理的触发条件，部署章节改为昇腾镜像；验证 README 中的配置键与 `config.toml` 逐项一致。
- [x] 8.3 新增 `docs/` 下的 NPU 迁移验证报告（文件名带日期），记录：依赖栈矩阵、Qwen3-ASR 实机前向输出与耗时、CAM++/emotion2vec 的 NPU 实测结果、与可追溯 CUDA golden 或明确标注的 CPU FP32 参考的数值比对、已知限制与不适用场景；验证报告中的每个数值都能对应到可复算的命令或日志。

## 9. 实机验收

- [x] 9.1 在宿主 npu:7 用真实课堂音频跑通 v3；接口字段、类型、时间轴与分段契约严格不变。对文本、语种、聚类和情绪按预先明确的参考及质量标准回归，不要求不同后端生成文本逐字必然相同；无人工参考时不宣称 WER/CER 达标。
- [x] 9.2 验证 `4011`/`4012`/`4009` 三种错误码在真实请求中按预期返回，且失败不被误判为 `4008`；验证 `/get_status` 正确报告转写模型加载状态、NPU 设备与 HBM 占用，且查询不触发加载。
- [x] 9.3 按实测标定 `segment_concurrency` 与 `request_concurrency` 的 NPU 可承载档位（不沿用 NVIDIA 时期的 16/128），并在 `config.toml` 写入取值与依据；验证标定报告给出逐请求原始测量、并发配置与 P50/P95。
- [x] 9.4 用真实课堂音频复验 `speaker_similarity_threshold` 与复读修复阈值在 NPU 精度下是否仍适用；验证结论写入报告，若需调整则同步更新 `config.toml`、`README.md` 与 `tests/test_config_contract.py` 的默认值断言。
- [x] 9.5 产出八路完整音频并发验收报告（可复算的 P50/P95、样本清单与 SHA256、逐请求原始测量、活动 B 峰值与资源指标）；验证报告明确对比维度为「torch_npu 在不同并发档位下的表现」，而非 NVIDIA 后端对比。

## 10. 依赖、后端选型与构建闭包（交叉前置任务）

- [x] 10.1 统一全部 `requirements*.txt` 与环境重建入口：锁定 torch/torch-npu/torchaudio 2.5.1、transformers 5.13.0、numpy 1.26.4、funasr 1.2.6 的已验证目标组合；修正 av 14.4.0 与当前 conda av 14.2.0 的冲突，记录 Python/aarch64/CANN/驱动、分发来源及 pip/conda 分工。
- [x] 10.2 分离通用运行、NPU、开发测试和可选优化依赖；保留 torchaudio fbank/音频读取、WeTextProcessing/pynini、attrs。依据实际 import/依赖元数据决定 requests/httpx/av/torchvision 的去留，覆盖测试客户端和压测脚本，不按核心代码无直接 import 就删除传递依赖。
- [x] 10.3 在独立临时前缀验证可重建性和依赖一致性，分别记录 metadata、导入、NPU 单模型前向、端到端结果；不得原地升级/降级当前 asr 环境验证候选，也不得以安装成功替代实机验收。
- [x] 10.4 建立 Qwen eager 预热后基线；评估长度分桶微批、匹配当前版本的图编译，以及隔离环境内锁定历史版本的 vLLM-Ascend。记录权重格式/模型 revision、raw 输出、语言契约、补丁、维护状态、质量和稳态性能；不预先宣称原生 eager 最优或 vLLM 路线永久不可用。
- [x] 10.5 完成生产后端选择记录：默认原生实现是基线，候选通过不自动改变交付架构；若选择候选替换原生主链，先同步 proposal/design/specs/tasks 和依赖边界，再实施。正式交付仅保留选定执行路径及必要适配契约。
- [x] 10.6 若启用微批，配置批大小、等待上限及长度策略；保持各段独立音频、语言、结果、VAD 时间轴和错误归属，明确实例/批次/段的容量计数及取消收尾。不得拼接音频或简单将不同长度输入补零后无 mask 求均值。
- [x] 10.7 比较 CAM++/emotion2vec 的 eager FP32、质量合格的 BF16/FP16 和匹配运行时的 TorchAir；记录动态长度、冷编译、重编译、稳态延迟及同卡干扰。只有收益足够且算子/长度/精度可控时再选 ONNX→ATC→OM，不以原权重不是 ONNX 作为永久排除理由。
- [x] 10.8 验证 CAM++ 池化算子在 T=99/100/101/199/200/201、短输入和负值下的等价性；平均池化尾段按有效元素计数，最大池化无效位置不得用零改变负值。适配放在项目中，不修改 conda site-packages；同时验证 embedding 与聚类。
- [x] 10.9 审计 NPU worker 与模型池生命周期：保留 VAD spawn/FIFO/静态容量，验证实际落卡；查询状态不加载模型，NPU 加载失败不静默改走 CPU；超时/取消后等待设备工作结束再归还实例及音频租约。
- [x] 10.10 修正 VAD 和增强模型内存采样中 NPU 返回 0 的路径；区分 PyTorch allocated/reserved、进程 HBM 与设备聚合 HBM，无法归因时明确标记，不把既有 uvicorn 占用算作本项目模型占用。
- [x] 10.11 改写保留通用脚本：bench_local_model_latency、check_cam_cluster_boundary、speaker_threshold_sweep、profile_role_emotion、verify_vad_pool_regression、VAD 扩展压测及 CAM/emo 汇总；清理其 TRT 导入、CUDA autocast/device/synchronize 与报告中的硬编码。
- [x] 10.12 补全测试迁移：test_docker_profile、test_model_pool、test_model_pool_route、test_local_model_runtime、test_vad_pool、test_config_contract；反转 test_tensorrt_config 中“VAD 拒绝 NPU”的旧断言。保留 test_inference_backends 中 NPU 不查询 CUDA、隔离、失败池与取消收尾场景，不整份删除带 TRT 替身的业务测试。
- [x] 10.13 删除纯 TRT 模块/脚本后执行导入闭包检查，覆盖业务、tests、保留 scripts 和构建清单；可删除纯 TRT 的 test_model_assets。移除 validate_backend_admission 的 TRT 工装前，确认 tests/test_audio_admission* 仍覆盖其通用准入/容量/取消契约。
- [x] 10.14 保留并适配 Cython 交付：锁定构建工具，更新 build_cython 元数据中的 TRT 为 NPU/transformers/架构信息，在目标 aarch64/Python ABI 下验证扩展导入，确保新模块纳入构建且不残留已删模块引用。
- [x] 10.15 将 NPU 设备映射、runtime/IPC/可见物理卡参数放在 Compose 或宿主启动入口；docker/start.sh 只处理容器内解释器、预检和进程启动。初始单服务进程，明确 instance_count 复制本地 Qwen 权重的资源成本。
- [x] 10.16 用同一份清单记录删/改/保留的资产、理由和替代验证；保留归档 OpenSpec、模型权重、原始音频及可复算标定证据。清理源文件不等于允许删除运行镜像、日志、缓存或既有服务。
- [x] 10.17 分层验收：先短样本与失败路径，再课堂长录音，最后八路并发；记录质量、P50/P95、音频吞吐、RTF、HBM、CPU 与卡 7 共享占用。无人工参考时仅报告结果差异，不宣称 WER/CER 或有效对话召回达标。
- [x] 10.18 复核规划一致性与状态：修正 VAD 同进程、v2 路径和固定 16/128 等历史表述；清理其他 NVIDIA open change 时保留当前 change 到验收归档。交付文件适配纳入本轮，实际镜像构建和上线待服务稳定后执行。

## 11. 专项测试落地与最终门槛

> 详细矩阵与通过标准见 [test-plan.md](test-plan.md)。本节补齐测试缺口，并引用既有 0.x、2.x、3.x、6.x、9.x、10.x 任务；不要求重复执行相同测试，也不以规划写入代表测试完成。

- [x] 11.1 建立可追溯测试清单和冻结基准：将 v3 协议/确定性后处理用例、8种增强开关组合、GBK/编码/解码/静音/异常输入及实际音频子集映射到现有任务；登记样本ID/SHA256、参考来源、容差和计划执行层次，明确有无人工标注，不能把同源副本作为独立验收样本。
- [x] 11.2 补充加载锁、队列、设备执行、结果读取和释放阶段的故障/取消/重复取消/超时用例，验证实例所有权、错误归属、其他实例隔离及最终计数；破坏性故障用独立配置/替身注入，增加本项目独立 NPU 实例的安全取消冒烟，不改原始权重或耗尽共享卡资源。
- [x] 11.3 对采用的微批、图编译与低精度路径补混合长度/语言、尾批、mask、有效长度归一化及输出恢复顺序回归；与逐段数值/业务参考按同一标准比较，覆盖失败及取消归属。未采用的优化路径记录不适用依据，不冒充已通过。
- [x] 11.4 增加连续请求稳态与资源收尾测试：预登记预热基线、轮次/持续时间、冷却及容差依据，采集HBM allocated/reserved、RSS、线程/FD、缓存、worker与租约；区分缓存常驻和持续增长，排空/优雅关闭后验证适用基线恢复，不把共享服务占用归因本项目。
- [x] 11.5 按 test-plan 分层汇总实际命令、日志、用例到任务的覆盖及通过/失败/未执行/不适用原因；CPU桩、版本元数据或skip不替代四模型真实前向与v3验收，必需项缺证据不勾选或归档。仅在对应既有验收任务具备证据时一起完成，禁止仅凭测试数量或单次短样本宣布迁移完成。
- [x] 11.6 测试启动时登记本次实例/进程树/容器及监听端口，在成功、失败、取消或测试批次中止的统一收尾路径停止接收请求、等待在途工作结束并关闭本次启动的服务、代理、候选引擎及 worker；核对进程退出、无孤儿进程、测试端口解除监听且未被测试自身重启机制拉起，保存关闭前后证据与日志。关闭超时只处置已确认归属本次测试的资源；失败则报告并阻止验收完成，不按端口/进程名批量终止共享服务。批次内连续阶段可复用服务，批次完成后不得留接口后台运行；收尾自身收到重复取消时延迟传播取消，等待有界收尾完成并按约定报告超时/失败，不能遗弃关闭任务。

## Workflow follow-up

- 评审本 change 的规划产物（proposal / design / specs / tasks）后，再运行 `$openspec-apply-change` 实施。
- 仅在实施任务和适用验收均有通过证据、测试服务及端口已关闭后完成收尾；未执行、失败或仅 skip 的必需项不得当作通过，不提前归档。
- 按用户要求，将迁移代码、配置、依赖、测试、报告和 OpenSpec 产物以中文 Conventional Commits 提交，并推送当前 `refactor/adapt-qwen-asr-2huawei910b` 分支；不强推、不改写已推送历史。
- 随后同步本 change 的 delta 到 `openspec/specs/` 并归档；归档与规格同步用独立提交记录，再次推送。检查本地 HEAD 与上游一致、预期工作区干净，并记录提交号、归档位置和测试端口释放证据。
- 部署工作（昇腾镜像构建与上线）按用户要求放在整个服务能够稳定运行之后执行，不在本 change 的实施范围内。
