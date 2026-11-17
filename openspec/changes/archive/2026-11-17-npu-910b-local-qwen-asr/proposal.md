# Proposal

## Why

本项目从 NVIDIA CUDA/TensorRT 与外部 Qwen HTTP 链路迁移到华为昇腾 910B。迁移须保持 `/v3/seacraft_asr` 的字段、类型、默认值、错误码和业务规则，并处理推理后端、VAD worker、资源所有权、依赖和交付方式的变化。

Qwen 原生 Transformers + torch_npu 作为首个正确性基线；历史 vLLM-Ascend 有 910B 部署资料，但存在版本停止维护、权重与处理器适配成本，不能再表述为“没有可用 NPU 方案”。本轮纳入隔离候选的受控比较，结果不自动改变主链；若决定替换原生交付路径，先同步规划和规格。CAM++/emotion2vec 优先直接使用原始 checkpoint，经算子适配与数值回归后评估图编译；不以原权重不是 ONNX 为由永久排除格式转换。

## What Changes

### 推理链路

- Qwen3-ASR 从「外部 vLLM 服务 + HTTP multipart 调用」改为「进程内 `transformers` 推理」，模型以 `Qwen3ASRForConditionalGeneration` 加载，使用 `processor.apply_transcription_request(audio=..., language=...)` 直接接收内存 numpy 音频。
- 保留并严格对齐既有对外契约：1 个 VAD 段 = 1 个独立逻辑转写任务 = 至多 1 个有效响应分段；物理批执行不拼接音频、不改变归属和时间轴；输出仍为 `language <语种><asr_text>...` 前缀形态，由既有 `utils/qwen_response.py` 解析；错误码 `4011`/`4012`/`4009` 语义不变；`language=auto` 不下发具体语种。
- 并发语义从「HTTP 请求数」改为「本地推理任务数」：`segment_concurrency` 仍限制单音频内段并发，`request_concurrency` 改为全局本地推理并发上限，并新增模型实例级串行约束（同一模型实例同时只执行一次推理）。
- 模型发现从调用 `/v1/models` 改为加载本地权重目录；模型目录缺失、权重不可读或设备不可用时返回 `4011`。

### NPU 运行时

- 新增 `npu-910b-runtime` 能力，固定以下运行时基线，并区分历史实机记录、安装元数据和本轮待验证事项：
  - 依赖栈：`torch 2.5.1` + `torch_npu 2.5.1` + `transformers 5.13.0` + `numpy 1.26.4` + `funasr 1.2.6`，在名为 `asr` 的独立虚拟环境中安装。
  - 容器内必须使用虚拟环境自带解释器；镜像系统 site-packages 的 `torchvision` 与 torch 2.5.1 ABI 不兼容，会导致 `operator torchvision::nms does not exist`。
  - CANN 编译器子进程依赖 `attrs` 模块，缺失会表现为 `ModuleNotFoundError: No module named 'attr'` 与 `AclSetCompileopt ... error code is 500001`。
  - `ASCEND_VISIBLE_DEVICES=N` 会使容器内设备重编号为 `0`，容器内设备标识 MUST 使用 `npu:0` 而非宿主机物理编号。
  - 推理输入 MUST 显式转换为模型 dtype（`input_features` 为 float32、权重为 bfloat16 时直接调用会报 dtype 不匹配）。
  - 容器必须以 `--privileged --runtime=ascend --ipc=host` 启动并映射 NPU 设备节点，否则报 `Invalid device ID ... Expected value: [0, 0)`。

### 本地模型后端

- CAM++ 与 emotion2vec 移除 TensorRT 后端，统一使用 `torch_npu` 原始网络（`load_native_network()` 已在 NPU 路径上就位）。
- CAM++ 的 `seg_pooling` 算子在 NPU 上不可直接执行（`AvgPoolV2` 不支持大 stride），MUST 以 reshape→sum/max→expand→slice 的等价序列替换，且替换后 MUST 保持 192 维 embedding，并按约定容差对比可追溯参考（CUDA golden 优先，缺失时明确 CPU FP32 参考及限制）。
- 移除 `speech_campplus_use_tensorrt`、`speech_campplus_tensorrt_precision`、`emotion2vec_use_tensorrt`、`emotion2vec_tensorrt_precision` 四个配置项；移除 `core/tensorrt_runner.py` 与 `core/model_assets.py` 的 TensorRT 资产校验路径。
- 设备数组语义扩展为 `cpu` / `npu:N`；`npu:N` MUST 固定走 `torch_npu`，忽略任何加速器开关。

### 配置体系

- `config.toml` 全面改写为 NPU 口径：宿主开发 `[vad].vad_device` 仅使用 `cpu`/`npu:7`，容器模板使用经可见设备映射确认的逻辑编号；`[model_device_pool]` 两个设备数组改为 `npu:N`；删除全部 TensorRT 与 `cuda:` 配置项。
- 新增 `[qwen_asr].local_model_dir` 等本地推理必需配置项，保留 `base_url` 字段仅用于兼容既有配置结构（本地推理模式不发起 HTTP 调用）。
- `core/config.py` 的设备校验、`validate_model_backends()` 与 `/get_status` 的 `deployment` 段 MUST 改为 NPU 口径，不再暴露 `cuda_tensorrt`。

### 状态与诊断口径

- `/get_status` 的本地模型资源字段由「显存」改为「NPU HBM 占用」口径；`deployment` 段改为报告 NPU profile 与设备标识。
- `/get_status` 是本 change 中唯一允许调整字段语义的接口；`/v3/seacraft_asr` 的请求与响应字段 MUST NOT 变更。
- 资源压力诊断从 `nvidia-smi` 改为 `npu-smi`，GPU 显存水位改判 NPU HBM 水位；「测试结束后验证回收」中「Qwen 8806 仍保持运行」改为「Qwen 推理已内嵌进程、不存在独立上游进程」。

### 交付与清理

- Docker 交付改为昇腾底座（`--privileged --runtime=ascend --ipc=host` + `ASCEND_VISIBLE_DEVICES`），移除三份 CUDA profile Dockerfile 与 TensorRT engine 打包逻辑。
- 按职责清理纯 NVIDIA 实现、engine 工装和不可复用脚本；通用音频/阈值/准入工具改写保留，历史标定证据和归档 OpenSpec 保留。历史文档先区分现行指导与证据，不按 CUDA 关键词批量删除。
- 逐项核实其他明确废弃的 NVIDIA open change；当前 NPU change 保留到验收归档，不要求实施期间 `openspec list` 为空。
- `AGENT.md` 与 `README.md` 改为 NPU 910B 口径。

## Capabilities

### New Capabilities

- `npu-910b-runtime`: 昇腾 910B 目标硬件的运行时契约，含依赖栈矩阵、虚拟环境边界、容器设备绑定方式、设备编号规则、dtype 与算子约束，以及 NPU 资源观测口径。
- `local-qwen-asr-inference`: 进程内 Qwen3-ASR 推理运行时，含本地权重加载、内存音频直传、语种注入、输出解析、并发闸门、实例生命周期与失败语义。

### Modified Capabilities

- `offline-asr-service`: 明确 v3 序列化和 `gpu_time_ms` 原计时边界不变，允许保持段独立性的物理批执行。

- `qwen-asr-v2-adapter`: 模型发现改为本地权重加载；转写调用改为进程内推理；语言参数语义由「HTTP 透传」改为「processor 注入」；并发上限由「HTTP 请求数」改为「本地推理任务数」。
- `local-model-inference-backends`: 移除 TensorRT 后端与资产校验，实际后端收敛为 `pytorch_cpu` / `torch_npu`；移除 CUDA Docker profile 强制策略；CAM++ 算子替换必须保持数值等价。
- `local-model-runtime`: 「统一命名」与「显存明细」需求的 NVIDIA 口径改为 NPU HBM 口径；设备数组语义扩展到 `npu:N`。
- `service-operations`: 移除四个 TensorRT 参数需求；运维状态按 NPU 后端与 HBM 口径展示；部署前置条件与 Docker 入口改为昇腾形态。
- `container-model-packaging`: 移除 TensorRT engine 资产结构与 CUDA 12.4 镜像要求，改为昇腾底座与 torch_npu 依赖交付。
- `resource-pressure-diagnostics`: GPU 资源采集、告警闸门与归因报告改为 NPU HBM 口径；上游 Qwen 进程存活检查改为内嵌推理检查。

## Impact

- **核心代码**：`core/qwen_asr_client.py`（改为本地推理客户端，保留对外错误码与 `normalize_language_param` 契约）、`core/local_inference.py`（CAM++ `seg_pooling` 算子替换、dtype 对齐）、`core/config.py`、`core/models.py`、`core/status.py`、`core/deployment.py`、`api/routes/asr_qwen_common.py`。
- **删除文件**：`core/tensorrt_runner.py`、`core/model_assets.py`、三份 CUDA profile Dockerfile 及 TensorRT 打包脚本。
- **配置**：`config.toml` 与 `docker/config.toml`。
- **测试**：`tests/test_tensorrt_config.py`、`tests/test_docker_deployment.py` 等与 TensorRT/CUDA 强绑定的用例需改写为 NPU 口径；`tests/test_qwen_v2.py` 的客户端替身需切换到本地推理接口。
- **对外契约**：`/v3/seacraft_asr` 的请求与响应字段不变；`/get_status` 的本地模型与 deployment 字段语义随 NPU 口径调整。
- **风险**：Qwen3-ASR-1.7B-hf 实测在 NPU 上单段 4.1 秒音频推理耗时 13.11 秒、NPU 占用 3.80 GiB，该历史单样本提示吞吐风险但不构成同条件后端对比；吞吐与并发档位需在迁移完成后重新标定，`speaker_similarity_threshold` 等既有阈值需在 NPU 精度下复验。

## 补充审计与实施门槛

- 开发限定 `xjtu910b7` 宿主 `npu:7`、conda `asr`，不使用其他卡，不停止卡 7 既有服务；共享卡干扰须记录。
- 三个小模型目录及其关键权重/配置已现场确认存在，emotion2vec 标签文件也存在；这不代表 checkpoint 匹配、加载或 NPU 前向通过，Qwen 权重单独核验。
- 递归发现 52 WAV、2 MP3，合计 5,144,064,354 字节；52 WAV 文件头均为 16 kHz/单声道/16 位，共约 43.66 小时。目录名存在 GBK 原始字节，增加清单和路径兼容，不覆盖或重新上传样本。
- 所有 requirements 与构建脚本统一 NPU 依赖来源和版本；补 torchaudio 配套锁定、attrs、文本处理与测试依赖，修正 av 版本冲突。安装元数据不代替前向验证。
- 保留 VAD spawn 固定进程池；主服务内承载增强模型及原生 Qwen，修正“三个小模型都同进程”的旧表述。
- `gpu_time_ms` 保留原字段、字符串类型、主链路计时起止和口径；NPU 内部指标另记。
- 交付文件适配属于迁移范围；设备绑定参数放 Compose/宿主入口，容器内 start.sh 只做预检和启动。实际镜像构建与上线在服务稳定后执行。
- 详细证据见 [migration-review.md](migration-review.md)，补充任务已纳入 [tasks.md](tasks.md)。本次只修改规划，不表示业务代码、依赖或验收已完成。
