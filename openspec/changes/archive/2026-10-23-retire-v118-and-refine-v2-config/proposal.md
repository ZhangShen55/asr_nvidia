# Proposal

## Why

`/v1.1.8/seacraft_asr`（SeACo-Paraformer + PUNC + Whisper 法语链路）已被 Qwen3-ASR 的 `/v2/seacraft_asr` 取代并判定淘汰，但代码、模型、依赖与配置项仍在镜像中常驻：启动即加载一份 SeACo-Paraformer（连带 PUNC、VAD、CAM++），镜像里还要塞进已不使用的 Paraformer/PUNC/Whisper 模型，配置文件里也残留着一批无人读取的历史参数。与此同时，实测发现 ITN 文本规范化会引入「段首标点、连续标点、纯标点段」等脏输出，且 Qwen 服务侧并发能力（`--api-server-count`）被本服务的默认并发参数束缚，需要一次把接口、模型、配置与输出质量一起收敛。

## What Changes

**接口下线（BREAKING）**

- 移除 `POST /v1.1.8/seacraft_asr` 路由及其链路实现，请求该路径返回未找到该路由。
- 服务对外只保留 `POST /v2/seacraft_asr`、`GET /get_status`、`DELETE /clear_tasks_list`。

**模型与依赖收敛**

- 镜像不再内置 `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`、`punc_ct-transformer_cn-en-common-vocab471067-large`、`faster-whisper-large-v3` 三个模型目录。
- 移除 `faster-whisper` / `ctranslate2` 运行时依赖；保留 WeTextProcessing 与 pynini（ITN 仍需要）。
- 移除 v1.1.8 的模型加载路径（`load_models_if_needed`）与 Paraformer/Whisper 转写实现。

**v2 本地模型开关独立化**

- 新增 `[features] open_v2_role`（CAM++ 说话人）与 `[features] open_v2_emotion`（emotion2vec 情绪）两个独立开关，默认均为 `true`。
- 移除只服务于 v1.1.8 的 `[features] open_spk`、`open_emotion`、`ban_hotword`。

**配置项清理**

- 删除从未被读取的 `ncpu`、`concurrency`、`hotword_path`。
- `id_engine` 不再从配置读取，但 `/get_status` 保留该字段并返回固定占位值，避免运维平台字段结构变化。
- `version` 不再来自配置，固定返回 `latest`。
- 删除 v1.1.8 专属的 `device`、`ngpu`、`[model_paths]` 的 `asr_model_dir` / `punc_model_dir` / `whisper_model_dir`、`[audio_chunk]` 整段、`[features].open_mul_lang`。
- 保留 `log_path` 并新增日志保留策略：最多 7 天、单文件最多 100 MB。
- 保留 `instance_count`（控制容器内 uvicorn 实例数量）但默认值改为 1；保留 `[qwen_asr]`、`[model_device]`、`[itn]`、`[speech_rate]`。

**输出质量修正**

- 语种占比字典的键改用简写（`Chinese` → `zh`、`English` → `en`），语种判定逻辑不变，未知语种原样透传。
- ITN 之后增加分段文本卫生规则：整段只有标点符号时丢弃该分段；分段文本以标点开头时去掉开头标点。
- ITN 保留并默认开启（`[itn] enabled = true`）；`remove_interjections` 保持默认 `true`（继续过滤「呃」「啊」等语气词），其把语气词替换成标点带来的脏输出由上述标点卫生规则兜底清理。
- 顶层 `language` 字段维持返回 `Chinese` / `English` 全称，不随 `language_ratio` 的键改简写。

**进程命名与显存可观测**

- v2 链路的三个本地模型（VAD、CAM++、emotion2vec）仍在同一服务进程内加载，进程名统一覆写为 `asr-offline-model`，使运维能在 `nvidia-smi` 的进程列表中直接定位本服务占用的显存。
- `instance_count` 默认值改为 1；多实例时每个实例各持一份模型权重，显存随实例数等比放大，关系显式可预期。
- `/get_status` 额外暴露三个本地模型各自的加载状态与显存明细，作为 `nvidia-smi` 聚合视图下的区分手段。

**NPU（昇腾 910B）可行性调研**

- 产出 `docs/npu-910b-feasibility-report.md`：调研 VAD、CAM++、emotion2vec 三个模型在 torch_npu 上的运行可行性、改造点与风险，回答本项目能否在 NPU 与 GPU 上都跑通。
- Qwen-ASR 的 NPU 部署不在本次调研范围。

**吞吐配套**

- 在部署文档中记录 Qwen 容器按 `--api-server-count 4` 等参数扩容的方式与实测收益；本服务侧的并发默认值维持不变（实测 Qwen 不再是瓶颈后，提高本服务侧并发无额外收益）。

## Capabilities

### New Capabilities

- `service-operations`: 服务运行期的可观测与配置契约——日志保留策略、状态接口字段、实例数量语义、配置文件有效项清单。
- `local-model-runtime`: v2 本地模型的运行时承载约定——统一进程名、单进程承载、懒加载、显存明细暴露与实例数对显存的影响。

### Modified Capabilities

- `offline-asr-service`: 移除 v1.1.8 接口与配套的请求/响应契约、语言分流、Whisper 法语链路、长音频分块与语速相关说明，接口集收敛为 v2 + 状态 + 清队列。
- `qwen-asr-v2-adapter`: 语种占比字典改用简写键；新增分段文本标点卫生规则；ITN 开关与参数语义调整；v2 本地模型加载改为独立开关；并发参数默认值维持。
- `container-model-packaging`: 镜像只交付 v2 链路所需的三个模型目录，移除法语模型的条件校验与 Whisper 运行时，保留 ITN 规则包运行时。

## Impact

- **代码**：`api/routes/asr_v18.py`（删除）、`api/routes/asr_common.py`（瘦身为 v2 共用工具）、`api/routes/asr_qwen_common.py`（标点卫生规则、语种缩写、开关解耦）、`api/routes/status.py` 与 `core/status.py`（`id_engine` 固定占位、`version` 固定、本地模型显存明细）、`core/config.py`（配置项增删、`instance_count` 默认 1）、`core/models.py`（删除 v1.1.8 模型加载）、`core/vad.py`、`core/logging.py`（保留策略）、`main.py`（路由装配、进程命名）、`utils/qwen_response.py`（语种缩写映射）、`utils/itn.py`（后处理钩子）。
- **接口**：`/v1.1.8/seacraft_asr` 移除（BREAKING）；`/v2/seacraft_asr` 的 `language_ratio` 键值与分段文本内容规则变化；`/get_status` 响应字段变化。
- **镜像**：`Dockerfile` 的模型目录校验从五个减为三个；`requirements-pip-extra.txt` 移除 faster-whisper 相关依赖；镜像体积下降。
- **配置**：`config.toml` / `config.docker.toml` 字段结构变化，部署方需按新结构同步；新增 `setproctitle` 运行时依赖。
- **不在本次范围**：构建镜像与部署上线需单独确认后执行（本 change 只做后端运行、测试与效果比对）；NPU 适配落地（只做调研报告）；Qwen 容器参数属于部署侧配置，不在本仓库代码内。
