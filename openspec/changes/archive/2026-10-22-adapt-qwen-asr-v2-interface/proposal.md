# Proposal

## Why

12 服务器上已经部署好 Qwen3-ASR 容器（`qwen3-asr-0.6b-8806`，OpenAI 兼容接口），但当前离线 ASR 服务只有基于 Paraformer 的 `/v1.1.8/seacraft_asr` 一条转写链路，模型能力、语种覆盖和并发形态都无法利用 Qwen3-ASR。现需要在**不改动 `/v1.1.8` 行为**的前提下，新增一条 `/v2` 适配链路：本地做 VAD 切分 → 并发调用 Qwen3-ASR → 按 `/v1.1.8` 的响应结构封装回传，并把并发度、排队与模型发现等运维参数从代码中解耦到配置文件。

## What Changes

- 新增 `POST /v2/seacraft_asr` 接口：请求参数与 `/v1.1.8` 保持一致（multipart 表单），响应顶层字段与分段结构保持同构；`/v1.1.8/seacraft_asr` 的行为、参数默认值与错误码**完全不动**。
- 新增 `api/routes/asr_v2.py` 与 `api/routes/asr_qwen_common.py`：实现「音频预处理 → VAD 切分 → 并发调用 Qwen3-ASR → 时间轴回填 → 响应封装」的完整链路。
- VAD 复用镜像内已有的 `speech_fsmn_vad_zh-cn-16k-common-pytorch`（FunASR），在服务本地完成，保证不依赖 Qwen 容器。
- **VAD 段与 Qwen 请求的映射策略**：只按 VAD 逐段切，「1 个 VAD 段 = 1 个 Qwen 请求 = 1 个 `segment_text` 粒度」，以 VAD 段边界作为响应时间轴；**不做相邻段合并**。理由：Qwen3-ASR 的 OpenAI 兼容接口不返回时间戳（`verbose_json` 对该模型不支持，服务端也未加载 forced aligner），时间轴只能由 VAD 提供，合并后无法再切回 VAD 段的文本粒度。
- 响应封装时 `segment_words` 保留为空数组、`role` 与 `emotion` 保留为 `null` 字段，字段结构先对齐 `/v1.1.8`，真实取值后续再补。
- 响应顶层新增字典字段 `language_ratio`：展示各识别语种的文本字符占比（百分比），用于多语种场景的可视化。
- `language` 请求参数保留、默认 `auto`；`auto` 或未提交时**不下发** `language` 字段（Qwen 服务端不支持 `auto`，实测会 400 拒绝），由模型自动识别语种；提交具体语种时原样透传。
- 新增 `core/qwen_asr_client.py`：封装 `/v1/models` 模型发现、`/v1/audio/transcriptions` 调用、返回文本中 `language <xx><asr_text>` 前缀的解析、失败重试与超时控制。
- 新增 `core/qwen_asr_queue.py`（或等价实现）：三级并发控制——「同时处理的音频文件数」（`audio_concurrency`，默认 8，**队列满立即拒绝**，不静默排队）、「单个音频内同时发往 Qwen 的段请求数」（`segment_concurrency`，默认 4）与「全部音频合计发往 Qwen 的请求数」（`request_concurrency`，默认 32，即 4 × 8）；参数全部下沉到 `config.toml`。
- `config.toml` / `config.docker.toml` 新增 `[qwen_asr]` 配置段（服务地址、模型、并发度、队列、VAD 参数等）与 `[model_device]` 配置段（VAD / emotion2vec / speech_campplus 的落卡，三者绑定同一张卡，默认 GPU 0）。
- 新增性能与效果测试：基于 `test/audio_file/` 下三个音频，在 1/2/4/8 路并发下观测 GPU 算力占用、端到端时延、转写准确度与时间戳准度，并定位出现幻觉/重复的并发边界；产出可归档的测试报告。
- 在本变更内实现真实 `role` 与 `emotion`（独立任务组、默认关闭）：用镜像内 `speech_campplus_sv_zh_en_16k-common_advanced` 与 `emotion2vec_plus_large` 产出结果，与 Qwen 转写异步并行执行，避免串行等待；两个模型仅在 `open_spk` 与 `open_emotion` 同时为 true 时加载。

## Capabilities

### New Capabilities

- `qwen-asr-v2-adapter`: 面向 Qwen3-ASR 的 `/v2` 转写接口能力，涵盖 VAD 切分与时间轴映射、Qwen 请求并发与音频排队、模型自动发现、响应结构封装，以及后续真实 role/emotion 的异步并行计算。

### Modified Capabilities

- `offline-asr-service`: 「只暴露三项业务接口」的要求需要调整为在保留 `/v1.1.8/seacraft_asr`、`/get_status`、`/clear_tasks_list` 的同时新增 `/v2/seacraft_asr`；`/v1.1.8` 自身的契约要求保持不变。

## Impact

- API：新增 `POST /v2/seacraft_asr`；`/v1.1.8/seacraft_asr` 及其余已有接口的行为不变。
- 代码（新增）：`api/routes/asr_v2.py`、`api/routes/asr_qwen_common.py`、`core/qwen_asr_client.py`、`core/qwen_asr_queue.py`、`core/vad.py`（VAD 切分与分段策略）、`utils/qwen_response.py`（响应解析与文本抽取）。
- 代码（修改）：`main.py`（挂载 v2 路由）、`core/config.py`（新增 `[qwen_asr]` 配置属性）、`core/models.py`（按需加载独立 VAD 模型）、`config.toml`、`config.docker.toml`、`README.md`。
- 模型：不新增模型文件；复用镜像内 `speech_fsmn_vad_zh-cn-16k-common-pytorch`（VAD）、`speech_campplus_sv_zh_en_16k-common_advanced`（说话人）、`emotion2vec_plus_large`（情绪）。Qwen3-ASR 模型由独立容器提供，不进本镜像；本期也不引入 Qwen3-ForcedAligner（只提升时间戳精度、不提升准确率，需额外 1.8 GB 权重与约 2 GB 显存，评估结论见 design 决策 10）。
- 依赖：新增 `httpx`（异步 HTTP 客户端）；`funasr` VAD 推理沿用现有依赖。
- 部署：本阶段不依赖容器同网络，`base_url` 统一使用「IP + 8806」（`config.docker.toml` 写 `http://192.168.29.12:8806`），且不需要 API Key。
- 测试：新增接口契约单测与并发/性能测试脚本；性能测试在目标服务器上使用独立 GPU 观测 Qwen 容器的算力占用。
