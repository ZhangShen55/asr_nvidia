# Design

## Context

现有服务只有一条 Paraformer 为主、Whisper 为辅的离线转写链路（`/v1.1.8/seacraft_asr`），模型全部随镜像内置。Qwen3-ASR 以独立容器形式部署在 12 服务器上，服务地址 `http://127.0.0.1:8806`（容器 `qwen3-asr-0.6b-8806`，内部监听 8000），提供 OpenAI 兼容接口且不需要 API Key。

对目标容器实测得到的事实（决定本次设计的形状）：

- 容器为 `qwenllm/qwen3-asr:latest`，内部是 vLLM 0.14.0 + `qwen-asr` 0.0.4，以 `qwen-asr-serve` 启动，模型 `Qwen3-ASR-0.6B` 可通过 `GET /v1/models` 获取（`max_model_len=4096`、`max-num-seqs=32`、`--enforce-eager`）。
- `POST /v1/audio/transcriptions` 支持 `response_format` 为 `text` / `json` / `verbose_json`，但 **`verbose_json` 对该模型被服务端拒绝**（`Currently do not support verbose_json for Qwen3-ASR-0.6B`）：`qwen-asr` 的 vLLM 后端未声明 `supports_segment_timestamp`，且容器未加载 forced aligner（`return_time_stamps=True` 需要额外的 `Qwen3-ForcedAligner` 权重）。**结论：Qwen 服务不返回任何时间戳，只有文本。**
- 返回文本带模型原生的 `language <语种><asr_text>` 前缀；服务端把长音频按 `max_model_len` 自动切成多 chunk、每 chunk 独立生成后再拼接，因此长音频的返回文本里会出现多个前缀且丢失 chunk 边界。
- 服务端对单次请求的文件大小有上限（`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`），实测 95 MB 的音频被直接拒绝（`Maximum file size exceeded`）。
- **`language=auto` 不被支持**：传入 `language=auto` 会被服务端以 400 拒绝并返回支持语种清单；不传 `language` 字段（或传空串）时模型自动识别语种。传具体语种（如 `en`）处理中文音频仍返回 `Chinese`，说明该参数只是软提示。
- 本地 VAD（FunASR FSMN-VAD，`model/speech_fsmn_vad_zh-cn-16k-common-pytorch`）实测：`test/audio_file/教师1.wav` 前 5 分钟切出 45 段，段长 0.63–8.01 s（均值 3.20 s、中位 2.89 s），段间静音间隙中位 0.83 s、P90 2.49 s，语音占比 144 s / 300 s。
- 实测单段请求端到端时延：3 s 段 0.39 s、4 s 段 0.47 s、6.5 s 段 0.67 s、8 s 段 0.87 s。
- 12 服务器宿主 IP 为 `192.168.29.12`，Qwen 容器宿主侧暴露端口 8806；本阶段不依赖两个容器同网络，统一走 `IP:8806`。
- GPU 现状：Qwen 0.6B 容器运行在 GPU 0（与 1.7B 容器、stream 容器同卡，约 20.7 GB），GPU 1 占用约 2.6 GB，GPU 2 占用约 11.3 GB。

## Goals / Non-Goals

**Goals:**

- 在不改动 `/v1.1.8/seacraft_asr` 任何行为的前提下，新增 `/v2/seacraft_asr`，把 Qwen3-ASR 接入离线转写服务。
- 时间轴准确：响应分段的起止时间来自 VAD 边界，不依赖估计。
- 并发与排队参数全部外置到 `config.toml`，运维可调。
- 用 CAM++ 与 emotion2vec 产出真实 `role` / `emotion`（本变更内实现、默认关闭），且与转写并行，不拖慢主链路。

**Non-Goals:**

- 不引入 Qwen3-ForcedAligner 等额外模型（评估结论见「决策 10」，本期不做，列为可选后续增强）。
- 不做流式转写、不做说话人实时跟踪。
- 不改动 v1.1.8 的语言分流、错误码与响应口径。
- 不在本变更内替换 Paraformer 链路；两套链路并存。
- 本阶段不依赖容器间 Docker 网络，Qwen 服务地址以 `IP:8806` 配置。

## Decisions

### 1. 新增独立 v2 路由，复用既有响应工具

新增 `api/routes/asr_v2.py`（薄路由）与 `api/routes/asr_qwen_common.py`（链路实现），`main.py` 挂载新路由。v1.1.8 的 `asr_common.py` 保持原样，两条链路只共享 `utils/feature_utils.py`（语速统计）、`utils/itn.py`（文本规范化）与 `utils/audio_utils.py`（音频预处理/裁剪）。

选择理由：用户明确要求 v1.1.8 一行不改；共用响应工具可以保证 v2 的 `text` / `speed_info` 口径与 v1.1.8 一致，避免两侧再次分叉。

备选方案：在 `asr_common.py` 内按参数分支。放弃原因是 v1.1.8 链路已经承载 Paraformer/Whisper/分块/重试等复杂分支，再叠加 Qwen 分支会让两条语义完全不同的链路互相干扰。

### 2. VAD 段与 Qwen 请求的映射：只按 VAD 逐段切，不做合并

**评估过程。** 用户提出的问题是「一个音频 VAD 后得到多段，这些段是否需要合并；合并后能否仍以一个 VAD 段作为 `segment_text` 粒度」。判断这件事的前提是时间戳从哪来：

| 方案 | 时间戳来源 | 文本与时间的对应 | 请求数（1 小时讲座，语音占比 40%，均值 3.2 s/段） | 评价 |
|------|-----------|----------------|------------------------------------------|------|
| A. 逐段发送（1 VAD 段 = 1 请求 = 1 分段） | VAD 段边界，误差约 0.1–0.3 s | 精确对应 | 约 450 | **采用** |
| B. 合并后发送 | 合并区间 | 变粗：合并窗口内文本只能挂在整个窗口上 | 取决于合并阈值，显著减少 | 不采用 |
| C. 合并后按标点/字数比例回切到 VAD 段 | 估算 | 误差随窗口增大线性放大，且切点不可验证 | 同 B | 不采用 |
| D. 引入 Qwen forced aligner 拿词级时间戳 | 模型强制对齐 | 精确到词，可任意切分 | 同 A，但需额外模型与 GPU | 列为后续可选增强 |

关键约束是：**Qwen 服务端不返回时间戳**（`verbose_json` 不支持、容器未加载 forced aligner），所以本地 VAD 是唯一可靠的时间轴来源。一旦把多个 VAD 段合并成一次请求，Qwen 返回的是一整段文本而没有任何内部时间戳，想再切回「一个 VAD 段一个 `segment_text`」就只能靠标点或字数比例去猜（方案 C），在 30 s 窗口上误差可以到数秒，而且无法验证——这正好回答了用户的疑问：**合并之后无法再维持 VAD 段的 `segment_text` 粒度，除非引入 forced aligner。**

**结论。** 采用方案 A：1 个 VAD 段 = 1 次 Qwen 请求 = 1 个响应分段，`bg` / `ed` 直接取 VAD 段边界。**合并方案整体取消**（不引入 `merge_enabled` / `merge_max_gap_s` / `merge_max_duration_s` 等配置，也不实现对应逻辑）：既然合并之后无法再维持 VAD 段的 `segment_text` 粒度，而用户明确要求分段粒度稳定，就不必为不确定的吞吐收益牺牲时间轴精度。

补充取舍说明：

- 逐段发送会丢失跨静音间隙的上下文。实测段间间隙中位 0.83 s，多为自然停顿，Qwen 逐段处理后语义完整（实测「那个我们那个交的课程论文那个封面和去年的封面是一样的」这类跨句表达完整保留），因此这一损失可接受。
- 逐段请求数偏多，但单请求很快（3–8 s 音频 0.4–0.9 s），配合「每音频 4 路 × 8 音频 = 32 路」全局并发，1 小时讲座的 Qwen 侧总耗时约 1 分钟量级，可接受。
- 逐段发送还顺带规避了服务端按 `max_model_len` 自动切 chunk 造成的上下文割裂与重复风险。

### 3. VAD 在服务本地用 FunASR 完成，独立懒加载

`core/models.py` 增加独立 VAD 单例（`AutoModel(model=settings.vad_model_dir, device=..., vad_kwargs={max_single_segment_time: 30000, max_end_silence_time: 800})`），与 Paraformer 链路现有的 VAD 模型实例分开，避免 `showSpk` 开关影响 v2。VAD 推理在 `asyncio.to_thread` 中执行，不阻塞事件循环。

选择理由：镜像内已有该模型，无需新增交付物；本地 VAD 让时间轴与 Qwen 解耦，Qwen 容器只需提供转写能力。

备选方案：把整段音频直接丢给 Qwen，让它自己切。放弃原因是它既不返回时间戳，也会被单请求文件大小上限（实测 95 MB 被拒）挡住，长课堂录音基本无法直接发送。

### 4. Qwen 客户端：httpx 异步 + 原生前缀解析

新增 `core/qwen_asr_client.py`：

- 启动时 `GET {base_url}/v1/models` 缓存模型标识（配置已填模型名时跳过）；调用失败时在首次请求时重试一次发现流程，仍失败则返回 `4011`。
- 转写用 `POST {base_url}/v1/audio/transcriptions`，multipart 字段 `file` / `model`，`language` 字段按「决策 4.1」的规则可选携带，不携带任何鉴权头。
- 解析返回文本中的 `language <语种><asr_text>` 前缀：按正则切分并拼接各片段文本，取第一个非 `None` 语种；前缀**只用于解析，绝不进入 `segment_text`**；文本为空或语种为 `None` 时判定该段无有效语音。
- 超时（`request_timeout_s`，默认 120 s）、连接错误、5xx 按 `retry_count`（默认 1）重试，指数退避；耗尽后返回 `4012`。

选择理由：httpx 自带连接池与超时控制，适合高并发；Qwen 服务本身是 OpenAI 兼容实现，无需引入 openai SDK 依赖。

### 4.1 language 参数透传规则（auto 不下发）

`/v2/seacraft_asr` 的 `language` 请求参数保留并默认 `auto`，但**不下发给 Qwen**：

| 请求参数取值 | 行为 |
|--------------|------|
| 未提交 / 空串 / `auto` | 请求体**不包含** `language` 字段，由模型自动识别语种（服务端不支持 `auto`，实测会 400 拒绝） |
| 具体语种（如 `zh`、`en`） | 原样透传给 Qwen；服务端不支持该语种时返回 400，系统转换为 `{"msg": "不支持的语言", "code": 4009}` |

响应中的 `language` 取模型实际识别出的语种（Qwen 原生语种名，如 `Chinese` / `English`），而不是请求参数的回显。

### 4.2 语种占比字段 `language_ratio`

响应顶层新增一个字典字段 `language_ratio`，用于展示不同语言的占比，键为模型识别出的语种名，值为该语种文本字符数占总字符数的百分比（保留一位小数，各值之和为 100.0）。

- 统计口径：按 VAD 段逐段取「该段识别语种 → 该段去前缀后的文本字符数」累加；纯空白字符不计入。
- 单语种音频返回 `{"Chinese": 100.0}`；没有任何有效文本时返回空字典 `{}`。
- 该字段只做展示，不参与 `text` 与 `segments` 的构造，也不影响其它字段。

### 5. 两级并发控制：音频级排队 + 请求级限流

并发约束分两级，都由配置驱动：

- **L1 音频级**（`audio_concurrency`，默认 8）：同时进入处理的音频文件数上限。实现为「自建处理槽位计数 + 条件变量」：
  - 进入处理前 `acquire`：若当前在处理数 < 上限则占用槽位并立即返回；否则按配置决定是等待还是直接拒绝。
  - **默认直接拒绝**（`queue_wait_timeout_s = 0`）：第 9 个并发音频立刻返回 `{"msg": "处理队列已满，请稍后再试", "code": 4013}`，不占用上游任何等待时间，把背压立刻反馈给业务方。
  - 若运维把 `queue_wait_timeout_s` 配成大于 0，则最多等待该时长后仍未获得槽位则返回 `{"msg": "请求排队超时，请稍后再试", "code": 4010}`；等待队列容量 `queue_max_size`（默认 64）满时同样返回 `4013`。
  - 无论成功、失败还是异常，槽位 MUST 在 `finally` 中释放；所有临时文件（原始上传文件、每段裁剪的 WAV）MUST 在 `finally` 中删除，确保不泄漏、不占磁盘。
- **L2a 段级**（`segment_concurrency`，默认 4）：**单个音频内部**同时发往 Qwen 的 VAD 段请求数上限，每个音频各持一个 `asyncio.Semaphore`。
- **L2b 全局请求级**（`request_concurrency`，默认 32）：所有音频合计发往 Qwen 的**单次 HTTP 请求**数上限，用一个全局 `asyncio.Semaphore` 控制。

选择理由：用户提出的第 5、6、7 点分别对应这两级，两级解耦后既能限制 Qwen 侧压力，又能限制本服务的内存/临时文件占用。

备选方案：复用 `core/concurrency.py` 的 `acquire_gpu_slot`。放弃原因是它服务于本地 GPU 推理，采用 5 秒轮询式等待（延迟粒度太粗）、且语义是「GPU 槽位」而非「音频槽位」；另外 Qwen 的 GPU 占用在另一个容器内，本地信号量管不到。

**三层并发参数的区别（避免歧义）：**

| 配置项 | 含义 | 默认值 |
|--------|------|--------|
| `audio_concurrency` | 同时处于处理中的**音频文件**数量 | 8 |
| `segment_concurrency` | **单个音频内**同时发往 Qwen 的 VAD 段请求数 | 4 |
| `request_concurrency` | **所有音频合计**发往 Qwen 的请求数上限（全局） | 32 |

**并发关系。** 按用户方案：`segment_concurrency (4) × audio_concurrency (8) = 32`，正好等于全局上限 `request_concurrency (32)`。也就是说 8 个音频各自最多同时跑 4 个 VAD 段请求，理论上把 Qwen 打满到 32 路，与容器 `--max-num-seqs 32` 的名义上限对齐。

**必须标注的风险（留给压测回答，不在此擅自改参数）。** Qwen 容器 `--max-num-seqs 32` 是「同时在飞的最大序列数」，但实测其 KV cache 实际只支撑约 16 路并发。因此 32 路在飞时，vLLM 会在**容器内部**排队（部分请求被调度延迟或抢占重算），表现为尾延迟上升，高负载下还可能出现幻觉/重复。本变更按用户给定的 32 配置实现，把「32 路是否真的跑满算力、从几路开始出现幻觉/重复」列为压测必须回答的问题；若实测 32 路明显劣化，处置方案（下调 `request_concurrency` 或 `segment_concurrency`、或改用容器名 + 独立 Qwen 实例）由压测结论决定后再定，不在设计阶段先行改写。

### 6. 响应封装与 v1.1.8 同构

v2 响应顶层字段与 v1.1.8 一致：`language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms`，**并额外增加 `language_ratio` 字典字段**（见决策 4.2）。分段字段：`segment_text`、`bg`、`ed`、`speed`、`segment_words`（空数组）、`role`（`null`）、`emotion`（`null`）。`text` 为规范化后的分段文本按空格拼接，复用 `utils/itn.py`（`zh` / `en` 走规则包，其他语言跳过）。`speed` 与 `speed_info` 复用 `utils/feature_utils.py`。

选择理由：用户要求「封装成与 v1.1.8 同样的响应」，同时保留 `segment_words` / `role` / `emotion` 三个字段占位，为后续真实取值预留结构。

`language` 字段在 v2 中取 Qwen 返回的识别语种（原生名称，如 `Chinese` / `English`），而非 v1.1.8 的请求语言回显。

### 7. 真实 role / emotion 的异步并行计算（本变更内实现，默认关闭）

结论：**在本变更内实现**，但作为独立任务组、默认关闭，避免拖累 v2 主链路的交付节奏。分两路并行、与 Qwen 转写同时进行：

- **说话人链路**：用 CAM++ 对每个 VAD 段提取说话人嵌入 → 聚类得到 `spk_0 / spk_1 ...` → 结合关键词特征（复用 `utils/feature_utils.py` 的 `extract_features` / `identify_teacher`）推断 `teacher` / `studentN`。
- **情绪链路**：用 emotion2vec 对每个 VAD 段（长度在 1–10 s 之间时判定，其余记为「平淡」）推断情绪，经 `EMOTION_LABEL_MAP` 映射为业务标签。

实现上以 `asyncio.gather` 与「Qwen 转写」并发执行：**每个 VAD 段的转写任务与该段的 role/emotion 任务在同一时刻发起**，所有段的转写任务与所有段的 role/emotion 任务一起被 `asyncio.gather` 汇总，两者都完成后再组装响应。**明确不做「等全部转写完成后再算 role/emotion」的串行实现**，也不做「转写完成一段才触发一段」的分段级流水线改造——并行是本方案的既定形态。任一路失败则对应字段回退为 `null`，转写结果照常返回。

**加载与触发条件（按用户要求收敛到已有开关）：**

- **模型加载**：CAM++ 与 emotion2vec 仅在配置 `[features] open_spk = true` **且** `open_emotion = true` **同时成立**时才加载；任一为 false 都不加载（省掉 emotion2vec 约 1.9 GB 权重与推理显存）。
- **实际计算**：请求参数 `showSpk` / `showEmotion` 为 true 时才真正跑对应模型；两者默认 false（与 12 服务器上 v1.1.8 的默认值保持一致）。模型未加载而请求开启了对应开关时，对应字段回退为 `null`，不影响转写结果。

**效率要点**：role 与 emotion 两条链路与 Qwen 转写**并行**（`asyncio.gather`）——Qwen 走 HTTP 到独立容器，role/emotion 走本机 GPU，三者互不阻塞；本机 GPU 侧对 CAM++ 与 emotion2vec 的调用用**可配置并发闸门**（`[qwen_asr].local_gpu_concurrency`，默认 4）限制并发路数，而不是完全串行——完全串行会让 8 路并发时 role/emotion 成为端到端时延的主瓶颈（实测 D 组 P50 增幅 +166%），限制并发后降到 +113%（见第 12 节与测试报告）。整体仍与 Qwen 请求真正并行，因为 Qwen 的算力在另一个容器里。

**CAM++ 的输入方式**：CAM++ 直接喂 numpy 数组而不是裁剪后的 WAV 路径——funasr 的文件分支里多出一次音频读取与重采样，实测单次调用 92.7 ms → 42.6 ms，嵌入结果逐位相同。情绪链路仍走文件路径，因为 fp16 的 emotion2vec 用 numpy 输入会触发 dtype 不匹配（`FloatTensor` vs `HalfTensor`）。

**精度取舍**：emotion2vec 以**半精度（fp16）**加载，精度直接减半即可，不为情绪精度做额外调优——业务上「有结果就行」。这样权重占用从约 1.9 GB 降到约 1 GB 级别，显存余量更大；emotion 标签的准确率以 v1.1.8 链路为参照基线做粗略比对即可，不设高门槛。CAM++ 的说话人嵌入仍用常规精度，因为说话人聚类的余弦相似度对精度更敏感。

### 8. 配置项

`config.toml` 与 `config.docker.toml` 新增 `[qwen_asr]` 段：

```toml
[qwen_asr]
enabled = true
base_url = "http://127.0.0.1:8806"      # 本地；容器部署用 http://192.168.29.12:8806
model = ""                              # 留空表示从 /v1/models 自动发现
request_concurrency = 32                # 所有音频合计同时发往 Qwen 的请求数
audio_concurrency = 8                  # 同时处理的音频文件数
segment_concurrency = 4                # 单个音频内同时发往 Qwen 的 VAD 段请求数
queue_max_size = 64                     # 等待队列容量
queue_wait_timeout_s = 0                # 0 = 队列满立即拒绝；>0 才排队等待
request_timeout_s = 120                 # 单次 Qwen 请求超时
connect_timeout_s = 10
retry_count = 1
min_segment_duration_s = 0.3            # 过滤过短 VAD 段
vad_max_single_segment_time_ms = 30000
vad_max_end_silence_time_ms = 800
```

新增 `[model_device]` 段，控制 v2 链路三个本地模型的落卡，**三者绑定在同一张卡上，默认 GPU 0**：

```toml
[model_device]
vad = 0                 # FunASR FSMN-VAD
emotion2vec = 0         # emotion2vec_plus_large
speech_campplus = 0     # speech_campplus_sv_zh_en_16k-common_advanced
```

`core/config.py` 为上述每项提供同名属性与默认值（读取 `[qwen_asr]` 段，缺项回落到默认值 16 / 8 / 队列满立即拒绝 / GPU 0）。

说明与边界：

- 该段只作用于 **v2 链路**新增的三个模型实例；`/v1.1.8` 链路继续使用原有的 `device` / `ngpu` 配置，行为不变。
- v1.1.8 的 CAM++ 是嵌入在 FunASR 转写流水线里的（`spk_model` 参数），无法拆成独立实例，因此 v2 的说话人链路是**独立的一份 CAM++**（权重仅 27 MB，可接受）。
- v2 的 emotion2vec 也是独立实例（权重 1.9 GB）。若 v1.1.8 与 v2 同时活跃，emotion2vec 可能驻留两份（约 3.8 GB）；实施时在文档中说明「同一时刻只跑一条链路」的规避方式。

### 9. 部署与镜像

镜像不引入 Qwen 模型，仅复用已有的 VAD（role/emotion 开启时再加 CAM++ 与 emotion2vec）。

**服务地址**：本阶段**不依赖两个容器同网络**，统一用「IP + 8806」：`config.toml` 写 `http://127.0.0.1:8806`（本机开发），`config.docker.toml` 写 `http://192.168.29.12:8806`（12 服务器）。进入正式部署阶段时再切换为容器名 + 同网络。

**GPU 分配**：Qwen 容器当前占用 GPU 0。为避免压测时算力数据互相污染，v2 链路的 VAD / CAM++ / emotion2vec 通过 `[model_device]` 显式指定落卡，默认 GPU 0（按用户要求）；**压测时建议改到 GPU 1**（当前仅占 2.6 GB），以获得干净的 Qwen 侧算力曲线。两种配置都写进文档，由运维按压测结论决定。

### 10. Qwen3-ForcedAligner 的评估与结论

用户提出的问题是「加 ForcedAligner 会多耗多少算力、能否让结果更准、是否必须用 GPU」。评估结论：

| 问题 | 结论 |
|------|------|
| 是否提升**转写准确率** | **不提升**。forced aligner 只做「已有文本 ↔ 音频时间轴」的对齐，不改变识别出的文字内容。 |
| 能提升什么 | **只提升时间戳精度**：词级绝对偏移约 33–47 ms，可把 `segment_words` 从空数组变成真实词级时间戳，并允许在段内自由切分。 |
| 是否需要 GPU | 需要。0.6B 模型 bf16 权重约 1.8 GB，推理显存约 2 GB，且必须以 `AutoModel.from_pretrained(device_map="cuda:0", dtype=bfloat16)` 在**我们容器内**跑——vLLM 服务端不提供该能力（`verbose_json` 因此被拒）。 |
| 其它限制 | 仅支持 11 个语种（中文之外的多数语种不支持），对中英课堂场景价值有限；5 分钟音频对齐耗时 < 2 s，耗时不是瓶颈。 |

**结论：本期不做。** 理由是它只能解决时间戳精度（当前 VAD 边界误差 0.1–0.3 s 已满足业务需要）、需要额外 1.8 GB 交付物 + 约 2 GB 显存、且只能在本地进程内跑从而与 VAD / CAM++ / emotion2vec 抢同一张卡。列为后续可选增强。

### 11. 验证方案

- **单元测试**：VAD 段映射、前缀解析、language 透传规则、语种占比统计、并发/排队行为、响应结构与错误码，用桩（stub）替换 Qwen HTTP 调用，不依赖真实容器。
- **效果测试**：`test/audio_file/` 下 `储能学术英语实践.wav`（英文）、`教师1.wav`（中文课堂）、`课堂实录.wav`，对比 v1.1.8 的转写文本与时间戳，检查漏字、重复、时间戳偏移。
- **性能测试**：1 / 2 / 4 / 8 路并发，观察 Qwen 容器所在 GPU（GPU 0）的算力占用与 SM 利用率、端到端时延、排队时延；逐档位提高到出现明显幻觉或重复为止，记录边界并写入测试报告。

### 12. role / emotion 的专项测试方案

role 与 emotion 在本变更内实现，必须有独立的效果与性能验证，否则「并行不拖慢主链路」的说法无法证实。方案分三块：

**（1）效果验证——以 v1.1.8 链路为参照基线**

- 同一音频、同一套 VAD 切分参数下，v2 的 `role` / `emotion` 与 v1.1.8 的 `role` / `emotion` 逐段比对：role 比较「角色语义」（`teacher` / `studentN` / `spk_N`）而非编号本身；emotion 比较业务标签。
- 目标：role 语义一致率 ≥ 90%，emotion 标签一致率 ≥ 80%；不一致样本人工核查并记录。
- 聚类阈值调参：对 CAM++ 余弦相似度阈值做 sweep，以「同一说话人跨段标签稳定、不同说话人不被合并」为准则选出生产取值。**实测结论见决策 13**：原计划 sweep 的 0.6 / 0.7 / 0.75 / 0.8 全部严重过分割，sweep 区间因此向下扩展到 0.15。
- 长音频稳定性：`教师1.wav` 全量跑一遍，检查同一说话人在前 5 分钟与后 5 分钟的标签是否漂移（整段音频统一聚类的关键风险点）。

**（2）性能验证——四组开关 × 三档并发**

| 组 | showSpk | showEmotion | 目的 |
|----|---------|-------------|------|
| A | false | false | 转写基线 |
| B | true | false | 说话人链路的增量开销 |
| C | false | true | 情绪链路的增量开销 |
| D | true | true | 两者同时开启的真实开销 |

每组在 1 / 4 / 8 音频并发下记录：端到端时延（P50 / P95）、每 VAD 段平均耗时、吞吐（音频/分钟、段/秒）、本机 GPU（`[model_device]` 指定卡）的 SM 利用率与显存占用增量、Qwen 容器侧时延是否被拖慢。

- **并行是否真的并行**：对比 A 与 D 在同一并发档位的 P50 时延，目标是 D 的增幅 ≤ 30%。**该指标只用于判定并行方案是否成立，不作为改架构的依据**：若增幅超限，处置方式是把实测数据与可选手段（下调并发、降精度、把 role/emotion 挪到独立 GPU）列给业务方决策，**不在实现中擅自改成串行或分段级流水线**——并行是本方案的既定形态（见决策 7）。
- **显存**：记录 emotion2vec（fp16 权重约 1 GB）+ CAM++ 的显存增量峰值，确认不超过目标卡余量；若仍紧张，优先改为按需加载。
- **互不污染**：role/emotion 是纯后处理，开启后 D 组的转写文本必须与 A 组**逐段完全一致**；一旦出现差异，说明并发下 VAD 段与 Qwen 响应的对应关系被破坏，属阻断性缺陷。

**（3）稳定性验证**

- 连续 8 音频并发 + role/emotion 全开跑 30 分钟，确认无显存持续增长、无文件描述符泄漏、无槽位泄漏。
- 注入异常：单段音频损坏、emotion2vec 推理失败、Qwen 超时，确认 role/emotion 回退为 `null` 而转写结果照常返回，且槽位与临时文件正常释放。

### 13. CAM++ 聚类阈值的实测标定（对决策 12 的修正）

决策 12 计划对余弦相似度阈值 sweep `0.6 / 0.7 / 0.75 / 0.8`。实测发现这四个取值全部**严重过分割**，因此把 sweep 区间向下扩展到 `0.15`，并改用 **BCubed F1** 作为判据（贪心最佳匹配对「簇数多」有系统性偏好，不能用）。

**方法**：`scripts/speaker_threshold_sweep.py` 抽一次 CAM++ 嵌入后离线扫阈值；以 `/v1.1.8` 的说话人标签（同一 CAM++ 权重，作为参照真值）按时间重叠映射到 v2 的段上；BCubed 同时惩罚「把不同说话人合并」（precision 下降）与「把同一说话人拆散」（recall 下降）。

**结果**（每档取 10 分钟窗口；`教师1` 与 `储能学术英语实践` 为单说话人，`课堂实录` 为三说话人）：

| 阈值 | 教师1（F1 / 簇数） | 储能学术英语实践（F1 / 簇数） | 课堂实录（F1 / 簇数，真值 3 簇） |
|------|-------------------|------------------------------|------------------------------|
| 0.15 / 0.20 / 0.25 / 0.30 | **1.000 / 1** | **1.000 / 1** | 0.815 / 2 ~ 3 |
| 0.35 | 0.990 / 2 | 0.973 / 2 | **0.823 / 4** |
| 0.45 | 0.971 / 3 | 0.915 / 5 | 0.788 / 5 |
| 0.60 | 0.941 / 6 | 0.601 / 26 | 0.592 / 16 |
| 0.70 | 0.661 / 22 | 0.321 / 45 | 0.292 / 27 |
| 0.75 | 0.439 / 35 | 0.117 / 52 | 0.159 / 35 |
| 0.80 | 0.227 / 52 | 0.041 / 63 | 0.089 / 43 |

**结论：生产取值 `speaker_similarity_threshold = 0.30`。** 该值下两个单说话人音频各聚为 1 簇（F1 = 1.0，无虚假拆分），三说话人的课堂实录聚为 3 簇（与真值一致，F1 = 0.816）。原计划中的 0.6 以上取值会把同一说话人拆成 6~63 簇，F1 跌破 0.6，不可采用。

### 14. 实现过程中确定的技术细节

以下细节在实现阶段由实测确定，补充进设计以保持制品与代码一致：

- **VAD 实例池**（`core/vad.py`）：FunASR 的 `FsmnVADStreaming` 内部带状态缓存，**同一个实例不可并发复用**，因此用「实例池 + 空闲标记」隔离，并发度由 `[qwen_asr].vad_concurrency`（默认 4）限制。
- **`cache={}` 显式传参**：`FsmnVADStreaming.inference` 的 `cache` 是**可变默认参数**（默认 `{}`），多个并发调用会共享同一份内部状态并抛出 `IndexError`。`model.generate(..., cache={})` 必须每次显式传入独立的 cache。
- **张量裁剪替代 ffmpeg**：逐段调用 ffmpeg 裁剪 118 段耗时 7.97 s；改为在已加载的音频张量上切片再 `torchaudio.save`，同规模只需 0.25 s（快约 30 倍）。该优化只在 `ctx.audio_tensor` 可用时启用，否则回退 ffmpeg。
- **唯一临时文件名**：v2 链路并发上传同名文件会互相覆盖 `/tmp/{filename}`，因此 v2 走 `prepare_asr_context(unique_tmp=True)`，生成带 UUID 前缀的临时文件名；v1.1.8 保持原行为不变。
- **响应语种**：顶层 `language` 取 Qwen 实际识别出的语种（`Chinese` / `English`），而非请求参数回显；`language_ratio` 的统计口径为「各段语种 → 该段去前缀后的文本字符数」。

### 15. 镜像依赖分层：`httpx` 放在运行层而不是基础依赖层

v2 链路的 Qwen 客户端需要 `httpx`。最初的写法是直接改写 `requirements-pip.txt`，实测**破坏了 Docker 构建缓存**：基础依赖层的 `COPY requirements-pip.txt` 内容变化 → `RUN pip install` 层缓存失效 → 重新下载 torch（766 MB）与 torchaudio，实测在当前带宽（~200 KB/s）下需要 1 小时以上。

**最终做法**：

- `requirements-pip.txt` 保持与历史构建**逐字节一致**，基础依赖层全程命中缓存；
- 新增 `requirements-pip-runtime.txt`（仅 `httpx`），在 Dockerfile 的**运行层**单独 `COPY` + `pip install`（实测该层 4.6 s）；
- 宿主机 conda 环境仍在 `requirements.txt` 中新增 `httpx`。

代价是 `httpx` 不在 `requirements-pip.txt` 里，需要在 Dockerfile 与 `docs/DOCKER_BUILD_LOG.md` 中说明；收益是新增依赖不再触发大包重下。同样的分层思路已在 `requirements-pip-extra.txt`（小语种与 ITN 依赖）上验证过。

**容器验证结果（v3.2）**：镜像构建成功，依赖层全部 CACHED；容器内 `/openapi.json` 同时暴露 `/v1.1.8/seacraft_asr` 与 `/v2/seacraft_asr`，v2 真实音频转写、role/emotion 填充、错误码（4008 / 4009）均符合预期，临时文件无残留。

## Risks / Trade-offs

- **逐段请求导致请求数放大**（1 小时讲座约 450 次）→ 用「每音频 4 路 × 8 音频」消化，实测单请求 0.4–0.9 s，预计整体 1 分钟量级；不通过合并换吞吐。
- **短段幻觉**：Qwen 对极短音频（< 0.5 s）可能输出无意义文本 → 过滤小于 `min_segment_duration_s` 的段，且丢弃解析后为空的段。
- **时间戳边界偏保守**：VAD 段边界落在静音处，文字实际起点可能比 `bg` 略晚，误差约 0.1–0.3 s → 可通过调 `vad_max_end_silence_time_ms` 收紧，并在测试报告中给出实测偏差。
- **服务端并发上限**：Qwen 容器 `--max-num-seqs 32` 但实际 KV cache 约只支撑 16 路，32 路在飞时 vLLM 会在容器内部排队并拉长尾延迟，高负载下可能出现幻觉/重复 → 按 32 实现并用压测定位真实饱和点，再决定是否下调。
- **服务端单请求文件大小上限**（实测 95 MB 被拒）→ 本地切段后单请求仅数百 KB，天然规避。
- **Qwen 容器不可用时 v2 整体不可用** → 与 v1.1.8 互不影响，v1.1.8 仍可服务；v2 返回明确的 `4011` / `4012`。
- **role/emotion 的模型显存占用**与 Qwen 争抢 → 仅在 `open_spk` 与 `open_emotion` 同时为 true 时加载，压测时通过 `[model_device]` 把本服务放到 GPU 1。
- **emotion2vec 重复驻留**：v1.1.8 与 v2 各持一份实例时约占 3.8 GB → 文档中说明「同一时刻只跑一条链路」。
- **队列满即拒绝会削峰**：默认 `queue_wait_timeout_s = 0`，突发流量下第 9 个请求直接失败 → 上游按 `4013` 做退避重试；若业务更希望排队，把该值调大即可，无需改代码。

## Migration Plan

1. 合并代码与配置（`[qwen_asr]` 默认 `enabled = true`），本地用 `asr` conda 环境起服务，验证 `/v2/seacraft_asr`。
2. 构建镜像（不清缓存），用 `http://192.168.29.12:8806` 形式的 `base_url` 验证容器内可访问 Qwen 服务。
3. 跑 1/2/4/8 路并发压测，产出测试报告，确认幻觉/重复边界后确定生产 `request_concurrency`（32）、`segment_concurrency`（4）与 `audio_concurrency`（8）取值。
4. 回滚方式：把配置中的 `qwen_asr.enabled` 置为 `false` 或直接回退镜像，`/v1.1.8` 不受影响。

## Open Questions

- 响应新增字段命名：当前定为 `language_ratio`（顶层字典，值为各语种文本字符占比百分比）。若业务侧更希望 `language_percent` / `language_distribution`，可在实现前调整。
- `request_concurrency` 默认值定为 32（= `segment_concurrency` 4 × `audio_concurrency` 8，与 Qwen `--max-num-seqs 32` 对齐）。压测若显示 32 路在容器内部排队并明显劣化，将下调到 16 或 8，最终取值以测试报告为准。
- 说话人聚类是否跨整段音频统一进行（同一说话人跨段同标签），当前设计为整段音频统一聚类。
- 后续是否引入 Qwen3-ForcedAligner 以获得词级时间戳（当前决策：本期不做，评估结论见决策 10）。
