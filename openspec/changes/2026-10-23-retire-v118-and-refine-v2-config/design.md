# Design

## Context

当前仓库同时承载两条转写链路：`/v1.1.8/seacraft_asr`（本地 SeACo-Paraformer + PUNC，Whisper 兜底法语）与 `/v2/seacraft_asr`（本地 VAD 切段 + 外部 Qwen3-ASR 容器转写 + 本地 CAM++ / emotion2vec）。前者已被业务判定淘汰，但其代码、模型与依赖仍在镜像中常驻。

本轮调研在 12 服务器（`192.168.29.12`）上实测得到的事实（决定本设计的形状）：

**接口与模型的占用**

- `core/models.py` 的 `load_models_if_needed()` 在 FastAPI 启动钩子里执行，只要 `open_spk=true` 就会加载一份 SeACo-Paraformer，并把 PUNC、VAD、CAM++ 打包进同一个 `AutoModel`；`faster-whisper` 在 `open_mul_lang=true` 时另行加载。也就是说容器启动即占掉一条已经不再提供服务的链路的显存与启动时间。
- `api/routes/asr_qwen_common.py` 只从 `api/routes/asr_common.py` 取三样东西：`AsrContext`、`map_emotion`、`prepare_asr_context`；其余（`run_paraformer_asr`、`run_whisper_asr`、`_apply_itn`、长音频分块）都是 v1.1.8 专属，可以随接口下线删除。
- 镜像中三个待下线模型的体积：`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`、`punc_ct-transformer_cn-en-common-vocab471067-large`、`faster-whisper-large-v3`。

**配置项审计**（逐键对照 `core/config.py` 的读取点）

| 配置项 | 读取方 | 结论 |
|---|---|---|
| `ncpu`、`concurrency` | 无 | 删除 |
| `hotword_path` | 无（热词来自请求参数 `request.hotWords`） | 删除 |
| `id_engine` | 仅 `core/status.py` / `api/routes/status.py` | 配置项删除，状态接口保留该字段并返回固定占位值 |
| `version` | 仅状态接口 | 删除配置项，代码固定返回 `latest` |
| `device`、`ngpu` | v1.1.8 模型加载 | 删除，v2 用 `[model_device]` |
| `[model_paths].asr_model_dir` / `punc_model_dir` / `whisper_model_dir` | v1.1.8 | 删除 |
| `[audio_chunk]` 五项 | v1.1.8 长音频分块 | 删除整段 |
| `[features].open_mul_lang` | Whisper 加载 | 删除 |
| `[features].open_spk` / `open_emotion` | v1.1.8 模型加载 **且** v2 的 CAM++/emotion2vec 加载 | 拆分：v2 改用 `open_v2_role` / `open_v2_emotion` |
| `[features].ban_hotword` | v1.1.8 热词 | 删除 |
| `log_path`、`instance_count`、`[speech_rate]`、`[itn]`、`[qwen_asr]`、`[model_device]` | 仍在使用 | 保留（`log_path` 扩展保留策略） |

**ITN 是否需要（实测，4 个新音频，`showSpk`/`showEmotion` 全关，ITN 开与关两套服务逐段对比）**

| 音频 | 段数 | 被 ITN 改写 | 其中数字规范化 | 纯标点段 | 段首标点 | 连续/异常标点 |
|---|---|---|---|---|---|---|
| 习近平新时代…（中文） | 540 | 281（52%） | 76 | 2 | 44 | 36 |
| 教师1 3.aac（中文） | 548 | 131（24%） | 7 | 2 | 17 | 58 |
| 专业学位英语（英文） | — | — | — | — | — | — |
| 南航上课铃声 | 2 | 0 | 0 | 0 | 0 | 0 |

结论：**ITN 需要保留**。`二零二五年 → 2025年`、`那第十三章 → 那第13章`、`四点六升 → 4.6升` 这类改写是刚需，关闭 ITN 等于把数字原样吐给业务方。但 `remove_interjections=true` 是有害的：它把「呃」「啊」替换成逗号，实测造成 44 段以 `，` 开头、36~58 段出现 `，，` / `，。` 这类连续标点，并产出只有 `。` `！` 的纯标点段。这正是用户要求的两条标点卫生规则的来源。

**Qwen 侧吞吐（按 `docs/qwen-asr-v2-throughput-optimization.md` 的方案 A 实测）**

Qwen 层直打（118 个约 2.5 s 片段）：

| 并发 | 基线容器 8806 | 方案 A（`--api-server-count 4` + `--max-model-len 1024` + `--max-num-seqs 128`） | 倍数 |
|---|---|---|---|
| 8 | 8.79 段/s | 62.9 ~ 64.7 段/s | 7.4× |
| 16 | 17.23 段/s | 110.3 ~ 114.5 段/s | 6.4× |
| 32 | 32.53 段/s | 84.3 ~ 138.9 段/s | 2.6 ~ 4.3× |
| 64 | 38.69 段/s | 131.7 段/s | 3.4× |
| 128 | — | 183.7 段/s | — |

方案 A 的 KV cache 从 62,384 token 提升到 149,936 token，并发上限从 15.23× 提升到 146.42×。GPU 0 峰值利用率从 13~16% 提升到 82~92%。

端到端（`zh600.wav` 10 分钟 / 118 段，纯转写）：

| 并发 | 基线 | 方案 A（本服务参数不变） | 方案 A + 本服务 `segment_concurrency=16`/`request_concurrency=128` |
|---|---|---|---|
| 1 | 19.97 s / 5.91 段/s | 8.91 s / 13.24 段/s | 4.92 s / 24.0 段/s |
| 2 | 21.79 s / 10.83 段/s | 9.69 s / 24.35 段/s | 7.62 s / 30.96 段/s |
| 4 | 28.18 s / 16.75 段/s | 16.55 s / 28.52 段/s | 16.08 s / 29.36 段/s |
| 8 | 45.54 s / 20.73 段/s | 32.16 s / 29.36 段/s | 35.56 s / 26.54 段/s |

开启 `showSpk` / `showEmotion` 后（方案 A，8 并发）：119.85 s / 7.88 段/s，GPU 0 平均利用率仅 3.5~5.2% —— 瓶颈回到本地模型的 Python/GIL，Qwen 侧扩容帮不上忙，与吞吐文档 §6.5 的判断一致。

## Goals / Non-Goals

**Goals:**

- 让镜像与运行时只承载 v2 链路真正需要的模型、依赖与配置。
- 解除 v2 的 CAM++ / emotion2vec 加载与 v1.1.8 开关的耦合。
- 配置文件中只保留实际生效的项，并让日志体积有明确上限。
- 修正 ITN 带来的标点脏输出，并把语种占比字典的键统一为简写。
- 给出 Qwen 侧扩容的落地方式与本服务侧的配套参数取值结论。
- 产出 NPU（昇腾 910B）上 VAD / CAM++ / emotion2vec 的可行性调研报告，回答项目能否在 NPU 与 GPU 上双跑通。

**Non-Goals:**

- 不改 `/v2/seacraft_asr` 的接口路径、请求参数与响应字段集合（`language_ratio` 的键值取值与分段文本内容属于既有字段的取值变化）。
- 不改 VAD 切段策略、时间轴口径与说话人聚类阈值。
- 不在本次构建镜像或部署上线（需单独确认）。
- 不引入 Qwen 多后端负载均衡（方案 B）：实测方案 A 已能把 Qwen 层吞吐提升 3~7 倍，多后端的客户端改造收益不足。
- 不改 Qwen 容器本身的参数（属于部署侧配置），只在文档中记录。
- 不落地 NPU 适配（只做调研报告）；设备抽象层与 NPU 镜像流水线属于后续独立 change。
- 本次不构建镜像、不部署上线。

## Decisions

### 1. v1.1.8 采用「删代码」而不是「加开关」

删除 `api/routes/asr_v18.py` 及其在 `main.py` 的注册；`api/routes/asr_common.py` 瘦身为 v2 共用工具，只保留 `AsrContext`、`map_emotion`、`prepare_asr_context`、`cleanup_temp_files`、`generate_task_id`；`core/models.py` 删除 `load_models_if_needed` 与 Whisper/情绪（v1.1.8 版）加载路径，只保留 v2 的 `load_v2_models_if_needed`。

备选方案是保留代码并加 `enable_v118=false` 开关。否决原因：业务已明确「v1.1.8 已淘汰」，保留死代码会让模型目录校验、依赖清单与配置项继续背着三份已下线能力的重量，配置清理也无从谈起。

`prepare_asr_context` 里的热词分支（`request.hotWords` + `ban_hotword`）随 v1.1.8 删除——热词是 v1.1.8 的请求参数，v2 的请求体没有该字段。`ban_hotword` 随之从 `[features]` 移除。

### 2. v2 本地模型改用独立开关 `open_v2_role` / `open_v2_emotion`

`load_v2_models_if_needed` 当前的 `role_ready = settings.open_spk and settings.open_emotion` 有两个问题：语义错位（`open_spk` 是 v1.1.8 的能力开关），且把两个模型绑死成一个组合开关。新配置：

```toml
[features]
open_v2_role = true      # CAM++ 说话人嵌入
open_v2_emotion = true   # emotion2vec 情绪
```

关闭任一个，对应模型的推理入口直接返回 `null`、不加载模型，对应响应字段保持 `null`。两个开关互相独立，避免「只想用情绪却被迫加载 CAM++」。默认值都为 `true`，保持当前线上行为（两个模型按需懒加载，请求不带 `showSpk`/`showEmotion` 时不加载）。

### 3. 配置项按「有则必用、无则必删」重建

新配置结构（`config.toml` 与 `config.docker.toml` 同构）：

```toml
id_engine = "1"          # ← 删除（配置项不再读取，状态接口返回固定占位）
version = "..."          # ← 删除（代码固定 latest）
device = "cuda:1"        # ← 删除
ngpu = 1                 # ← 删除
ncpu = 4                 # ← 删除
concurrency = 5          # ← 删除
instance_count = 1       # ← 保留（默认值由 4 改为 1，见决策 11）
log_path = "./asr_service.log"   # ← 保留

[log]                    # ← 新增
max_days = 7
max_size_mb = 100

[model_paths]            # ← 只留三个
vad_model_dir = "..."
spk_model_dir = "..."
emotion_model_dir = "..."

[speech_rate]            # ← 保留（v2 的 speed_info 也用它）
rate_factor = 0.4

[features]               # ← 重写
open_v2_role = true
open_v2_emotion = true

[itn]                    # ← 保留
enabled = true
enable_standalone_number = true
enable_0_to_9 = false
remove_interjections = true    # ← 保留过滤语气词（业务确认），副作用由标点卫生规则兜底
failure_mode = "passthrough"

[qwen_asr]               # ← 保留，取值不变
...

[model_device]           # ← 保留
vad = 0
emotion2vec = 0
speech_campplus = 0
```

`id_engine` 不再从配置读取，但 `/get_status` 仍输出该字段并返回固定占位值（沿用原配置的默认值），避免运维平台的采集脚本因字段缺失而报错。

### 4. 日志保留：按天 + 按大小双轮转

`core/logging.py` 现有的 `RotatingFileHandler` 只能二选一（按大小或按时间），无法同时满足「7 天 + 100 MB」。实现一个自定义 handler：写入前检查当天文件大小，超过 `max_size_mb` 就切出 `asr_service.log.1`（序号递增，超过上限的序号直接覆盖最旧的）；每天首次写入前切分出带日期的文件；启动时扫描日志目录，删除修改时间超过 `max_days` 的文件。

备选方案：只用 `TimedRotatingFileHandler(when="D", backupCount=7)`。否决原因：单文件一天内可能超过 100 MB（实测 8 并发半小时可产生数百 MB 级日志），无法满足体积上限。

日志文件命名保持 `asr_service.log` 为当前文件，历史文件用 `.1`、`.2` 后缀，避免运维脚本路径失效。

### 5. ITN 保留并默认剔除语气词，标点卫生规则兜底

实测证明数字规范化有业务价值，而 `remove_interjections` 会把语气词替换成逗号，制造段首标点、连续标点与纯标点段。业务明确要求继续过滤「呃」「啊」等语气词，因此 `remove_interjections` 保持默认 `true`（`config.toml` 中显式配置），并由 ITN 之后的标点卫生规则清理其副作用。

标点卫生规则（在文本规范化之后执行，与 ITN 开关无关）：

1. 分段文本 `strip()` 后若只剩标点符号 → 该分段从 `segments` 中移除，同时从整段文本拼接中排除；
2. 分段文本首个字符是标点 → 移除开头**连续的所有标点**（如 `，，就。` → `就。`），因为 ITN 产生的 `，，` 往往成对出现，只去一个会留下残缺的开头；首位必须落到有效字符。

规则的「标点」判定集合取中英文常用标点：`，。！？；：、,.?!;:~…—–""''（）()《》【】`。

> 说明：规则 2 选择「移除开头连续的所有标点」而不是「只移除一个」，是基于 ITN 实测中 `，，` / `，，` 的成对出现；业务已确认「首位应该是有效字符」，故连续序列整体移除。

### 6. 语种占比字典改用简写键

在 `utils/qwen_response.py` 增加一张「模型语种名 → 简写」映射表，命中则替换键，未命中则原样透传。映射表覆盖实测出现过的全部语种并留出常见扩展：

`Chinese→zh`、`English→en`、`Cantonese→yue`、`German→de`、`Italian→it`、`Portuguese→pt`、`Spanish→es`、`French→fr`、`Thai→th`、`Malay→ms`、`Japanese→ja`、`Korean→ko`、`Russian→ru`、`Hindi→hi`、`Arabic→ar`、`Vietnamese→vi`、`Indonesian→id`、`Dutch→nl`、`Turkish→tr`、`Polish→pl`、`Greek→el`、`Hebrew→he`。

语种判定逻辑（解析 Qwen 原生前缀）完全不动，只改键名——实测三个教学音频的语种占比本来就与真实授课语言吻合（教师1 中文 99.9%、课堂实录 英文 63.8%、储能英文 91.9%），不需要调整判定。

### 7. 并发默认值维持不变

吞吐文档 §7 建议把 `request_concurrency` 放大到 128、`segment_concurrency` 放大到 16。实测**不采纳**：Qwen 侧按方案 A 扩容后，端到端 8 并发吞吐在本服务参数不变时为 29.36 段/s，放大到 `segment_concurrency=16`/`request_concurrency=128` 后反而是 26.54 段/s（同一时段内的测量波动范围内，无实质收益），因为瓶颈已经转移到本服务自身的音频裁剪与 Python/GIL。

放大 `segment_concurrency` 还有副作用：单个音频会占用更多全局配额，8 音频并发时公平性变差。因此本服务侧默认值保持 `segment_concurrency=4` / `request_concurrency=32` / `audio_concurrency=8`，扩容收益由 Qwen 服务侧承担。运维若确认 Qwen 已按方案 A 扩容，可按需自行调大。

### 8. 镜像瘦身与依赖清理

- `Dockerfile` 的必需模型目录校验从五个减到三个（VAD、CAM++、emotion2vec），并删除 `faster-whisper-large-v3` 的条件校验分支。
- `requirements-pip-extra.txt` 中的 `faster-whisper` / `ctranslate2` / `importlib_resources` 删除；该层若只剩 WeTextProcessing，则并入基础依赖层的既有安装步骤，保持构建缓存命中（**构建时不清理缓存**）。
- WeTextProcessing 与 pynini 保留：ITN 仍需要，删除会让 `[itn].enabled=true` 静默降级为透传。

### 9. 本次不构建镜像

按项目约定，未在 change 中明确要求前不构建镜像、不部署。本 change 只改 Dockerfile 与配置文件；镜像构建与上线作为独立动作，需要单独确认后执行。

### 10. v2 本地模型单进程承载，进程名统一为 asr-offline-model

**目标**：让运维能在 `nvidia-smi` 的进程列表中一眼定位「哪一行是本服务占的显存」。

**机制约束（2026-10-23 在本机 3×4090 上实测）**：

| 做法 | nvidia-smi 进程名 | 结论 |
|---|---|---|
| 单进程内加载三个模型 | 一行（python 完整路径） | 无法按模型区分，CUDA context 按 PID 归集 |
| 子进程内 `prctl(PR_SET_NAME)` 改 comm | 仍显示 python 完整路径 | 无效，nvidia-smi 不读 comm |
| 子进程内 `setproctitle.setproctitle("vad")` | 正确显示 `vad` / `cam` / `emo` 三行 | 机制可行，nvidia-smi 读 `/proc/<pid>/cmdline` 的 argv[0] |
| 单进程内 `setproctitle.setproctitle("asr-offline-model")` | 正确显示 `asr-offline-model` | **本方案采用** |

**架构**：

- VAD、CAM++、emotion2vec 三个模型继续在同一个服务进程内加载，不引入子进程与跨进程 RPC。
- 服务进程启动后调用 `setproctitle.setproctitle("asr-offline-model")`，`nvidia-smi` 中该进程显示为 `asr-offline-model`。
- **模型仍然懒加载**：只有对应开关打开且请求需要时才加载权重，未使用的模型不占显存。
- 由于 `nvidia-smi` 只能给出聚合显存，`/get_status` 额外暴露三个模型各自的加载状态与显存明细，作为区分手段。

**被否决的方案：按模型拆进程**

拆成 `vad` / `cam` / `emo` 三个进程确实能在 `nvidia-smi` 中逐模型定位，但代价是每个进程各自创建 CUDA context，
实测**仅 context 就占 386 MiB**，从单进程拆三进程净增约 0.8 GB 显存，并且模型调用从函数调用退化为跨进程 RPC
（序列化、异常路径、超时与背压处理全部复杂化）。业务侧明确不接受这份额外显存开销，因此否决该方案。

折中：保留 `setproctitle` 的机制价值（统一命名，避免同机多个 Python 服务难以分辨），把「逐模型区分」下放到 `/get_status` 的显存明细。

### 11. instance_count 默认 1，实例数即显存倍数

`instance_count` 保留（容器内 uvicorn 实例数量），但默认值从 4 改为 **1**：

- 默认单实例 → `nvidia-smi` 只有一行 `asr-offline-model`；
- 配置为 N → N 个实例进程，每个实例各自持有一份 VAD / CAM++ / emotion2vec 权重，本地模型显存占用为单实例的 N 倍。

该关系是显式且可预期的：多实例用于提升 HTTP 层吞吐，代价是本地模型显存等比放大，运维按需权衡。

### 12. NPU（910B）可行性只做调研，不在本 change 落地

用户新增需求：调研 VAD、CAM++、emotion2vec 三个模型在华为昇腾 910B（torch_npu）上的运行可行性，
产出报告放在 `docs/` 下，回答「本项目能否在 NPU 与 GPU 上都跑通」。Qwen-ASR 的 NPU 部署不在调研范围。

本 change **只产出调研报告**，不改代码、不引入设备抽象层、不建 NPU 镜像流水线——这些属于后续独立 change。
报告结论与后续改造点见 `docs/npu-910b-feasibility-report.md`。

## Risks / Trade-offs

- [删除 v1.1.8 路由是破坏性变更，仍有调用方在用] → 在发布说明中明确标注，提供回滚镜像（保留 `v3.3` 标签）以便快速切回。
- [顶层 `language` 字段返回 `Chinese` 这类全称，而 `language_ratio` 的键改成了 `zh`] → 已确认保持现状：顶层字段维持全称，不做改动，调用方按字段各自取值。
- [`remove_interjections` 保持 `true` 会把语气词替换成逗号] → 已确认保留过滤语气词的行为，由标点卫生规则清理由此产生的段首标点、连续标点与纯标点段。
- [ITN 关闭时也会执行标点规则] → 已确认始终执行；规则是输出卫生要求，与 ITN 无关。
- [`nvidia-smi` 只能看到聚合显存，无法逐模型区分] → 已在 `/get_status` 暴露三个模型各自的加载状态与显存明细；放弃「nvidia-smi 里逐模型区分」的需求。
- [`instance_count > 1` 时本地模型显存倍增] → 默认值改为 1，把放大倍数变成显式可预期的配置选择；文档中说明 N 实例 = N 倍权重显存。
- [镜像瘦身同时改变了模型目录校验，缺模型时构建失败] → 构建期报错信息里保留缺失目录名，定位成本低。
- [Qwen 侧按方案 A 扩容后，`--max-model-len` 从 4096 降到 1024] → 30 秒 VAD 段的 prompt 约 400 token，生成预算 624 token（最坏情况 5 倍余量），但 1024 以上的长段会被 vLLM 直接 400；需要确认 VAD 最长段参数（`vad_max_single_segment_time_ms`，当前 30000）不被调大。

## Migration Plan

1. 合入代码与配置变更，跑通单元测试（`tests/`）。
2. 在宿主环境用 `config.toml` 启动，验证 `/v2/seacraft_asr` 的 `showSpk`/`showEmotion` 组合、`language_ratio` 简写键、分段标点规则与 ITN 开关。
3. 确认 `/v1.1.8/seacraft_asr` 返回 404，`/get_status` 字段符合预期。
4. 效果比对与验收：在宿主环境用 `test/audio_file` 下的音频跑 `/v2/seacraft_asr`，对照本 change 之前的转写结果，确认文字内容无回退、`language_ratio` 简写键正确、标点脏输出已消除。
5. **本次不构建镜像、不部署上线**（用户明确要求）：镜像构建与上线作为独立动作，需单独确认后执行。
6. Qwen 容器按方案 A 参数扩容（部署侧动作）：`--api-server-count 4 --max-model-len 1024 --max-num-seqs 128 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.45`，启动后核对 `GPU KV cache size` 与 `Maximum concurrency` 两行日志。
7. 回滚策略：镜像回滚到 `v3.3` 并把 Qwen 容器参数恢复原值即可，配置文件改动集中在 `config.toml`，可按旧结构还原。

## Open Questions

- （无待确认项：显存区分方案、`instance_count` 默认值、进程命名三项已由业务侧拍板，见决策 10、11）
