# SeaCraftASR

基于 FastAPI 的离线语音转写服务，面向教育课堂录音。服务本地用 FunASR FSMN-VAD 做语音切分，
在华为 Ascend 910B 上用本地 **Qwen3-ASR-1.7B** 原始权重逐段转写，CAM++ 与 emotion2vec 使用原生 torch_npu。v3 请求、响应、错误码和后处理契约保持不变。

> `/v1.1.8/seacraft_asr`（SeACo-Paraformer + PUNC + Whisper 法语链路）已随本次变更下线，
> 服务只保留 `/v3/seacraft_asr`、`/get_status`、`/clear_tasks_list` 三个接口。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v3/seacraft_asr` | 离线 ASR 转写（Qwen3-ASR 链路） |
| `GET` | `/get_status` | 服务状态、任务统计与本地模型显存明细 |
| `DELETE` | `/clear_tasks_list` | 清空当前实例的处理中任务列表 |

FastAPI 自动文档位于 `/docs`、`/redoc` 和 `/openapi.json`。

## 项目结构

```text
.
├── main.py                     # 应用装配 + 进程命名（asr-offline-model）
├── config.toml                 # 本地运行配置
├── docker/                     # 昇腾镜像、Compose 和部署说明
├── api/routes/
│   ├── asr_v3.py               # /v3/seacraft_asr
│   ├── asr_qwen_common.py      # v3 主链路：VAD → 并发转写 → 响应封装
│   ├── asr_common.py           # v3 共用：请求上下文、情绪标签映射、临时文件清理
│   └── status.py
├── core/
│   ├── config.py
│   ├── logging.py              # 日志按天 + 按大小双轮转
│   ├── models.py               # CAM++ / emotion2vec 懒加载 + 显存登记
│   ├── vad.py                  # v3 链路的本地 VAD 切分
│   ├── qwen_asr_client.py      # Qwen3-ASR 本地异步客户端（懒加载 / 实例串行 / 重试）
│   ├── qwen_asr_queue.py       # v3 链路的音频排队与两级请求并发控制
│   └── status.py               # 状态接口响应组装
├── entity/data.py
├── model/                      # 模型权重，不进入 Git
└── utils/
    ├── audio_utils.py
    ├── asr_stats.py
    ├── feature_utils.py
    ├── itn.py                  # 文本规范化（WeTextProcessing）+ 标点卫生规则
    └── qwen_response.py        # Qwen 返回文本解析与语种占比统计
```

## 模型目录

模型需要完整放入 `model/` 下的三个目录：

```text
model/
├── README.md
├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── speech_campplus_sv_zh_en_16k-common_advanced/
└── emotion2vec_plus_large/
```

| 模型 | 用途 |
|------|------|
| FSMN-VAD | 语音端点检测（切分转写单元） |
| CAM++ | 说话人嵌入与说话人聚类 |
| emotion2vec_plus_large | 情绪识别（原生 FP32 基线） |

模型权重被 `.gitignore` 忽略，但会进入 Docker 构建上下文并复制到镜像 `/app/model/`。

## 本地运行

开发主机 `xjtu910b7`，源码 `/home/zhangs/asr_nvidia`，仅使用宿主物理卡 `npu:7`。
卡 7 已有其他服务，测试结果属于共享卡测量。解释器固定为 `/home/zhangs/conda-envs/asr/bin/python`。

```bash
cd /home/zhangs/asr_nvidia
set +u
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/zhangs/conda-envs/asr/bin/python scripts/check_npu_environment.py --config config.toml
CONFIG_PATH=./config.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py \
  --host 127.0.0.1 --port 8083 --workers 1
```

正式入口为 `python main.py`（支持 --host/--port/--workers 1）。关闭时向本次父进程发送 TERM，先完成 lifespan 再走 Python atexit；当前 NPU 运行时可能接管 INT，不能仅以 Ctrl+C 判断退出。等待 NPU 工作及 VAD worker 退出。测试必须记录自身 PID/进程树和端口，统一 finally 收尾并核对端口解除监听；不要按名称或端口批量终止既有服务。
启动预检校验依赖、设备及三个小模型资产，VAD 固定进程池预热后才就绪；Qwen、CAM++、emotion2vec 在首次需要时加载。Qwen 目录默认 `/home/model/Qwen3-ASR-1.7B-hf`，不再依赖外部 HTTP 服务。

依赖分为 `requirements-pip-runtime.txt`（通用运行）、`requirements-npu.txt`（与 CANN 配套的 NPU 栈）、`requirements-dev.txt`、`requirements-build.txt`。环境重建使用 `scripts/rebuild_npu_env.sh` 的独立前缀，不原地升级当前 `asr`。

### 模型内存压力诊断

`scripts/model_memory_pressure_diagnostics.py` 只附着显式 `--backend-pid` 对应的本项目实例，采集 NPU HBM、RSS/PSS、Swap、VAD worker、缓存与健康状态；它不拥有被测服务的关闭权。启动服务的测试程序负责收尾。
`/get_status` 的 `resourceDiagnostics` 保留既有业务结构，资源段改为 `npu`；采样失败用 `available=false` 和原因表达。`requestTelemetry` 有界保存阶段统计，不保存音频或正文。

音频递归清单与原始 GBK 路径映射由 `scripts/build_audio_manifest.py` 生成，见 `docs/npu-910b-migration-20261117/audio-manifest.json`；使用稳定 ASCII sample_id 上传，不重命名或覆盖原文件。

## ASR 请求

```bash
curl -X POST "http://localhost:8083/v3/seacraft_asr" \
  -F "audioFile=@sample.wav" \
  -F "language=auto" \
  -F "wordTimestamps=true" \
  -F "showSpk=true" \
  -F "showEmotion=true" \
  -F "showRoleIdentify=true"
```

主要表单参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `audioFile` | 必填 | 上传音频 |
| `language` | `auto` | 下发给 Qwen 的语言标记；`auto` 表示不指定，由模型自动识别 |
| `wordTimestamps` | `false` | 是否返回词级时间戳（v3 链路固定返回空数组） |
| `showSpk` | `false` | 是否启用说话人分离 |
| `showEmotion` | `false` | 是否返回情绪标签 |
| `showRoleIdentify` | `false` | 是否识别教师/学生角色 |

> `showSpk`、`showEmotion`、`showRoleIdentify` 的默认值均为 `false`。

响应包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms` 与 `language_ratio`。
分段中的 `role`、`emotion` 和 `segment_words` 根据请求参数返回。`text` 为最终保留分段按空格拼接并经文本规范化（ITN）处理后的结果。纯「啊 / 哦 / 嗯 / 呃」分段会被移除；英文为有效文本占比最高时，边界明确的中文口气词会转换为 `ah / oh / um / uh`。

### v3 处理流程

1. 服务本地用 FunASR FSMN-VAD 把音频切成若干语音段；
2. 每个语音段交给本地 Qwen3-ASR 实例（单段一个请求，**不做相邻段合并**）；
3. 响应分段的 `bg` / `ed` 直接取该 VAD 段的起止时间；
4. 复读修复、单字语气词移除、语言占比判定、英文口气词替换、ITN 和标点清理依次执行，最终统计只使用最终保留分段。

### 语种占比 `language_ratio`

`language_ratio` 的键为语种**简写**（`zh` / `en` / `yue` / …），值为该语种文本字符占比（百分比，保留一位小数，合计 100.0）。
模型返回的语种名不在映射表内时原样透传。顶层 `language` 字段仍返回模型返回的语种全称（如 `Chinese` / `English`）。
完整映射表（22 个语种）见 [`utils/qwen_response.py`](utils/qwen_response.py) 的 `LANGUAGE_ABBREVIATIONS`：

| 语种 | 简写 | 语种 | 简写 | 语种 | 简写 |
|------|------|------|------|------|------|
| Chinese | `zh` | English | `en` | Cantonese | `yue` |
| German | `de` | Italian | `it` | Portuguese | `pt` |
| Spanish | `es` | French | `fr` | Thai | `th` |
| Malay | `ms` | Japanese | `ja` | Korean | `ko` |
| Russian | `ru` | Hindi | `hi` | Arabic | `ar` |
| Vietnamese | `vi` | Indonesian | `id` | Dutch | `nl` |
| Turkish | `tr` | Polish | `pl` | Greek | `el` |
| Hebrew | `he` | | | | |

### 文本后处理

上游拿到的文本已经过下列处理，与模型原始输出不完全一致，属于预期行为：

1. **复读修复**：模型偶尔在长片段上产生「同一短语无限重复」的循环，服务检测到后折叠重复部分。
2. **ITN 文本规范化**（默认开启，由 `[itn] enabled` 控制）：WeTextProcessing 负责数字规范化（如「二零二五」→「2025」）与部分语气词剔除。ITN 之后还有两条与 ITN 开关**无关**、
   始终执行的标点卫生规则：
   - 整段文本只有标点符号时，该段从 `segments` 中移除；
   - 分段文本以标点开头时，移除开头**连续的所有标点**，直到落到有效字符。
3. **独立语气词剔除**：纯「啊 / 哦 / 嗯 / 呃」这类独立成段的口头禅会被移除，不计入 `segments` 与 `text`。

因此 `segments` 条数会少于音频中实际的人声段数，`text` 是最终保留分段的有序拼接。

### 错误码

| 错误码 | 含义 |
|--------|------|
| `4008` | 音频为空或未检测到任何人声 |
| `4009` | 提交的语种不被 Qwen 支持 |
| `4010` | 请求排队等待超时（`queue_wait_timeout_s > 0` 时可能触发） |
| `4011` | 本地 Qwen 目录/权重不可读、设备不可用或加载失败 |
| `4012` | 本地 Qwen 推理异常或超时（重试耗尽） |
| `4013` | 总音频处理容量满（消息包含总活动数/上限），或等待模式的共享等待队列满 |
| `4015` | 请求的 CAM++ 说话人/角色识别能力未启用或没有可用实例 |
| `4016` | 请求的 emotion2vec 情绪能力未启用或没有可用实例 |
| `4017` | 仅 B 类增强容量满，消息包含 `spk_emo` 和当前增强活动数/上限 |
| `4018` | 同时请求说话人与情绪能力，但两个功能开关均关闭 |

> 默认 `queue_wait_timeout_s = 0`：总容量满立即返回 `4013`；总容量有余量但 B 容量满返回 `4017`。
> 两项容量同时满时优先返回 `4013`。能力开关校验早于容量检查。

## 配置

主要配置项：

| 配置 | 说明 |
|------|------|
| `instance_count` | `start.sh` 启动的 uvicorn 实例数，默认 `1` |
| `[log].log_path` | 日志文件路径，默认 `./asr_service.log`；修改后重启生效 |
| `[log]` | 日志保留策略：`max_days`（默认 7）、`max_size_mb`（默认 100） |
| `[model_paths]` | 三个模型目录 |
| `[speech_rate]` | 单句语速修正系数 |
| `[features]` | `open_v2_role`、`open_v2_emotion` |
| `[itn]` | 文本规范化开关与失败策略 |
| `[qwen_asr]` | v3 链路的 本地 Qwen 目录、设备、并发度和队列 |
| `[vad]` | FSMN-VAD worker、切分参数、全局 VAD 阈值和段级过滤参数 |
| `[model_device_pool]` | CAM++ 与 emotion2vec 的设备数组；每个元素对应一个独立模型实例和一个并发槽位 |

`speed_info` 按 1、5、10 分钟窗口统计，跨窗口句子按时间重叠比例分摊内容数量。单句语速使用 `[speech_rate].rate_factor` 修正，分时段语速不乘该系数。

### `[qwen_asr]` 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 是否开放 `/v3/seacraft_asr` |
| `model` | `""` | 本地模型显示名称；不发起模型发现请求 |
| `local_model_dir` | `/home/model/Qwen3-ASR-1.7B-hf` | 原始 HF 权重与 processor 目录 |
| `local_device` | `npu:7` | 宿主目标 NPU；容器仅映射物理 7 时使用逻辑 `npu:0` |
| `local_dtype` | `bfloat16` | 支持 bfloat16/float16/float32；默认采用已验证 BF16 |
| `local_instance_count` | `1` | 独立实例数，每份复制权重；每实例串行执行 |
| `max_new_tokens` | `256` | 每段生成上限 |
| `segment_concurrency` | `1` | 单音频在途本地转写任务上限 |
| `request_concurrency` | `1` | 所有音频的在途本地转写任务上限 |
| `audio_concurrency` | `8` | A+B 同时处理的音频总数 |
| `audio_spk_emo_concurrency` | `4` | 请求任一增强能力的 B 类整音频上限，必须为正整数 |
| `queue_max_size` | `64` | A+B 共享等待队列容量 |
| `queue_wait_timeout_s` | `0` | `0` = 队列满立即拒绝；`>0` 才排队等待，超时返回 `4010` |
| `request_timeout_s` | `120` | 单次 Qwen 请求超时（秒） |
| `retry_count` | `2` | Qwen 请求失败重试次数 |
| `speaker_similarity_threshold` | `0.30` | CAM++ 说话人聚类的余弦相似度阈值（按实测 sweep 取值，阈值过高会严重过分割） |

### `[vad]` 配置项

下表列配置声明值。原FunASR通过nested kwargs传参，当前权重的FSMN实际选项来自原始model_conf：
单段8000ms、结束静音800ms、speech_noise=0.6、全局decibel=-100、全局snr=-100。
CPU原始AutoModel与原生NPU保持同一既有行为；服务段级过滤独立使用SNR18、segment_decibel=-40等参数。
不能将声明值、实际FSMN选项和段级过滤混为一层；本次不变更参数绑定或数值。
完整来源和原始选项见[课堂阈值评审](docs/npu-910b-migration-20261117/classroom-threshold-review.md)。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `vad_device` | `["cpu", "cpu", "npu:7", "npu:7"]` | 每个元素启动一个固定 FSMN-VAD worker；数组长度决定完整音频 VAD 并发 |
| `min_segment_duration_s` | `0.3` | 过滤过短 VAD 段，单位：秒 |
| `vad_max_single_segment_time_ms` | `12000` | 历史nested声明；当前权重实际单段上限8000ms |
| `vad_max_end_silence_time_ms` | `800` | VAD 结束静音时长，单位：毫秒 |
| `vad_speech_noise_thres` | `0.8` | 历史nested声明；当前模型实际采用0.6 |
| `vad_decibel_thres` | `-40.0` | 全局log能量声明，非dBFS；当前模型实际为-100，段级门槛另设 |
| `vad_snr_thres` | `18.0` | 服务段级SNR真实取值；nested声明未覆盖FSMN全局-100 |
| `vad_decibel_thres_filter_enabled` | `true` | 段级低能量帧比例过滤开关 |
| `vad_segment_decibel_thres` | `-40.0` | 段级 dBFS 低能量帧阈值 |
| `vad_decibel_frame_sample_ratio` | `0.5` | 段内 decibel 抽帧比例，范围 `(0,1]` |
| `vad_decibel_low_frame_ratio_thres` | `0.8` | 抽样帧中低于段级阈值的比例达到该值时删除段 |
| `vad_snr_thres_filter_enabled` | `true` | 段级 SNR 过滤开关 |
| `vad_low_volume_rms_dbfs_thres` | `-40.0` | 段级 RMS dBFS 门限，低于该值时删除段 |
| `vad_low_volume_filter_enabled` | `true` | 段级 RMS 过滤开关 |
| `after_start` / `before_end` | `3` / `3` | 段级过滤作用于完整音频前后 N 分钟；`0` 表示整段，`-1` 表示该侧不限制 |

音频与段级并发参数的区别：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `audio_concurrency` | 同时处于处理中的 A+B **音频文件**总数 | 8 |
| `audio_spk_emo_concurrency` | 总音频中请求增强能力的 B 类音频数量 | 4 |
| `segment_concurrency` | **单个音频内**同时执行本地转写的 VAD 段任务数 | 1 |
| `request_concurrency` | **所有音频合计**本地转写任务数上限 | 1 |

上述配置均在重启后生效，每个服务实例独立限流。旧配置缺少新增的
`audio_spk_emo_concurrency` 时同样采用 **4**，升级后增强音频并发会受此上限限制。
B 上限允许大于总上限，实际 B 并发至多为 `min(audio_concurrency, audio_spk_emo_concurrency)`；
配置值不会被静默截断。0 不能用于关闭增强能力，应使用 `[features]` 开关。

三个参数 `showSpk`、`showEmotion`、`showRoleIdentify` 全为 false 的音频称为 A；
任一个为 true 称为 B。每个 B 同时占用一个总音频名额和一个增强名额，即使开启多项也只计一个 B。
不为 B 预留总名额，不改变 Qwen 段级信号量或模型设备数组决定的推理容量。

总上限 8、B 上限 4、默认不等待时：

| 正在处理 | 新 A | 新 B |
|---|---|---|
| 3A + 4B | 正常处理 | `4017` |
| 5A + 3B | `4013` | `4013` |
| 8A | `4013` | `4013` |
| 4A + 4B | `4013` | `4013` |

`queue_wait_timeout_s > 0` 时，暂时满额的请求进入 A+B 共享等待队列，同类按到达顺序处理。
等待 B 不预占任何处理名额；B 因增强容量满而受阻时，A 可绕过它使用空闲总名额。
当较早 B 已满足两类容量条件时，它优先于后到 A。共享等待队列满返回 `4013`，
等待截止仍未取得完整名额返回 `4010`；`queue_max_size=0` 只允许立即获准的请求。

处理名额从准入成功持有至音频准备、VAD、转写、增强推理、后处理及临时文件清理结束。
成功、失败和取消都会回收；取消会等待已经启动且无法立即中断的线程或 VAD 工作必要收尾，
再清理文件和归还名额，因此取消完成可能晚于取消信号发出。

`GET /get_status` 用同一次快照返回音频准入状态，例如正在处理 `5A+3B`、等待 `1A+2B`：

```json
{
  "audioConcurrency": {"configured_total": 8, "active_total": 8, "waiting_total": 3},
  "spkEmoConcurrency": {"configured": 4, "active": 3, "waiting": 2}
}
```

A 的活动数和等待数分别用总数减去 B 数。这里的 B 等待按完整音频计数；
`localModels` 中的 `waiting` 则按等待 CAM++ 或 emotion2vec 的分段计数。
首次请求前活动数和等待数均为 0，查询不会加载模型，配置关闭而被拒绝的请求不增加计数。

新增 `[V3准入]` 日志记录 `request_id`、`category`（A/B）、`outcome`、`reason`、
`code` 和两组容量 `snapshot`。既有 `[V3 耗时]` 阶段日志保留，音频准入等待通过
遥测 `phases.audio_wait_ms` 记录，与模型池分段等待的 `spk_wait_ms` / `emo_wait_ms` 分开。
`admission.outcome` 为 `admitted`、`rejected`、`timeout` 或 `cancelled`；
未完成准入判定时为 `not_attempted`。准入后处理失败或取消仍为 `admitted`，
最终结果由请求 `status` 表达。快照在授予或拒绝时取得，等待取消时为回收后的状态。

`resourceDiagnostics.requestTelemetry` 增加 `admission.by_class`（A/B 请求数）和
`admission.rejections_by_reason`（按原因计数），并以 `window=retained_events` 标明统计窗口。
原因包括 `total_full`、`spk_emo_full`、`waiting_full`、`wait_timeout`、
`role_disabled`、`emotion_disabled`、`both_disabled`、`cancelled`、`qwen_disabled`。
拒绝汇总包含准入前的超时与取消；准入后的失败不计作拒绝。
遥测最多保留 4096 个请求事件，包含近期明细的诊断快照最多输出 20 条，
状态接口只输出汇总；这些是保留窗口统计，并非进程启动以来的累计值。
新增日志与遥测不保存音频、转写正文或模型对象。

### `[model_device_pool]` 配置项

控制 v3 链路本地模型副本和并发容量。数组长度是唯一槽位来源，不需要 `local_gpu_concurrency`、`cam_slots` 或 `emo_slots`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `speech_campplus_devices` | [`"npu:7", "npu:7", "npu:7"`] | 每个元素加载一个 CAM++ 实例；长度决定 CAM++ 最大并发 |
| `emotion2vec_devices` | [`"npu:7"`] | 每个元素加载一个 emotion2vec 实例；长度决定 emotion2vec 最大并发 |

例如 CAM++ 3 个实例、emotion2vec 1 个实例：

```toml
[model_device_pool]
speech_campplus_devices = ["npu:7", "npu:7", "npu:7"]
emotion2vec_devices = ["npu:7"]
```

两个模型池独立排队、互不借用容量；空数组表示禁用对应能力，请求会得到 CAM++ `4015` 或 emotion2vec `4016`。部分副本加载失败时，只有成功加载的实例计入容量，失败原因可从 `/get_status` 的 `localModels` 查询。该接口同时提供每个池的 `configured_instances`、`instances`、`available_instances`、`busy`、`waiting`、`failed_instances` 和等待摘要。

同一张 NPU 上增加数组元素会同时增加模型 HBM 和推理争用，可能降低单次推理速度或触发 OOM；应结合状态接口和 `npu-smi info` 的峰值 HBM、利用率及压测 P95 选择实例组合。VAD 设备由 `[vad].vad_device` 独立控制，Qwen 本地实例也在主服务进程内，共享目标 NPU。

FSMN-VAD 的设备由 `[vad].vad_device` 数组独立控制，例如 `['cpu', 'npu:7']` 会启动两个固定 worker。旧 `[qwen_asr]` VAD 字段和旧动态池字段不再参与实例数计算，迁移时会记录告警。

### 细小声音阈值回归

参考音频的人工标注文件不是严格 JSON，使用以下可选入口解析并比较 `0.3`、`0.6`、`0.8` 三档：

```bash
python scripts/vad_reference.py --output /tmp/vad-reference.json
```

报告包含 46 个“细小声音”标注段的抑制率、未标注段保留率、VAD 段数、Qwen 请求数、耗时以及 RMS/峰值诊断。区间命中定义为 VAD 输出与标注区间重叠达到标注区间的 50%；响度只用于诊断，不作为唯一判据。该长测不会进入默认快速测试，也不修改原始音频和标注文件。

`vad_decibel_thres` 使用 FunASR 内部 log 能量单位，不能直接填入 RMS dBFS；`vad_low_volume_rms_dbfs_thres` 才是面向采集音量的段级门限。历史试验最初默认关闭；当前宿主与容器模板启用三项段级过滤、窗口为开头/结尾各3分钟。新推荐值必须来自参考音频扫描并同时满足细小声音抑制率、标注区间无有效识别内容和正常语音保留率。扫描可用 `--target-suppression-rate`、`--target-retention-rate`、`--coarse-step`、`--fine-step` 和 `--output` 覆盖目标与输出路径。回滚时将开关改回 `false` 并重启，原有 `vad_speech_noise_thres` 语义不变。现有 `min_segment_duration_s` 仍是通用兼容过滤，但不参与本次低音量第一阶段标定。

### role / emotion 的加载条件

v3 链路的真实 `role` 与 `emotion` 由本机模型给出，加载与计算条件是两回事：

- **加载**：CAM++ 与 emotion2vec **各自独立**，分别由 `[features].open_v2_role` 与
  `[features].open_v2_emotion` 控制（默认均为 `true`）。关闭任一个，另一个仍可正常加载。
  emotion2vec 使用禁用 HF32 的原生 FP32 正确性基线；低精度/编译候选独立测量后再决定是否采用。
- **计算**：`showSpk` 或 `showRoleIdentify` 请求说话人能力，`showEmotion` 请求情绪能力，
  三个参数默认均为 `false`。未请求的能力对应字段为 `null`，不触发其模型加载。
- **配置拒绝**：请求要求的能力被 `[features]` 关闭时，在音频准入前返回 CAM++ `4015` 或
  emotion2vec `4016`；同时请求两项且两项都关闭返回 `4018`。设备数组非空也不能覆盖功能开关。
  请求整体失败，不开始转写，不以 `null` 字段成功返回；仅请求已开启的另一项能力可正常处理。
  两个功能开关均关闭且三个请求参数均为 `false` 时，仍可普通转写。
- **并行**：role / emotion 与 Qwen 转写通过 `asyncio.gather` 并行执行，不是「转写完成后再算」。
  已开启且有健康实例的模型在某个分段推理失败时，该段对应字段回退为 `null`，转写照常返回；
  模型池为空或全部加载失败则返回对应的 `4015` / `4016`，不使用组合关闭码。

### 进程与 HBM 定位

VAD 每数组元素对应独立 spawn worker；CAM++、emotion2vec、Qwen 由主服务内的独立模型池实例承载。`localModels` 与 `vadPool` 原键名保留，`qwenModel` 为同级新增字段，状态查询不触发加载。
`allocated_mb`/`memory_allocated_mb` 是 PyTorch 分配估计，reserved 是缓存分配器；`npu-smi info` 提供进程 HBM 与整卡 HBM，不可把共享卡全部占用归因本项目。未知值为 null 或 available=false，不能以零冒充无占用。

```bash
npu-smi info
curl http://127.0.0.1:8083/get_status
```

`instance_count` 默认 1；增加 uvicorn 进程或 `local_instance_count` 会复制本地 Qwen 权重，必须重新标定资源与并发。
`ASCEND_VISIBLE_DEVICES` 指定宿主物理卡，容器内设备编号从 0 重新映射；单独映射物理 7 的模板使用 `npu:0`。
旧 `base_url`/`connect_timeout_s` 仅兼容读取，不发 HTTP；四个旧 TensorRT 键被忽略并提示迁移，`cuda:N` 配置直接拒绝。

`bg`/`ed` 继续为两位小数字符串，`load_audio_time_ms`/`gpu_time_ms` 保持字符串。`gpu_time_ms` 保持原主链路墙钟计时范围，不表示纯 NPU kernel 时间。

## Docker

昇腾 Dockerfile、Compose、独立 conda-pack 环境和设备/IPC 映射见 [docker/README.md](docker/README.md)。
本轮验证交付文件和构建闭包，实际镜像构建与上线在服务稳定后进行。候选基础镜像存在不等于已通过容器验收。

性能、精度、样本与执行证据见 [NPU 迁移报告](docs/npu-910b-migration-validation-20261117.md)。原 NVIDIA 报告仅作历史标定来源。

## License

MIT License
