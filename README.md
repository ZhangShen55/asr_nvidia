# SeaCraftASR

基于 FastAPI 的离线语音转写服务，面向教育课堂录音。服务本地用 FunASR FSMN-VAD 做语音切分，
逐段并发调用独立的 **Qwen3-ASR** 服务完成转写，并在本机用 CAM++ 与 emotion2vec 给出说话人角色与情绪。

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
├── config.docker.toml          # 容器默认配置
├── Dockerfile
├── start.sh                    # Nginx + 多 uvicorn 实例
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
│   ├── qwen_asr_client.py      # Qwen3-ASR 异步客户端（模型发现 / 转写 / 重试）
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
| emotion2vec_plus_large | 情绪识别（fp16） |

模型权重被 `.gitignore` 忽略，但会进入 Docker 构建上下文并复制到镜像 `/app/model/`。

## 本地运行

本地推荐使用已安装依赖的 Conda 环境 `asr`：

```bash
conda activate asr
python main.py
```

服务监听 `0.0.0.0:8083`。默认读取当前目录的 `config.toml`，也可以显式指定配置：

```bash
CONFIG_PATH=./config.toml python main.py
```

本地配置中的模型路径为 `./model/<目录名>`，因此需要从仓库根目录启动。

### 前台启动（8083）

下面的命令会占用当前终端，适合调试和查看实时日志；按 `Ctrl+C` 停止服务：

```bash
cd /home/zhangs/workspace/asr_nvidia
CONFIG_PATH=/home/zhangs/workspace/asr_nvidia/config.toml \
PYTHONPATH=/home/zhangs/workspace/asr_nvidia \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/anaconda3/envs/asr/bin/python -m uvicorn main:app \
  --host 0.0.0.0 --port 8083 --workers 1
```

### 后台启动（8083）

后台启动后，日志写入 `logs/asr-nvidia-8083.log`，PID 写入
`logs/asr-nvidia-8083.pid`：

```bash
cd /home/zhangs/workspace/asr_nvidia
mkdir -p logs
nohup env \
  CONFIG_PATH=/home/zhangs/workspace/asr_nvidia/config.toml \
  PYTHONPATH=/home/zhangs/workspace/asr_nvidia \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/anaconda3/envs/asr/bin/python -m uvicorn main:app \
  --host 0.0.0.0 --port 8083 --workers 1 \
  > logs/asr-nvidia-8083.log 2>&1 &
echo $! > logs/asr-nvidia-8083.pid
```

查看后台日志和服务状态：

```bash
tail -f /home/zhangs/workspace/asr_nvidia/logs/asr-nvidia-8083.log
curl http://127.0.0.1:8083/get_status
```

停止后台服务：

```bash
kill -TERM "$(cat /home/zhangs/workspace/asr_nvidia/logs/asr-nvidia-8083.pid)"
```

### 使用 Conda `asr` 环境启动指定端口

下面的命令从仓库根目录启动一个 uvicorn 后端实例，监听 `8085` 端口：

```bash
cd /home/zhangs/workspace/asr_nvidia
conda run --no-capture-output -n asr \
  env CONFIG_PATH=./config.toml PYTHONPATH=. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uvicorn main:app --host 0.0.0.0 --port 8085 --workers 1
```

服务启动后，v3 接口地址为 `http://127.0.0.1:8085/v3/seacraft_asr`；按 `Ctrl+C` 停止该后端实例。`OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 只限制底层算子线程，不限制 VAD worker 进程数；VAD 有效完整音频并发等于 `len(qwen_asr.vad_device)`。Qwen-ASR 服务地址由 `[qwen_asr].base_url` 配置，默认是 `http://127.0.0.1:8806`。

### 模型内存压力安全诊断

诊断工具跳过已确认正常的 VAD-only A 项，按 CAM++ 单独加载、emotion2vec 单独加载、两者同时加载和 2/4/8 路完整音频逐级采样。它每秒记录 RSS/PSS、Private 内存、`VmSwap`、`MemAvailable`、Swap I/O、GPU 显存/利用率、VAD worker、缓存目录和 8085/8806 健康状态，并在红色安全条件触发时停止后续请求。工具不会发送停止或重启 8806 Qwen 的命令。

```bash
cd /home/zhangs/workspace/asr_nvidia
conda run --no-capture-output -n asr \
  python scripts/model_memory_pressure_diagnostics.py \
  --base-url http://127.0.0.1:8085 \
  --qwen-url http://127.0.0.1:8806 \
  --audio-dir /home/zhangs/workspace/asr_nvidia/test/audio_file/音频文件 \
  --out-dir /tmp/seacraft-model-memory-diagnostics \
  --cooldown-s 60
```

运行前先确认 `curl http://127.0.0.1:8085/get_status` 和
`curl http://127.0.0.1:8806/v1/models` 均正常。`--plan-only` 只生成包含 Git 提交号、配置快照、设备数组和阶段清单的 `metadata.json`，不会连接服务。输出目录中的 `snapshots.jsonl`、`stage-*.json` 和 `run-summary.json` 可直接复算报告；红色终止条件包括可用内存低于 `max(8 GiB, 总内存 20%)`、Swap 增量超过 256 MiB、连续 Swap I/O、GPU 显存使用率达到 90%、后端/Qwen 健康检查失败或 VAD worker 数量下降。

`/get_status` 的 `resourceDiagnostics` 是附加字段，采集失败时通过 `available=false` 和 `reason` 表示，不把缺失值当作零；其中 `requestTelemetry` 只保留有界的阶段耗时统计，不保存音频和转写正文。

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

> `showSpk`、`showEmotion`、`showRoleIdentify` 的默认值均为 `false`（12 服务器口径）。

响应包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms` 与 `language_ratio`。
分段中的 `role`、`emotion` 和 `segment_words` 根据请求参数返回。`text` 为最终保留分段按空格拼接并经文本规范化（ITN）处理后的结果。纯「啊 / 哦 / 嗯 / 呃」分段会被移除；英文为有效文本占比最高时，边界明确的中文口气词会转换为 `ah / oh / um / uh`。

### v3 处理流程

1. 服务本地用 FunASR FSMN-VAD 把音频切成若干语音段；
2. 每个语音段并发发往 Qwen3-ASR 服务（单段一个请求，**不做相邻段合并**）；
3. 响应分段的 `bg` / `ed` 直接取该 VAD 段的起止时间；
4. 复读修复、单字语气词移除、语言占比判定、英文口气词替换、ITN 和标点清理依次执行，最终统计只使用最终保留分段。

### 语种占比 `language_ratio`

`language_ratio` 的键为语种**简写**（`zh` / `en` / `yue` / …），值为该语种文本字符占比（百分比，保留一位小数，合计 100.0）。
模型返回的语种名不在映射表内时原样透传。顶层 `language` 字段仍返回模型返回的语种全称（如 `Chinese` / `English`）。

### 文本规范化与标点卫生

ITN（WeTextProcessing）默认开启，负责数字规范化与语气词剔除。ITN 之后还有两条与 ITN 开关**无关**、
始终执行的标点卫生规则：

- 整段文本只有标点符号时，该段从 `segments` 中移除；
- 分段文本以标点开头时，移除开头**连续的所有标点**，直到落到有效字符。

### 错误码

| 错误码 | 含义 |
|--------|------|
| `4008` | 音频为空或未检测到任何人声 |
| `4009` | 提交的语种不被 Qwen 支持 |
| `4010` | 请求排队等待超时（`queue_wait_timeout_s > 0` 时可能触发） |
| `4011` | Qwen 服务不可用或未发现可用模型 |
| `4012` | Qwen 服务调用失败（超时、连接错误、重试耗尽） |
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
| `[qwen_asr]` | v3 链路的 Qwen 服务地址、并发度、队列与 VAD 参数 |
| `[model_device_pool]` | CAM++ 与 emotion2vec 的设备数组；每个元素对应一个独立模型实例和一个并发槽位 |

`speed_info` 按 1、5、10 分钟窗口统计，跨窗口句子按时间重叠比例分摊内容数量。单句语速使用 `[speech_rate].rate_factor` 修正，分时段语速不乘该系数。

### `[qwen_asr]` 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 是否开放 `/v3/seacraft_asr` |
| `base_url` | `http://127.0.0.1:8806` | Qwen3-ASR 服务地址（容器部署用 `http://192.168.29.12:8806`） |
| `model` | `""` | 模型名；留空表示启动时从 `/v1/models` 自动发现（不需要 API Key） |
| `segment_concurrency` | `16` | 单个音频内同时发往 Qwen 的 VAD 段请求数 |
| `request_concurrency` | `128` | 所有音频合计发往 Qwen 的请求数上限 |
| `audio_concurrency` | `8` | A+B 同时处理的音频总数 |
| `audio_spk_emo_concurrency` | `4` | 请求任一增强能力的 B 类整音频上限，必须为正整数 |
| `queue_max_size` | `64` | A+B 共享等待队列容量 |
| `queue_wait_timeout_s` | `0` | `0` = 队列满立即拒绝；`>0` 才排队等待，超时返回 `4010` |
| `request_timeout_s` | `120` | 单次 Qwen 请求超时（秒） |
| `connect_timeout_s` | `10` | Qwen 服务连接超时（秒） |
| `retry_count` | `1` | Qwen 请求失败重试次数 |
| `min_segment_duration_s` | `0.3` | 过滤过短 VAD 段 |
| `vad_device` | `8 个设备元素` | 每个元素启动一个固定 FSMN-VAD worker；数组长度就是完整音频 VAD 并发 |
| `speaker_similarity_threshold` | `0.30` | CAM++ 说话人聚类的余弦相似度阈值（按实测 sweep 取值，阈值过高会严重过分割） |
| `vad_max_single_segment_time_ms` | `30000` | VAD 单段最长时长 |
| `vad_max_end_silence_time_ms` | `800` | VAD 静音切分阈值 |
| `vad_speech_noise_thres` | `0.6` | VAD 语音/噪声判定阈值：`0.3` 偏灵敏，`0.6` 正常，`0.8` 偏不灵敏；修改后重启生效 |
| `vad_decibel_thres` | `-100.0` | FunASR 帧级 log 能量门限，单位不是 dBFS；`-100` 保持原行为 |
| `vad_snr_thres` | `-100.0` | FunASR 帧级 SNR 门限；`-100` 保持原行为 |
| `vad_low_volume_rms_dbfs_thres` | `-100.0` | 段级 RMS dBFS 门限；低于该值可判定为低音量，范围 `-100~0` |
| `vad_low_volume_filter_enabled` | `false` | 是否在创建 clip/Qwen 请求前启用低音量过滤；修改后重启生效 |

音频与段级并发参数的区别：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `audio_concurrency` | 同时处于处理中的 A+B **音频文件**总数 | 8 |
| `audio_spk_emo_concurrency` | 总音频中请求增强能力的 B 类音频数量 | 4 |
| `segment_concurrency` | **单个音频内**同时发往 Qwen 的 VAD 段请求数 | 16 |
| `request_concurrency` | **所有音频合计**发往 Qwen 的请求数上限 | 128 |

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
| `speech_campplus_devices` | [`"cuda:0"`] | 每个元素加载一个 CAM++ 实例；长度决定 CAM++ 最大并发 |
| `emotion2vec_devices` | [`"cuda:0"`] | 每个元素加载一个 emotion2vec 实例；长度决定 emotion2vec 最大并发 |

例如 CAM++ 3 个实例、emotion2vec 1 个实例：

```toml
[model_device_pool]
speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0"]
```

两个模型池独立排队、互不借用容量；空数组表示禁用对应能力，请求会得到 CAM++ `4015` 或 emotion2vec `4016`。部分副本加载失败时，只有成功加载的实例计入容量，失败原因可从 `/get_status` 的 `localModels` 查询。该接口同时提供每个池的 `configured_instances`、`instances`、`available_instances`、`busy`、`waiting`、`failed_instances` 和等待摘要。

同一张 GPU 上增加数组元素会同时增加模型显存和推理争用，可能降低单次推理速度或触发 OOM；应结合状态接口和 `nvidia-smi` 的峰值显存、利用率及压测 P95 选择实例组合。VAD 设备仍由 `[qwen_asr].vad_device` 独立控制，Qwen 服务本身在另一个容器内。

FSMN-VAD 的设备由 `[qwen_asr].vad_device` 数组独立控制，例如 `['cpu', 'cuda:0']` 会启动两个固定 worker。旧的动态池字段和旧模型设备结构不再参与实例数计算，迁移时会记录告警。

### 细小声音阈值回归

参考音频的人工标注文件不是严格 JSON，使用以下可选入口解析并比较 `0.3`、`0.6`、`0.8` 三档：

```bash
python scripts/vad_reference.py --output /tmp/vad-reference.json
```

报告包含 46 个“细小声音”标注段的抑制率、未标注段保留率、VAD 段数、Qwen 请求数、耗时以及 RMS/峰值诊断。区间命中定义为 VAD 输出与标注区间重叠达到标注区间的 50%；响度只用于诊断，不作为唯一判据。该长测不会进入默认快速测试，也不修改原始音频和标注文件。

`vad_decibel_thres` 使用 FunASR 内部 log 能量单位，不能直接填入 RMS dBFS；`vad_low_volume_rms_dbfs_thres` 才是面向采集音量的段级门限。新过滤默认关闭，推荐值必须来自参考音频扫描并同时满足细小声音抑制率、标注区间无有效识别内容和正常语音保留率。扫描可用 `--target-suppression-rate`、`--target-retention-rate`、`--coarse-step`、`--fine-step` 和 `--output` 覆盖目标与输出路径。回滚时将开关改回 `false` 并重启，原有 `vad_speech_noise_thres` 语义不变。现有 `min_segment_duration_s` 仍是通用兼容过滤，但不参与本次低音量第一阶段标定。

### role / emotion 的加载条件

v3 链路的真实 `role` 与 `emotion` 由本机模型给出，加载与计算条件是两回事：

- **加载**：CAM++ 与 emotion2vec **各自独立**，分别由 `[features].open_v2_role` 与
  `[features].open_v2_emotion` 控制（默认均为 `true`）。关闭任一个，另一个仍可正常加载。
  emotion2vec 以**半精度（fp16）**加载，权重约占 1 GB。
- **计算**：`showSpk` 或 `showRoleIdentify` 请求说话人能力，`showEmotion` 请求情绪能力，
  三个参数默认均为 `false`。未请求的能力对应字段为 `null`，不触发其模型加载。
- **配置拒绝**：请求要求的能力被 `[features]` 关闭时，在音频准入前返回 CAM++ `4015` 或
  emotion2vec `4016`；同时请求两项且两项都关闭返回 `4018`。设备数组非空也不能覆盖功能开关。
  请求整体失败，不开始转写，不以 `null` 字段成功返回；仅请求已开启的另一项能力可正常处理。
  两个功能开关均关闭且三个请求参数均为 `false` 时，仍可普通转写。
- **并行**：role / emotion 与 Qwen 转写通过 `asyncio.gather` 并行执行，不是「转写完成后再算」。
  已开启且有健康实例的模型在某个分段推理失败时，该段对应字段回退为 `null`，转写照常返回；
  模型池为空或全部加载失败则返回对应的 `4015` / `4016`，不使用组合关闭码。

### 进程名与显存定位

三个本地模型由**同一个服务进程**承载（拆成三个进程会多出约 0.8 GB 的 CUDA context 开销）。
服务启动时把进程名覆写为 `asr-offline-model`，因此在 `nvidia-smi` 的进程列表里可以直接定位本服务：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

由于同进程承载，`nvidia-smi` 只能看到聚合显存。**逐模型的显存明细**通过状态接口获取：

```bash
curl http://localhost:8083/get_status | jq .localModels
```

```json
{
  "vad": { "loaded": true, "device": "cuda:0", "allocated_mb": 1.6, "instances": 4 },
  "speech_campplus": { "loaded": true, "device": "cuda:0", "allocated_mb": 26.7, "instances": 1 },
  "emotion2vec": { "loaded": true, "device": "cuda:0", "allocated_mb": 313.3, "instances": 1 }
}
```

`instance_count` 默认 `1`；配置为 N 时容器内并行 N 个 uvicorn 实例，**每个实例各自持有一份模型权重**，
本地模型显存占用为单实例的 N 倍。

### Qwen 服务部署

Qwen3-ASR 由独立容器提供，**不进本镜像**。本服务通过「IP + 8806」访问，不需要 API Key：

```bash
# 容器侧示例
docker run -d --name qwen3-asr-0.6b-8806 -p 8806:8000 qwenllm/qwen3-asr:latest
```

然后把 `base_url` 指向该地址（`config.docker.toml` 默认写 `http://192.168.29.12:8806`），重启服务即可。
本阶段不要求两个容器在同一 Docker 网络。

Qwen 服务侧的吞吐扩容方式（`--api-server-count` 等参数）与实测收益见
`docs/qwen-asr-v2-throughput-optimization.md`；本服务侧的并发默认值维持不变。

## Docker

模型已经随镜像交付，默认运行不需要挂载模型目录。

```bash
docker build -t seacraft-asr:offline .
```

Dockerfile 会检查三个模型目录（FSMN-VAD、CAM++、emotion2vec）是否存在且非空。
依赖安装保留 Docker 构建缓存，不使用 `--no-cache`。

默认启动：

```bash
docker run -d \
  --name seacraft-asr \
  --gpus '"device=0"' \
  -p 9000:9000 \
  seacraft-asr:offline
```

对外入口为 Nginx `9000` 端口，容器内默认读取 `/config.toml`，设备为 `cuda:0`，单实例运行。

挂载配置仅用于调整并发、设备、语速等参数时，模型路径应继续保留 `/app/model/`：

```bash
docker run -d \
  --name seacraft-asr \
  --gpus '"device=0"' \
  -p 9000:9000 \
  -v "$(pwd)/config.docker.toml:/config.toml:ro" \
  -e CONFIG_PATH=/config.toml \
  seacraft-asr:offline
```

## 验证

```bash
curl http://localhost:9000/get_status
curl http://localhost:9000/openapi.json
```

## License

MIT License
## CAM++ / emotion2vec 可选 TensorRT

四项后端配置位于 `[model_device_pool]`：两个 `*_use_tensorrt` 默认 false，两个
`*_tensorrt_precision` 默认 `"fp16"`，支持 fp32/fp16。设备数组每个元素代表一个实例；
CUDA 按开关选择后端，NPU 固定 torch_npu，CPU 使用 PyTorch。完整配置、可选依赖、资产校验、
状态与回滚见 [TensorRT 说明](docs/TENSORRT.md)。修改后重启生效。
