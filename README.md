# SeaCraftASR

基于 FastAPI 的离线语音转写服务，面向教育课堂录音。服务本地用 FunASR FSMN-VAD 做语音切分，
逐段并发调用独立的 **Qwen3-ASR** 服务完成转写，并在本机用 CAM++ 与 emotion2vec 给出说话人角色与情绪。

> `/v1.1.8/seacraft_asr`（SeACo-Paraformer + PUNC + Whisper 法语链路）已随本次变更下线，
> 服务只保留 `/v2/seacraft_asr`、`/get_status`、`/clear_tasks_list` 三个接口。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v2/seacraft_asr` | 离线 ASR 转写（Qwen3-ASR 链路） |
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
│   ├── asr_v2.py               # /v2/seacraft_asr
│   ├── asr_qwen_common.py      # v2 主链路：VAD → 并发转写 → 响应封装
│   ├── asr_common.py           # v2 共用：请求上下文、情绪标签映射、临时文件清理
│   └── status.py
├── core/
│   ├── config.py
│   ├── logging.py              # 日志按天 + 按大小双轮转
│   ├── models.py               # CAM++ / emotion2vec 懒加载 + 显存登记
│   ├── vad.py                  # v2 链路的本地 VAD 切分
│   ├── qwen_asr_client.py      # Qwen3-ASR 异步客户端（模型发现 / 转写 / 重试）
│   ├── qwen_asr_queue.py       # v2 链路的音频排队与两级请求并发控制
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

## ASR 请求

```bash
curl -X POST "http://localhost:8083/v2/seacraft_asr" \
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
| `wordTimestamps` | `false` | 是否返回词级时间戳（v2 链路固定返回空数组） |
| `showSpk` | `false` | 是否启用说话人分离 |
| `showEmotion` | `false` | 是否返回情绪标签 |
| `showRoleIdentify` | `false` | 是否识别教师/学生角色 |

> `showSpk`、`showEmotion`、`showRoleIdentify` 的默认值均为 `false`（12 服务器口径）。

响应包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms` 与 `language_ratio`。
分段中的 `role`、`emotion` 和 `segment_words` 根据请求参数返回。`text` 为最终保留分段按空格拼接并经文本规范化（ITN）处理后的结果。纯「啊 / 哦 / 嗯 / 呃」分段会被移除；英文为有效文本占比最高时，边界明确的中文口气词会转换为 `ah / oh / um / uh`。

### v2 处理流程

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
| `4013` | 处理队列已满（同时处理的音频数达到 `audio_concurrency`） |

> 默认 `queue_wait_timeout_s = 0`，即第 9 个并发音频**立即**返回 `4013`，不做静默排队。

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
| `[qwen_asr]` | v2 链路的 Qwen 服务地址、并发度、队列与 VAD 参数 |
| `[model_device]` | CAM++ 与 emotion2vec 的设备串，使用 `cuda:N`（默认 `cuda:0`）；VAD 设备独立配置 |

`speed_info` 按 1、5、10 分钟窗口统计，跨窗口句子按时间重叠比例分摊内容数量。单句语速使用 `[speech_rate].rate_factor` 修正，分时段语速不乘该系数。

### `[qwen_asr]` 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 是否开放 `/v2/seacraft_asr` |
| `base_url` | `http://127.0.0.1:8806` | Qwen3-ASR 服务地址（容器部署用 `http://192.168.29.12:8806`） |
| `model` | `""` | 模型名；留空表示启动时从 `/v1/models` 自动发现（不需要 API Key） |
| `segment_concurrency` | `4` | 单个音频内同时发往 Qwen 的 VAD 段请求数 |
| `request_concurrency` | `32` | 所有音频合计发往 Qwen 的请求数上限（= 4 × 8） |
| `audio_concurrency` | `8` | 同时处理的音频文件数 |
| `queue_max_size` | `64` | 等待队列容量 |
| `queue_wait_timeout_s` | `0` | `0` = 队列满立即拒绝；`>0` 才排队等待，超时返回 `4010` |
| `request_timeout_s` | `120` | 单次 Qwen 请求超时（秒） |
| `connect_timeout_s` | `10` | Qwen 服务连接超时（秒） |
| `retry_count` | `1` | Qwen 请求失败重试次数 |
| `min_segment_duration_s` | `0.3` | 过滤过短 VAD 段 |
| `vad_concurrency` | `4` | VAD 模型实例池大小（FSMN-VAD 实例不可并发复用） |
| `speaker_similarity_threshold` | `0.30` | CAM++ 说话人聚类的余弦相似度阈值（按实测 sweep 取值，阈值过高会严重过分割） |
| `local_gpu_concurrency` | `4` | 本机 GPU（CAM++ / emotion2vec）推理并发路数 |
| `vad_max_single_segment_time_ms` | `30000` | VAD 单段最长时长 |
| `vad_max_end_silence_time_ms` | `800` | VAD 静音切分阈值 |
| `vad_speech_noise_thres` | `0.6` | VAD 语音/噪声判定阈值：`0.3` 偏灵敏，`0.6` 正常，`0.8` 偏不灵敏；修改后重启生效 |

三层并发参数的区别：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `audio_concurrency` | 同时处于处理中的**音频文件**数量 | 8 |
| `segment_concurrency` | **单个音频内**同时发往 Qwen 的 VAD 段请求数 | 4 |
| `request_concurrency` | **所有音频合计**发往 Qwen 的请求数上限 | 32 |

### `[model_device]` 配置项

控制 v2 链路本地 GPU 模型的落卡，设备值使用 `cuda:N`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `emotion2vec` | `cuda:0` | emotion2vec_plus_large |
| `speech_campplus` | `cuda:0` | speech_campplus_sv_zh_en_16k-common_advanced |

Qwen 服务本身在另一个容器内，不占用这三张卡。

FSMN-VAD 的设备由 `[qwen_asr].vad_device` 独立控制，默认 `cpu`，也可设置为 `cuda:N`。旧部署中的整数模型设备索引仍会短期兼容读取并记录迁移告警。

### 细小声音阈值回归

参考音频的人工标注文件不是严格 JSON，使用以下可选入口解析并比较 `0.3`、`0.6`、`0.8` 三档：

```bash
python scripts/vad_reference.py --output /tmp/vad-reference.json
```

报告包含 46 个“细小声音”标注段的抑制率、未标注段保留率、VAD 段数、Qwen 请求数、耗时以及 RMS/峰值诊断。区间命中定义为 VAD 输出与标注区间重叠达到标注区间的 50%；响度只用于诊断，不作为唯一判据。该长测不会进入默认快速测试，也不修改原始音频和标注文件。

### role / emotion 的加载条件

v2 链路的真实 `role` 与 `emotion` 由本机模型给出，加载与计算条件是两回事：

- **加载**：CAM++ 与 emotion2vec **各自独立**，分别由 `[features].open_v2_role` 与
  `[features].open_v2_emotion` 控制（默认均为 `true`）。关闭任一个，另一个仍可正常加载。
  emotion2vec 以**半精度（fp16）**加载，权重约占 1 GB。
- **计算**：由请求参数 `showSpk` / `showEmotion` 触发（默认均为 `false`）。模型未加载而请求开启了
  对应开关时，对应字段回退为 `null`，不影响转写结果。
- **并行**：role / emotion 与 Qwen 转写通过 `asyncio.gather` 并行执行，不是「转写完成后再算」。
  任一路失败则该字段回退为 `null`，转写结果照常返回。

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
