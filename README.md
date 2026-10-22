# SeaCraftASR

基于 FastAPI 的离线语音转写服务，面向教育课堂录音，提供 Paraformer 中文转写、VAD、标点恢复、说话人分离、教师/学生角色识别、情绪识别和语速统计。`/v1.1.8/seacraft_asr` 的接口契约与 Mac 侧 `asr_offline` 保持一致：`auto`/`zh`/`en` 走 Paraformer，`fr` 走 Whisper，其余语言返回 `4009`。

服务同时提供 `/v2/seacraft_asr`：本地做 VAD 切分，逐段并发调用独立的 **Qwen3-ASR** 服务，响应结构与 `/v1.1.8` 同构，并额外返回语种占比字典 `language_ratio`。两条链路互不影响，可同时在线。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1.1.8/seacraft_asr` | 离线 ASR 转写 |
| `POST` | `/v2/seacraft_asr` | 离线 ASR 转写（Qwen3-ASR 链路，响应与 v1.1.8 同构） |
| `GET` | `/get_status` | 服务状态与任务统计 |
| `DELETE` | `/clear_tasks_list` | 清空当前实例的处理中任务列表 |

FastAPI 自动文档位于 `/docs`、`/redoc` 和 `/openapi.json`。

## 项目结构

```text
.
├── main.py
├── config.toml                 # 本地运行配置
├── config.docker.toml          # 容器默认配置
├── Dockerfile
├── start.sh                    # Nginx + 多 uvicorn 实例
├── api/routes/
│   ├── asr_v18.py              # /v1.1.8/seacraft_asr
│   ├── asr_v2.py               # /v2/seacraft_asr
│   ├── asr_qwen_common.py      # v2 主链路：VAD → 并发转写 → 响应封装
│   └── status.py
├── core/
│   ├── config.py
│   ├── concurrency.py
│   ├── logging.py
│   ├── models.py
│   ├── vad.py                  # v2 链路的本地 VAD 切分
│   ├── qwen_asr_client.py      # Qwen3-ASR 异步客户端（模型发现 / 转写 / 重试）
│   └── qwen_asr_queue.py       # v2 链路的音频排队与两级请求并发控制
├── entity/data.py
├── model/                      # 模型权重，不进入 Git
└── utils/
    ├── audio_utils.py
    ├── asr_stats.py
    ├── feature_utils.py
    ├── itn.py                  # 文本规范化（WeTextProcessing）
    └── qwen_response.py        # Qwen 返回文本解析与语种占比统计
```

## 模型目录

模型需要完整放入 `model/` 下的五个目录，启用法语能力时另需放入 Whisper 模型：

```text
model/
├── README.md
├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── punc_ct-transformer_cn-en-common-vocab471067-large/
├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
├── speech_campplus_sv_zh_en_16k-common_advanced/
├── emotion2vec_plus_large/
└── faster-whisper-large-v3/   # 可选，启用法语时必需
```

| 模型 | 用途 |
|------|------|
| Paraformer | 中文语音识别 |
| VAD | 语音端点检测 |
| PUNC | 中英文标点恢复 |
| CAM++ | 说话人特征与说话人分离 |
| emotion2vec | 情绪识别 |
| faster-whisper-large-v3 | 法语等小语种转写（可选） |

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
curl -X POST "http://localhost:8083/v1.1.8/seacraft_asr" \
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
| `language` | `auto` | 请求中回传的语言标记 |
| `wordTimestamps` | `false` | 是否返回词级时间戳 |
| `hotWords` | 空 | 热词；`ban_hotword=true` 时不生效 |
| `showSpk` | `false` | 是否启用说话人分离 |
| `showEmotion` | `false` | 是否返回情绪标签 |
| `showRoleIdentify` | `false` | 是否识别教师/学生角色 |

> Mac 侧 `showSpk`、`showEmotion` 默认 `true`，12 服务器默认 `false`；其余参数默认值与行为一致。

响应包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms` 和 `gpu_time_ms`。分段中的 `role`、`emotion` 和 `segment_words` 根据请求参数返回。`text` 为分段文本按空格拼接并经文本规范化（ITN）处理后的结果。

语言分发规则：

| `language` | 引擎 | 说明 |
|------------|------|------|
| `auto`/`zh`/`en` | Paraformer | 大小写与空格自动归一化 |
| `fr` | Whisper | 需开启 `open_mul_lang` 并提供 Whisper 模型，否则返回 `4003` |
| 其他或显式空值 | — | 返回 `{"msg": "不支持的语言: <lang>", "code": 4009}` |

## v2 接口（Qwen3-ASR 链路）

`/v2/seacraft_asr` 的请求参数与 `/v1.1.8` 完全一致（multipart 表单），处理流程为：

1. 服务本地用 FunASR FSMN-VAD 把音频切成若干语音段；
2. 每个语音段并发发往 Qwen3-ASR 服务（单段一个请求，**不做相邻段合并**）；
3. 响应分段的 `bg` / `ed` 直接取该 VAD 段的起止时间。

```bash
curl -X POST "http://localhost:8083/v2/seacraft_asr" \
  -F "audioFile=@sample.wav" \
  -F "showSpk=true" \
  -F "showEmotion=true"
```

响应与 `/v1.1.8` 同构：`language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms`，并额外增加：

| 字段 | 说明 |
|------|------|
| `language_ratio` | 字典，键为模型识别出的语种名，值为该语种文本字符占比（百分比，保留一位小数，合计 100.0）；无有效文本时为 `{}` |

分段字段为 `segment_text`、`bg`、`ed`、`speed`、`segment_words`、`role`、`emotion`。`segment_words` 固定为空数组（Qwen 服务不返回时间戳）；`role` 与 `emotion` 在开启对应能力时返回真实值，否则为 `null`。

`language` 参数的透传规则：

| 请求参数 | 行为 |
|----------|------|
| 未提交 / 空串 / `auto` | **不向 Qwen 下发** `language` 字段，由模型自动识别（Qwen 服务端不接受 `auto`，会直接 400） |
| 具体语种（如 `zh`、`en`） | 原样透传给 Qwen；服务端不支持时返回 `{"msg": "不支持的语言", "code": 4009}` |

响应中的 `language` 取模型实际识别出的语种（如 `Chinese` / `English`），不是请求参数的回显。

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
| `device` | 推理设备 |
| `concurrency` | 单实例 GPU 并发限制 |
| `instance_count` | `start.sh` 启动的 uvicorn 实例数 |
| `[model_paths]` | 模型目录，含可选的 `whisper_model_dir` |
| `[speech_rate]` | 单句语速修正系数 |
| `[audio_chunk]` | 长音频分块与重试参数 |
| `[features]` | `open_spk`、`open_emotion`、`ban_hotword`、`open_mul_lang` |
| `[itn]` | 文本规范化开关与失败策略 |
| `[qwen_asr]` | v2 链路的 Qwen 服务地址、并发度、队列与 VAD 参数 |
| `[model_device]` | v2 链路三个本地模型的落卡索引 |

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

三层并发参数的区别：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `audio_concurrency` | 同时处于处理中的**音频文件**数量 | 8 |
| `segment_concurrency` | **单个音频内**同时发往 Qwen 的 VAD 段请求数 | 4 |
| `request_concurrency` | **所有音频合计**发往 Qwen 的请求数上限 | 32 |

### `[model_device]` 配置项

控制 v2 链路三个本地模型的落卡，**三者绑定同一张卡**：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `vad` | `0` | FunASR FSMN-VAD |
| `emotion2vec` | `0` | emotion2vec_plus_large |
| `speech_campplus` | `0` | speech_campplus_sv_zh_en_16k-common_advanced |

该段**只作用于 v2 链路**，`/v1.1.8` 仍按原有的 `device` / `ngpu` 配置运行。Qwen 服务本身在另一个容器内，不占用这三张卡。

### role / emotion 的加载条件

v2 链路的真实 `role` 与 `emotion` 由本机模型给出，加载与计算条件是两回事：

- **加载**：CAM++ 与 emotion2vec 仅在 `[features]` 中 `open_spk = true` **且** `open_emotion = true` **同时成立**时才加载；任一为 false 都不加载。emotion2vec 以**半精度（fp16）**加载，权重约占 1 GB。
- **计算**：由请求参数 `showSpk` / `showEmotion` 触发（默认均为 `false`，与 12 服务器上 v1.1.8 的默认值一致）。模型未加载而请求开启了对应开关时，对应字段回退为 `null`，不影响转写结果。
- **并行**：role / emotion 与 Qwen 转写通过 `asyncio.gather` 并行执行，不是「转写完成后再算」。任一路失败则该字段回退为 `null`，转写结果照常返回。

> v1.1.8 与 v2 各持一份 emotion2vec 实例（约 3.8 GB 合计），同一时刻建议只跑一条链路。

### Qwen 服务部署

Qwen3-ASR 由独立容器提供，**不进本镜像**。本服务通过「IP + 8806」访问，不需要 API Key：

```bash
# 容器侧示例
docker run -d --name qwen3-asr-0.6b-8806 -p 8806:8000 qwenllm/qwen3-asr:latest
```

然后把 `base_url` 指向该地址（`config.docker.toml` 默认写 `http://192.168.29.12:8806`），重启服务即可。本阶段不要求两个容器在同一 Docker 网络。

## Docker

模型已经随镜像交付，默认运行不需要挂载模型目录。

```bash
docker build -t seacraft-asr:offline .
```

Dockerfile 会检查五个模型目录是否存在且非空；当 `config.docker.toml` 中 `open_mul_lang = true` 时，额外要求 `faster-whisper-large-v3` 存在且非空。依赖安装保留 Docker 构建缓存，不使用 `--no-cache`。

默认启动：

```bash
docker run -d \
  --name seacraft-asr \
  --gpus '"device=1"' \
  -p 9000:9000 \
  seacraft-asr:offline
```

对外入口为 Nginx `9000` 端口，容器内默认读取 `/config.toml`，设备为 `cuda:0`，单实例运行。

挂载配置仅用于调整并发、设备、语速或分块参数时，模型路径应继续保留 `/app/model/`：

```bash
docker run -d \
  --name seacraft-asr \
  --gpus '"device=1"' \
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
