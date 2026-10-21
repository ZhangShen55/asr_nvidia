# SeaCraftASR

基于 FastAPI 的离线语音转写服务，面向教育课堂录音，提供 Paraformer 中文转写、VAD、标点恢复、说话人分离、教师/学生角色识别、情绪识别和语速统计。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1.1.8/seacraft_asr` | 离线 ASR 转写 |
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
│   └── status.py
├── core/
│   ├── config.py
│   ├── concurrency.py
│   ├── logging.py
│   └── models.py
├── entity/data.py
├── model/                      # 模型权重，不进入 Git
└── utils/
    ├── audio_utils.py
    ├── asr_stats.py
    └── feature_utils.py
```

## 模型目录

模型需要完整放入 `model/` 下的五个目录：

```text
model/
├── README.md
├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── punc_ct-transformer_cn-en-common-vocab471067-large/
├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
├── speech_campplus_sv_zh_en_16k-common_advanced/
└── emotion2vec_plus_large/
```

| 模型 | 用途 |
|------|------|
| Paraformer | 中文语音识别 |
| VAD | 语音端点检测 |
| PUNC | 中英文标点恢复 |
| CAM++ | 说话人特征与说话人分离 |
| emotion2vec | 情绪识别 |

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
| `showRoleIdentify` | `true` | 是否识别教师/学生角色 |

响应包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms` 和 `gpu_time_ms`。分段中的 `role`、`emotion` 和 `segment_words` 根据请求参数返回。

## 配置

主要配置项：

| 配置 | 说明 |
|------|------|
| `device` | 推理设备 |
| `concurrency` | 单实例 GPU 并发限制 |
| `instance_count` | `start.sh` 启动的 uvicorn 实例数 |
| `[model_paths]` | 五个模型目录 |
| `[speech_rate]` | 单句语速修正系数 |
| `[audio_chunk]` | 长音频分块与重试参数 |
| `[features]` | `open_spk`、`open_emotion`、`ban_hotword` |

`speed_info` 按 1、5、10 分钟窗口统计，跨窗口句子按时间重叠比例分摊内容数量。单句语速使用 `[speech_rate].rate_factor` 修正，分时段语速不乘该系数。

## Docker

模型已经随镜像交付，默认运行不需要挂载模型目录。

```bash
docker build -t seacraft-asr:offline .
```

Dockerfile 会检查五个模型目录是否存在且非空。依赖安装保留 Docker 构建缓存，不使用 `--no-cache`。

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
