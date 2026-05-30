# SeaCraftASR - 智能语音转写与分析服务

基于 FastAPI 搭建的企业级语音处理后端服务，专注于教育场景（课堂录音分析），提供多维度语音AI能力。

## 🎯 核心功能

| 功能模块 | 描述 | 技术方案 |
|---------|------|---------|
| **ASR 转写** | 语音转文字 | Paraformer(中文) + Whisper(多语言) |
| **说话人分离** | 区分不同说话人 | CAM++ + Pyannote |
| **情感识别** | 分析语音情绪 | emotion2vec |
| **实时转写** | WebSocket 流式识别 | Paraformer-online |
| **五何分类** | 教师提问分类 | BERT 文本分类 |
| **角色识别** | 自动识别教师/学生 | 特征工程 + 规则引擎 |

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        API 路由层                            │
├─────────────────────────────────────────────────────────────┤
│  /v1.1.8/seacraft_asr    │  离线ASR转写（主接口）           │
│  /v1.0.1/seacraft_asr_online  │  WebSocket实时转写          │
│  /text/question          │  五何分类分析                   │
│  /audio/db_snr           │  音频质量分析                   │
│  /audio/detect_mandarin  │  普通话检测                     │
│  /get_status             │  服务状态监控                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        核心层 (Core)                         │
├─────────────────────────────────────────────────────────────┤
│  config.py    │  配置管理（支持热更新）                      │
│  models.py    │  AI模型懒加载与管理                          │
│  concurrency.py  │  GPU并发控制（信号量机制）                  │
│  logging.py   │  日志配置与管理                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       模型层 (Models)                        │
├─────────────────────────────────────────────────────────────┤
│  Paraformer  │  Whisper  │  emotion2vec  │  BERT  │  CAM++  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
asr_refine/
├── main.py                    # FastAPI 入口
├── config.toml               # 配置文件（TOML 格式）
├── requirements.txt          # 依赖包
├── Dockerfile               # 容器化部署
├── nginx.conf               # Nginx 多实例配置
├── start.sh                 # 启动脚本
├── api/
│   └── routes/
│       ├── asr.py           # 离线ASR转写
│       ├── ws_online.py     # WebSocket实时转写
│       ├── text.py          # 五何分类
│       ├── audio.py         # 音频处理
│       └── status.py        # 状态监控
├── core/
│   ├── config.py            # 配置管理
│   ├── models.py            # 模型加载
│   ├── concurrency.py       # GPU并发控制
│   └── logging.py           # 日志配置
├── entity/
│   └── data.py              # 数据模型
└── utils/
    ├── audio_utils.py       # 音频处理工具
    ├── feature_utils.py     # 特征提取
    ├── asr_stats.py         # 统计信息
    └── pynanote_speaker.py  # 说话人分离
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (GPU模式)
- 16GB+ 内存
- 50GB+ 磁盘空间（模型文件）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

配置文件为 `config.toml`（TOML 格式）。服务通过环境变量 `CONFIG_PATH` 指定配置路径，未设置时默认读取当前目录下的 `./config.toml`。

```toml
# 基础配置
id_engine = "1"
version = "seacraft-asr-app-v1.1.9"

# 设备与并发配置
device = "cuda:1"          # 推理设备
ngpu = 1                   # GPU 数量
ncpu = 4                   # CPU 线程数
concurrency = 5            # 单实例 GPU 并发数
instance_count = 4         # uvicorn 实例数（供 start.sh 与 Nginx 使用）

# 日志配置
log_path = "./asr_service.log"

# 热词文件路径
hotword_path = "/var/model_zoo/model_asr/.../hotword.txt"

# 模型路径配置
[model_paths]
vad_model_dir = "/var/model_zoo/model_asr/speech_fsmn_vad_zh-cn-16k-common-pytorch"
punc_model_dir = "/var/model_zoo/model_asr/punc_ct-transformer_cn-en-common-vocab471067-large"
asr_model_dir = "/var/model_zoo/model_asr/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
spk_model_dir = "/var/model_zoo/model_asr/speech_campplus_sv_zh_en_16k-common_advanced"
emotion_model_dir = "/var/model_zoo/model_asr/emotion2vec_plus_large"
asr_online_model_dir = "/var/model_zoo/model_asr/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online"
asr_online_punc_model_dir = "/var/model_zoo/model_asr/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
whisper_model_dir = "/var/model_zoo/model_asr/faster-whisper-large-v3"
pyannote_model_yml = "/var/model_zoo/model_asr/speaker-diarization-3.1/config.yaml"
bert_model_tokenizer = "/var/model_zoo/model_asr/bert-base-chinese"
bert_model_dir = "/var/model_zoo/model_asr/bert_output/checkpoint-88"

# 计算配置（faster-whisper）
[compute]
compute_type = "int8"      # int8 / float16 等

# 功能开关配置
[features]
open_spk = true            # 说话人分离
open_emotion = true        # 情感识别
ban_hotword = true         # 禁用热词
open_mul_lang = true       # 多语言(Whisper)
open_mul_spk = true        # 多说话人分离(Pyannote)
open_online = false        # 实时转写
```

### 启动服务

```bash
# 开发模式
python main.py

# 生产模式（多进程）
bash start.sh
```

## 📡 API 使用示例

### 离线ASR转写

```bash
curl -X POST "http://localhost:8083/v1.1.8/seacraft_asr" \
  -F "audioFile=@test.wav" \
  -F "language=auto" \
  -F "showSpk=true" \
  -F "showEmotion=true"
```

### WebSocket 实时转写

```javascript
const ws = new WebSocket('ws://localhost:8083/v1.0.1/seacraft_asr_online');
ws.onopen = () => {
  // 发送音频数据
  ws.send(audioChunk);
};
ws.onmessage = (event) => {
  console.log(JSON.parse(event.data));
};
```

## ⚙️ 功能开关说明

以下开关均位于 `config.toml` 的 `[features]` 段下（下表默认值为示例 `config.toml` 的取值；代码缺省值除 `ban_hotword` 外均为 `false`）：

| 配置项 | 说明 | 示例值 |
|-------|------|--------|
| `open_spk` | 开启说话人分离 | `true` |
| `open_emotion` | 开启情感识别 | `true` |
| `open_mul_lang` | 开启多语言(Whisper) | `true` |
| `open_mul_spk` | 开启多说话人分离(Pyannote) | `true` |
| `open_online` | 开启实时转写 | `false` |
| `ban_hotword` | 禁用热词功能 | `true` |

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t seacraft-asr .

# 运行容器
docker run -d \
  --gpus all \
  -p 8083:8083 \
  -v /path/to/models:/var/model_zoo \
  -v $(pwd)/config.toml:/app/config.toml \
  -e CONFIG_PATH=/app/config.toml \
  seacraft-asr
```

## 📊 监控指标

访问 `/get_status` 获取服务状态：

```json
{
  "id_engine": "1",
  "status": "living",
  "appVersion": "seacraft-asr-app-v1.1.9",
  "runTime": "1天 2小时 30分",
  "totalHaveDoneProcessTasks": 1523,
  "totalFailedTasks": 12,
  "offlineDone": 1200,
  "onlineDone": 323
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

## 📄 License

MIT License
