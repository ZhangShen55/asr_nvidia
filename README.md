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

# 语速计算配置
[speech_rate]
rate_factor = 0.7          # 单句语速修正系数（数值偏高时下调）

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

响应示例：

```json
{
  "language": "auto",
  "segments": [
    {
      "segment_text": "如果与中文相比，",
      "bg": "0.17",
      "ed": "1.13",
      "speed": 230,
      "segment_words": [],
      "role": "teacher",
      "emotion": "平淡"
    }
  ],
  "text": "如果与中文相比，...",
  "speed_info": [
    { "unit": 1,  "segment_info": { "segment_count": 45, "speed": [237, 220] } },
    { "unit": 5,  "segment_info": { "segment_count": 9,  "speed": [237, 220] } },
    { "unit": 10, "segment_info": { "segment_count": 5,  "speed": [237, 220] } }
  ],
  "load_audio_time_ms": "163.24",
  "gpu_time_ms": "1349.49"
}
```

响应字段说明：

| 字段 | 说明 |
|------|------|
| `segments[].speed` | **单句语速**（字/分钟）。分子为去除标点、空格后的实际内容数（中文按字、英文按单词、数字串各计 1）；分母为该句说话时长；再乘 `config.toml` 中 `[speech_rate].rate_factor`（默认 0.7）做经验修正 |
| `speed_info` | **分时段语速统计**，按 1/5/10 分钟三种窗口单位分别统计 |
| `speed_info[].unit` | 时间窗口单位（分钟） |
| `speed_info[].segment_info.segment_count` | 该单位下切出的时间窗口个数（= `speed` 数组长度） |
| `speed_info[].segment_info.speed` | 每个时间窗口的语速（字/分钟）列表 |

#### `speed_info` 分时段语速计算方式

对 1 / 5 / 10 分钟三种窗口单位，分别独立按下述步骤计算（`unit` 为窗口分钟数）：

**① 划分时间窗口**

以时间轴 `0` 秒为起点，按 `unit×60` 秒等分。设整段最后一句的结束时间为 `max_end`（秒），则窗口个数：

```
segment_count = ceil(max_end / (unit×60))
```

第 `k` 个窗口（`k` 从 0 开始）覆盖时间区间 `[k×unit×60, (k+1)×unit×60)`。

**② 统计每句的"实际内容字数"**

去除标点、空格等无关内容后计数：中文按字、英文按单词、数字串各计 1（与单句 `speed` 同口径）。

```
words(句子) = 中文字数 + 英文单词数 + 数字串个数
```

**③ 跨窗口的句子按时间重叠比例拆分**

若一句 `[bg, ed]` 跨越多个窗口，则把它的字数按"与各窗口的重叠时长 / 该句总时长"的比例分摊到对应窗口：

```
overlap(句子, 窗口k) = min(ed, 窗口k末) − max(bg, 窗口k首)
窗口k获得的字数 += words(句子) × overlap(句子, 窗口k) / (ed − bg)
```

**④ 计算每个窗口的语速**

分母为窗口的**标称时长（固定为 unit 分钟）**，因此窗口内停顿（空闲）越多，语速越低：

```
窗口语速 = 窗口内累计字数 / (窗口标称时长 / 60)
```

补充规则：

- **末窗口**：最后一个不足 `unit` 的窗口，标称时长改用**实际剩余时长** `max_end − k×unit×60`，避免被整窗时长稀释。
- **空窗口**：窗口内完全没有说话内容时，语速记为 `0`。
- 该统计**不**乘 `rate_factor`（与单句 `speed` 不同：空闲已通过固定分母体现）。

**计算示例**（`unit=1`，即 60 秒窗口）：

某句 `bg=58s, ed=63s`、字数 10，跨第 0、1 两个窗口：

- 与窗口0（0–60s）重叠 2 秒 → 分到 `10 × 2/5 = 4` 字；
- 与窗口1（60–120s）重叠 3 秒 → 分到 `10 × 3/5 = 6` 字。

若窗口0内另有一句贡献 10 字，则窗口0共 14 字，其语速 = `14 / (60/60) = 14`（字/分钟）。

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
