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
├── config.json               # 配置文件
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

编辑 `config.json`：

```json
{
  "device": "cuda:0",
  "ngpu": 1,
  "concurrency": 5,
  "open_spk": true,
  "open_emotion": true,
  "asr_model_dir": "/path/to/paraformer",
  "whisper_model_dir": "/path/to/whisper"
}
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

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `open_spk` | 开启说话人分离 | `true` |
| `open_emotion` | 开启情感识别 | `true` |
| `open_mul_lang` | 开启多语言(Whisper) | `false` |
| `open_online` | 开启实时转写 | `false` |
| `open_mul_spk` | 开启多说话人分离(Pyannote) | `false` |
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
  -v $(pwd)/config.json:/app/config.json \
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
