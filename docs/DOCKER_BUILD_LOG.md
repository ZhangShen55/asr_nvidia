# Docker 构建记录（jy-algorithm-app-asr-offline:v2.0）

## 问题 1：Miniconda latest 需要 glibc >= 2.28

- **现象**：`Installer requires GLIBC >=2.28, but system has 2.17`
- **处理**：固定 `Miniconda3-py310_23.11.0-2-Linux-x86_64.sh`

## 问题 2：av==14.4.0 pip 源码编译失败

- **现象**：`Package libavformat was not found in pkg-config`
- **原因**：PyPI 下载 tar.gz 源码包，容器内无 ffmpeg dev 头文件；且 av 14.4 倾向 ffmpeg 7
- **处理**：`conda-forge` 预装 `av=14.4.0` + `ffmpeg`；pip 改用 `requirements-pip.txt`（不含 av）

## 问题 3：运行期 Segmentation fault（Cython）

- **现象**：模型加载阶段 `uvicorn` 崩溃，日志 `Segmentation fault (core dumped)`
- **处理**：镜像运行层改回 **Python 源码**，暂不启用 Docker 内 Cython（`scripts/build_cython.sh` 保留供本地可选使用）

## 构建与验证命令

```bash
docker build -t jy-algorithm-app-asr-offline:v2.0 .
docker run -d --name jy-asr-offline-v2 --gpus '"device=0"' -p 9000:9000 \
  -v /var/model_zoo:/var/model_zoo:ro \
  -v $(pwd)/config.docker.toml:/config.toml:ro \
  -e CONFIG_PATH=/config.toml \
  jy-algorithm-app-asr-offline:v2.0
```

## 问题 4：ASR 接口 500（torchaudio backend）

- **现象**：`ValueError: Unsupported backend 'ffmpeg'`
- **处理**：`utils/audio_utils.load_audio_tensor()` 优先 ffmpeg，失败回退默认 soundfile

## 验证结果（2026-06-04）

| 步骤 | 结果 |
|------|------|
| `docker build -t jy-algorithm-app-asr-offline:v2.0 .` | 成功（约 60min 首构建，依赖层可缓存） |
| `GET /get_status` | 200，`status: living` |
| `POST /v1.1.8/seacraft_asr`（chinEng-16k.wav） | 200，182 segments，含 `speed` |

生产部署建议：`config.docker.toml` 中 `device=cuda:0` 对应 `--gpus device=N` 映射；多卡宿主机用 `cuda:1` 时挂载原 `config.toml` 并 `--gpus all`。
