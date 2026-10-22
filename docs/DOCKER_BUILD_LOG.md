# Docker 构建与运行说明

## 镜像内容

镜像基于 CentOS 7 + CUDA 12.1.1，使用 Python 3.10 Conda 环境。业务只保留离线 ASR、状态查询和任务列表清理。

五个模型目录随镜像复制到 `/app/model/`：

- `speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `punc_ct-transformer_cn-en-common-vocab471067-large`
- `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `speech_campplus_sv_zh_en_16k-common_advanced`
- `emotion2vec_plus_large`

启用法语能力时另需提供 `faster-whisper-large-v3`（随镜像复制到 `/app/model/faster-whisper-large-v3`）。

## 构建

从仓库根目录执行：

```bash
docker build -t jy-algorithm-app-asr-offline:v3.2 .
```

构建要求：

1. 五个模型目录完整存在于 `model/`。
2. 构建过程保留 Docker 缓存，不执行 `--no-cache`。
3. 不要求提供 `wheel/pyarrow-*.whl`。
4. `config.docker.toml` 中 `open_mul_lang = true` 时，`model/faster-whisper-large-v3` 必须存在且非空，否则构建失败并提示缺失目录。

## 依赖处理

- `av` 由 conda-forge 安装，避免在 CentOS 7 上源码编译。
- `pyarrow==20.0.0` 由 `requirements-pip.txt` 显式锁定；CP310 wheel 标签兼容 CentOS 7 的 glibc 2.17。
- `pynini` 由 conda-forge 安装（PyPI 的 CP310 wheel 与 glibc 2.17 不兼容），`WeTextProcessing==1.2.0` 以 `--no-deps` 安装，复用 conda 提供的 pynini。
- 小语种与文本规范化依赖放在独立层 `requirements-pip-extra.txt`，基础依赖清单变化时不必重新下载 torch 等大包。
- v2 链路（Qwen3-ASR）新增的 `httpx` 放在独立层 `requirements-pip-runtime.txt`，在**运行层**安装；基础依赖清单 `requirements-pip.txt` 保持与历史构建逐字节一致，从而命中既有构建缓存（否则 torch 766 MB 需要重新下载，构建时间从分钟级变成小时级）。
- Cython 在容器内运行业务模块时曾出现段错误，因此镜像继续使用 Python 源码。

## 运行

默认镜像不需要挂载模型：

```bash
docker run -d \
  --name jy-asr-offline-v3.2 \
  --gpus '"device=1"' \
  -p 9000:9000 \
  jy-algorithm-app-asr-offline:v3.2
```

镜像内 `/config.toml` 来自 `config.docker.toml`。如需调整并发或设备，可挂载配置文件；模型路径继续使用 `/app/model/`。

## 已验证行为（v3.2）

验证时记录以下结果：

- 构建成功，五个模型目录检查通过。
- 不挂载模型时 `/get_status` 返回 `status=living`。
- OpenAPI 仅包含 `/v1.1.8/seacraft_asr`、`/get_status` 和 `/clear_tasks_list`。
- 已移除接口返回 `404`。
- 真实音频请求 `/v1.1.8/seacraft_asr` 成功。
- `language=fr` 在未开启小语种时返回 `{"msg": "未开启小语种识别或模型未就绪", "code": 4003}`；开启后由 Whisper 返回分段与文本。
- 未知语言与显式空 `language` 返回 `{"msg": "不支持的语言: <lang>", "code": 4009}`。
- `open_mul_lang = true` 且缺少 `faster-whisper-large-v3` 时构建失败，并输出缺失目录提示。
- OpenAPI 同时包含 `/v1.1.8/seacraft_asr` 与 `/v2/seacraft_asr`；`/v1.1.8` 的请求与响应行为未受影响（响应不含 `language_ratio`）。
- `/v2/seacraft_asr` 真实音频请求成功，返回 `language`、`segments`、`text`、`speed_info` 与 `language_ratio`；`showSpk=true&showEmotion=true` 时 `role` / `emotion` 填充真实值。
- `/v2/seacraft_asr` 提交不支持的语种返回 `{"msg": "不支持的语言", "code": 4009}`；无语音音频返回 `{"msg": "音频文件为空或未检测到任何人声", "code": 4008}`。
- Qwen3-ASR 服务不随本镜像交付，通过 `config.docker.toml` 的 `[qwen_asr].base_url`（默认 `http://192.168.29.12:8806`）以「IP + 端口」访问，不要求两个容器在同一 Docker 网络。
