# 变更提案

## 为什么

当前服务同时维护离线 ASR、小语种 Whisper、Pyannote 说话人分离、音频质量分析、普通话检测和 BERT 五何分类，导致接口面、模型体积、依赖项和部署复杂度都高于实际需要。现在需要收缩为以 `/v1.1.8/seacraft_asr` 为核心的离线 ASR 服务，并改为将所需模型直接随镜像交付，消除模型目录挂载带来的部署依赖。

## 变更内容

- **BREAKING**：移除 `POST /v1.1.7/seacraft_asr`、`POST /audio/db_snr`、`POST /audio/detect_mandarin` 和 `POST /text/question`。
- 保留 `POST /v1.1.8/seacraft_asr` 的请求参数、响应结构和业务行为。
- 保留 `/v1.1.8/seacraft_asr` 使用的 Paraformer、VAD、PUNC、CAM++ 和 emotion2vec 模型。
- 移除 Whisper、Pyannote 和 BERT 模型及其加载、推理、配置和依赖链。
- 删除仅服务于被移除接口的路由、工具函数、数据模型和依赖项；保留共享的语速统计与教师/学生规则能力。
- 将模型放入仓库本地 `model/` 目录，并在构建镜像时复制到镜像内 `/app/model/`。
- 镜像默认使用 `/app/model/` 下的模型；允许通过挂载配置文件覆盖其他参数，不再要求挂载模型。
- 默认容器配置改为 `config.docker.toml`，其中模型路径统一指向 `/app/model/`。
- 取消 Docker 构建对仓库外 `wheel/pyarrow-*.whl` 的硬依赖，改为在容器依赖中显式锁定与 CentOS 7 兼容的 `pyarrow==20.0.0`。
- 新增 `model/README.md`，说明必须放入 `model/` 的五个模型目录及用途；模型文件不纳入 Git。

## 能力

### 新增能力

- `offline-asr-service`：定义保留的离线 ASR 接口面、`/v1.1.8/seacraft_asr` 的兼容性，以及所需模型的最小集合。
- `container-model-packaging`：定义模型随镜像交付、容器内路径、配置覆盖方式和构建依赖。

### 修改能力

- 无。当前项目尚无已归档的基线规格，本次以两个新能力建立行为契约。

## 影响

- API：删除四个接口及其路由注册；保留 `/v1.1.8/seacraft_asr`、`/get_status` 和 `/clear_tasks_list`。
- 模型：保留 Paraformer、VAD、PUNC、CAM++、emotion2vec；删除 Whisper、Pyannote、BERT。
- 依赖：删除 `faster-whisper`、`ctranslate2`、`pyannote.audio`、`transformers` 的直接依赖；`librosa` 仍由 FunASR 使用，必须保留。
- 部署：模型从宿主目录挂载改为镜像内置；Docker 构建上下文必须包含本地 `model/`，但模型文件仍不进入 Git。
- 文档：README、Docker 构建记录和配置说明需要同步新的接口面、模型目录结构和构建方式。
- OpenSpec：后续 change 名称统一采用 `YYYY-MM-DD-任务需求`，本次为 `2026-10-21-remove-infrequently-used-interfaces`。
