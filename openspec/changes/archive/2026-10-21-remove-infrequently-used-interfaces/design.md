# 设计

## 背景

当前 `main.py` 装配四条业务路由，`core/models.py` 同时管理 ASR、情绪、Whisper、Pyannote 和 BERT 五类模型。`/v1.1.8/seacraft_asr` 与 `/v1.1.7/seacraft_asr` 共用 ASR 核心，但只有 v1.1.7 会使用 Whisper 和 Pyannote；BERT 仅由 `/text/question` 使用；音频分析工具仅由 `/audio/*` 使用。

镜像当前从 `config.toml` 读取 `/var/model_zoo` 路径，并要求运行容器时挂载模型。Docker 构建上下文还明确忽略 `.bin` 和 `.safetensors`，无法支持模型随镜像交付。需求动机见 `proposal.md`。

## 目标与非目标

**目标：**

- 将路由面收缩为 `POST /v1.1.8/seacraft_asr`、`GET /get_status` 和 `DELETE /clear_tasks_list`。
- 将模型集合收缩为 Paraformer、VAD、PUNC、CAM++ 和 emotion2vec。
- 让默认镜像无需模型挂载即可启动并完成离线 ASR。
- 保留配置文件覆盖能力，避免把运行参数烘焙进不可变镜像。
- 让模型可在本地存在、不进入 Git，但可靠进入 Docker 构建上下文。

**非目标：**

- 不修改 `/v1.1.8/seacraft_asr` 的参数、响应字段、长音频策略或角色识别规则。
- 不修复现有 GPU 并发信号量、临时文件名或统计重置问题。
- 不训练、转换、量化或重新分发模型。
- 不改变 Nginx 暴露端口、多实例启动方式或 CUDA 基础镜像。
- 不把模型权重提交到 Git。

## 决策

### 1. 以路由依赖删除而不是兼容层保留

直接删除四个路由及其注册：

- 删除 `api/routes/asr_v17.py`，不再保留法语分流入口。
- 删除整个 `api/routes/audio.py` 和 `api/routes/text.py`。
- 从 `main.py` 删除对应 import 与 `include_router()`。

这样能让被移除接口返回 FastAPI 默认的 `404`，且不会继续加载只在旧路由使用的模块。相比保留路由并返回“已下线”，直接移除可以真正缩小攻击面、依赖和镜像体积。

### 2. 按使用归属删除模型与工具，保留共享能力

删除 Whisper、Pyannote、BERT 的加载缓存、模型路径、功能开关和推理函数。同时删除仅被旧接口引用的 `pynanote_speaker.py`、`audio_analyze.py`、音频切片辅助函数和五何文本整理函数。

以下共享能力必须保留：

- `extract_features()` 和 `identify_teacher()`，供 v1.1.8 做角色推断。
- `calculate_speech_rate()` 和 `build_speed_info()`，供 v1.1.8 生成语速信息。
- 音频标准化、裁剪、长音频切割和临时文件清理。
- `AsrRequestParams` 与 `get_asr_params()`。

`librosa` 虽然曾用于被删除的音频分析接口，但 FunASR 1.2.6 自身依赖它，因此保留。

### 3. 使用五个固定模型目录并随镜像复制

仓库根目录采用以下结构：

```text
model/
├── README.md
├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── punc_ct-transformer_cn-en-common-vocab471067-large/
├── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
├── speech_campplus_sv_zh_en_16k-common_advanced/
└── emotion2vec_plus_large/
```

镜像运行层将整个目录复制为 `/app/model/`。复制放在依赖安装之后、业务源码复制之前，使模型变化不触发依赖重装，也避免源码变化导致模型层重新传输。

Dockerfile 在复制前后增加 shell 检查，逐一验证五个目录存在且非空。这样可以在构建阶段暴露模型缺失，而不是等容器启动或首个请求时才失败。

### 4. 使用双配置并固定模型路径口径

- `config.toml` 面向仓库本地运行，模型路径改为 `./model/<目录名>`。
- `config.docker.toml` 面向容器，模型路径改为 `/app/model/<目录名>`，设备为 `cuda:0`，实例数为 1。
- Dockerfile 只把 `config.docker.toml` 复制为镜像内 `/config.toml`。
- `CONFIG_PATH` 覆盖机制保持不变，运维可挂载配置修改并发、GPU、语速和长音频参数。

配置仍允许修改模型路径，但 `model/README.md` 明确默认部署应保留 `/app/model/` 路径。这样既满足“挂载配置改其他参数”，也不锁死未来显式替换模型的运维能力。

### 5. 模型不进入 Git，但必须进入 Docker 上下文

`.gitignore` 增加 `/model/*` 并放行 `/model/README.md`，防止权重被误提交。`.dockerignore` 删除 `*.bin` 和 `*.safetensors` 排除规则，确保 Docker 构建上下文包含模型。

不使用“Dockerfile 从外部路径下载模型”的方式，因为目标部署需要可离线构建和可复现镜像；也不使用 bind mount，因为本次明确要求模型随镜像交付。

### 6. 移除低频依赖并显式锁定 PyArrow

从两份依赖清单移除 `faster-whisper`、`ctranslate2`、`pyannote.audio` 和 `transformers`。`librosa`、`datasets`、`av`、`torch` 和 `torchaudio` 保留。

现有 Dockerfile 强制要求 `wheel/pyarrow-20.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`，但仓库不含该文件且 `.gitignore` 忽略所有 wheel。复核确认 PyPI 提供的 CP310 wheel 标签为 `manylinux_2_17_x86_64.manylinux2014_x86_64`，与 CentOS 7 的 glibc 2.17 兼容。

因此决策为：

- 从 Dockerfile 删除 `COPY wheel/...` 和显式本地 wheel 安装。
- 在 `requirements-pip.txt` 增加 `pyarrow==20.0.0`。
- 继续使用 `--prefer-binary` 和现有镜像源安装。

该方案消除必须人工准备外部 wheel 的硬前置条件，同时通过版本锁定避免 pip 选择可能要求更高 glibc 的新版本。

### 7. 以接口面和模型可用性作为验收主线

不新增单元测试框架，优先使用镜像级验收：

1. 启动默认镜像时不挂载模型。
2. 检查 OpenAPI 中只剩三个业务接口，四个旧接口返回 `404`。
3. 调用 `/get_status` 确认服务存活。
4. 使用音频样例调用 `/v1.1.8/seacraft_asr`，验证说话人、情绪、角色和语速字段。
5. 检查容器内不存在 Whisper、Pyannote 和 BERT 模型目录。

这样可以覆盖本次最容易出错的装配、路径和依赖问题。

## 风险与权衡

- **构建上下文变大**：模型会让 `docker build` 上下文显著增大，并使模型层缓存频繁失效。缓解方式是模型单独成层且放在依赖层之后，同时保持固定目录名。
- **`.dockerignore` 误排除权重**：当前规则会排除 `.bin` 和 `.safetensors`。实施时必须删除这两条并执行一次干净的 `docker build` 验证。
- **配置挂载仍可破坏模型路径**：运维挂载的配置若把模型指向其他路径，服务可能找不到模型。`config.docker.toml` 和 `model/README.md` 将提供可直接使用的 `/app/model/` 模板，并在文档中提示不要改动模型路径。
- **移除 `transformers` 可能暴露隐藏依赖**：代码层面 BERT 是唯一直接使用者，但 FunASR/ModelScope 的运行时分支可能存在可选导入。缓解方式是在最终镜像内启动服务并完成一次真实 v1.1.8 请求。
- **PyArrow 依赖网络下载**：构建仍需要访问配置的 PyPI 镜像。若部署环境完全离线，应另有内部包源；本次不继续依赖未跟踪的本地 wheel。
- **模型许可与镜像体积**：模型随镜像分发可能受许可证和存储配额限制。该问题属于交付策略，需要在放入模型前由模型提供方确认。

## 迁移计划

1. 现有 v1.1.7 调用方改调 v1.1.8；依赖法语 Whisper 的调用方必须迁移到其他服务。
2. 音频质量和普通话检测调用方改由上游或其他服务处理。
3. 五何分类调用方在请求 v1.1.8 后自行保留分段结果，或将分类任务迁移到独立服务。
4. 将五个模型目录放入仓库根目录 `model/`，确认 `model/README.md` 可被 Git 跟踪而权重不被跟踪。
5. 构建新镜像并在不挂载模型的情况下启动，完成健康检查和 v1.1.8 冒烟测试。
6. 若验收失败，回滚本次提交；旧镜像和旧接口仍可作为短期回退路径。

## 待决问题

无。模型目录结构、模型不纳入 Git、配置挂载边界和 PyArrow 处理方式均已确认。
