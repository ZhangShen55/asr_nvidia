# 任务

## 1. 收缩接口与模型代码

- [x] 1.1 删除 `api/routes/asr_v17.py`、`api/routes/audio.py` 和 `api/routes/text.py`，并从 `main.py` 移除对应 import 与 `include_router()`；验证应用路由表中不再出现四个旧接口
- [x] 1.2 确保 `api/routes/asr_v18.py` 和 `api/routes/asr_common.py` 的请求参数、响应字段及业务分支未被改动；验证 `git diff` 不包含 v1.1.8 行为变更
- [x] 1.3 从 `core/models.py` 移除 Whisper、Pyannote、BERT 的导入、缓存、加载逻辑和访问函数；使用 `rg` 验证代码中不再存在这些模型引用
- [x] 1.4 删除 `utils/pynanote_speaker.py` 和 `utils/audio_analyze.py`；使用 `rg` 验证没有路由或工具模块引用它们
- [x] 1.5 更新 README 的接口表和快速调用示例，只保留三个业务接口；验证文档中的每个路径都存在于应用路由表

## 2. 清理共享工具、实体与依赖

- [x] 2.1 从 `utils/audio_utils.py` 删除 Whisper 导入、`preprocess_audio2wav()`、`split_audio()` 和 `detect_language()`，保留 v1.1.8 使用的解码、标准化、裁剪和长音频分块函数；使用 `rg` 验证无残留引用
- [x] 2.2 从 `utils/feature_utils.py` 删除五何标签、文本分段整理及分类结果格式化函数，保留 `extract_features()`、`identify_teacher()`、`calculate_speech_rate()` 和 `build_speed_info()`；验证 v1.1.8 所需导入仍可解析
- [x] 2.3 从 `entity/data.py` 删除仅供已移除接口使用的 Pydantic 模型和未使用导入，保留 `AsrRequestParams` 与 `get_asr_params()`；验证 ASR 路由导入成功
- [x] 2.4 从 `core/logging.py` 删除 `faster_whisper` 日志降噪配置；验证日志初始化不引用已删除模型
- [x] 2.5 从 `requirements.txt` 和 `requirements-pip.txt` 删除 `faster-whisper`、`ctranslate2`、`pyannote.audio`、`transformers`，保留 `librosa`；运行依赖解析检查确认剩余包可解析
- [x] 2.6 执行 Python 编译检查并验证 `main.py`、`core`、`api`、`entity`、`utils` 无语法错误

## 3. 建立模型目录与配置

- [x] 3.1 创建 `model/README.md`，列出五个必需目录、用途、放行说明和“模型必须完整放入对应目录”的约束；验证 README 包含全部目录名且可被 Git 跟踪
- [x] 3.2 修改 `.gitignore` 忽略 `model/` 下的模型内容但放行 `model/README.md`；验证放入样例权重后 `git status` 不显示权重、仍显示 README
- [x] 3.3 修改 `config.toml`，移除 Whisper、Pyannote、BERT 路径及 `open_mul_lang`、`open_mul_spk`、`compute_type`，保留模型路径改为 `./model/<目录名>`；验证本地配置可被 `tomli` 加载
- [x] 3.4 修改 `config.docker.toml`，移除上述低频配置并将五个保留模型路径改为 `/app/model/<目录名>`；验证配置中不存在 `/var/model_zoo`
- [x] 3.5 更新 `core/config.py`，删除低频模型路径和开关属性；验证访问保留配置属性成功且旧属性引用为零

## 4. 修改 Docker 模型打包与构建依赖

- [x] 4.1 修改 `.dockerignore`，允许 `model/` 下的 `.bin`、`.safetensors` 和其他权重进入构建上下文；使用 `docker build` 前置文件检查验证模型文件可见
- [x] 4.2 修改 Dockerfile，在依赖层之后复制 `model/` 到 `/app/model/`，并在复制前验证五个必需目录存在且非空；验证缺少任一目录时构建明确失败
- [x] 4.3 修改 Dockerfile，默认将 `config.docker.toml` 复制为 `/config.toml`，保留 `CONFIG_PATH` 覆盖能力；验证未挂载配置时读取容器专用配置
- [x] 4.4 修改 Dockerfile 和 `requirements-pip.txt`，删除本地 `wheel/pyarrow-*.whl` 前置依赖并锁定 `pyarrow==20.0.0`；验证不存在 `wheel/` 时依赖解析仍选择 CP310 manylinux2014 wheel
- [x] 4.5 更新 README 和 `docs/DOCKER_BUILD_LOG.md` 的模型准备、构建、启动说明，移除模型挂载示例；验证文档命令不包含 `/var/model_zoo` 挂载且启动命令可直接运行

## 5. 镜像集成验收

- [x] 5.1 使用准备好的五个模型和 `config.docker.toml` 构建镜像；验证构建成功且日志显示保留模型目录检查通过
- [x] 5.2 在不挂载任何模型目录的情况下启动容器；验证 `/get_status` 返回服务存活并显示预期版本
- [x] 5.3 检查运行中容器的 `/app/model` 和镜像内 `/config.toml`；验证五个模型目录存在、路径均指向 `/app/model/`、且没有 Whisper、Pyannote 或 BERT 目录
- [x] 5.4 读取 OpenAPI 并请求四个已移除路径；验证业务接口只剩三个，旧路径均返回 `404`
- [x] 5.5 使用真实音频调用 `/v1.1.8/seacraft_asr`，启用说话人、情绪和角色识别；验证响应包含 `segments`、`text`、`speed_info`、`role`、`emotion` 和耗时字段
- [x] 5.6 输出最终 `model/` 放置说明，逐项列出五个目录、用途和完整目录复制要求；验证说明与 `model/README.md` 和镜像检查清单一致

## 后续流程

- 评审通过后使用 `openspec-apply-change` 执行本变更。
- 所有任务和镜像验收完成后归档该 change。
