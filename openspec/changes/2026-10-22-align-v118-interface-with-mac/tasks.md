# Tasks

## 1. 请求契约与语言分流

- [x] 1.1 修改 `entity/data.py`：`showRoleIdentify` 默认值改为 `false`（`showSpk`、`showEmotion` 保持 `false`），并通过 `inspect.signature` 断言四个参数的默认值
- [x] 1.2 修改 `entity/data.py`：`get_asr_params` 增加 `Request` 依赖并检查表单中是否显式包含 `language`，用测试覆盖“未提交 → auto”“显式空串 → 空串”两种情形
- [x] 1.3 修改 `api/routes/asr_v18.py`：加入语言归一化与 `auto/zh/en → Paraformer`、`fr → Whisper`、其余 → `4009` 的分发，用测试覆盖归一化、未知语言与空语言
- [x] 1.4 修改 `api/routes/asr_v18.py`：法语链路在 `open_mul_lang` 未开启或模型未就绪时返回 `4003` 且不读取音频，用测试覆盖两种失败分支

## 2. Whisper 小语种转写

- [x] 2.1 修改 `core/config.py`：新增 `whisper_model_dir` 与 `open_mul_lang` 配置项，默认值与 Mac 一致
- [x] 2.2 修改 `core/models.py`：按需导入 `faster_whisper` 并在 `open_mul_lang` 开启时加载模型（`compute_type` 固定 `int8`、设备索引取自 `settings.device`），提供 `get_whisper_model()`
- [x] 2.3 在 `api/routes/asr_common.py` 新增 `run_whisper_asr`：复用 `transcribe_with_gpu_lock`，输出与 Paraformer 一致的顶层字段，分段按需携带 `role`/`emotion` 空值
- [x] 2.4 新增 `tests/test_v118_multilingual.py`：验证法语请求走 Whisper、顶层字段集合、词级时间戳与 `4003`/`4009` 分支

## 3. 文本规范化与语速口径

- [x] 3.1 新增 `utils/itn.py`（与 Mac 同源）：提供 `ItnConfig` 与 `normalize_segments`，依赖缺失或失败时按 `passthrough` 降级
- [x] 3.2 修改 `api/routes/asr_common.py`：新增 `_apply_itn`，并把 Paraformer 与 Whisper 两条链路的 `text` 字段统一为规范化后的分段拼接
- [x] 3.3 修改 `utils/feature_utils.py`：`count_content_words` 改为 Unicode 感知实现，用测试验证带重音的法语词汇被计入词数
- [x] 3.4 新增 `tests/test_itn.py`：验证中文与英文规范化结果、不支持语言跳过、依赖缺失时透传

## 4. 配置与依赖

- [x] 4.1 修改 `config.toml` 与 `config.docker.toml`：新增 `whisper_model_dir`、`open_mul_lang` 与 `[itn]` 段，默认 `open_mul_lang = false`
- [x] 4.2 修改 `requirements.txt` 与 `requirements-pip.txt`：新增 `faster-whisper==1.1.1`、`ctranslate2==4.4.0`、`WeTextProcessing==1.2.0`，并在 `asr` conda 环境验证可导入
- [x] 4.3 修改 `core/config.py`：新增 `itn_config` 属性读取 `[itn]` 段并提供默认值

## 5. 镜像与模型清单

- [x] 5.1 修改 `Dockerfile`：依赖层通过 conda-forge 安装 `pynini`，`COPY config.docker.toml /config.toml` 提前到模型校验之前，按 `open_mul_lang` 条件校验 `faster-whisper-large-v3`
- [x] 5.2 修改 `model/README.md`：补充可选的 `faster-whisper-large-v3` 目录说明与开启方式
- [x] 5.3 构建镜像（不清理缓存），确认必需模型校验通过、缺失法语模型时在开启开关的场景下构建失败并给出明确提示

## 6. 端到端验证

- [x] 6.1 使用 `asr` conda 环境启动后端服务，验证中英文请求成功、`showRoleIdentify` 默认不返回角色字段
- [x] 6.2 临时开启 `open_mul_lang` 并指向本机 `faster-whisper-large-v3`，验证法语请求经 Whisper 返回分段与文本
- [x] 6.3 启动容器并验证 `/v1.1.8/seacraft_asr` 可用，测试完成后删除容器
- [x] 6.4 运行 `python -m unittest discover -s tests` 全部通过，并更新 `README.md` 的接口契约与模型说明

## Workflow follow-up

- 变更实现与验证完成后归档该 change，并同步主规格 `offline-asr-service` 与 `container-model-packaging`。
- 归档后按中文提交规范提交并推送到 `refactor/remove-infrequently-used-interfaces` 分支。
