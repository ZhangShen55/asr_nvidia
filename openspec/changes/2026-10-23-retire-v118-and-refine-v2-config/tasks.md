# Tasks

## 1. 下线 v1.1.8 接口与专属代码

- [x] 1.1 删除 `api/routes/asr_v18.py` 并在 `main.py` 中取消其路由注册，验证 `/v1.1.8/seacraft_asr` 返回 404 且 `/v2/seacraft_asr` 不受影响
- [x] 1.2 精简 `api/routes/asr_common.py`：仅保留 `AsrContext`、`map_emotion`、`prepare_asr_context`、`cleanup_temp_files`、`generate_task_id`，移除热词分支（`request.hotWords` / `ban_hotword`）、`run_paraformer_asr`、`run_whisper_asr`、`_apply_itn` 与长音频分块逻辑，验证 `asr_qwen_common.py` 仍能正常导入
- [x] 1.3 清理 `core/models.py`：移除 `load_models_if_needed` 与 Whisper / Pyannote / BERT / SeACo-Paraformer / PUNC 加载路径，仅保留 `load_v2_models_if_needed`，验证无 SeACo-Paraformer 模型目录时服务可正常启动
- [x] 1.4 移除 `core/config.py` 中已下线能力对应的配置字段（`device`、`ngpu`、`ncpu`、`concurrency`、`hotword_path`、`id_engine`、`version`、模型路径的 asr/punc/whisper 项、`[audio_chunk]` 整段、`[features].open_mul_lang`、`ban_hotword`、`open_spk`、`open_emotion`），验证 `config.toml` 与 `config.docker.toml` 结构一致且服务可启动
- [x] 1.5 更新 `api/routes/status.py` 与 `core/status.py`：`id_engine` 不再从配置读取但保留该字段并返回固定占位值，`appVersion` 固定返回 `latest`，验证 `GET /get_status` 响应字段与规格一致

## 2. v2 模型开关解耦

- [x] 2.1 在 `core/config.py` 新增 `open_v2_role` 与 `open_v2_emotion` 配置项（默认均为 true），并移除 `open_spk` / `open_emotion`，验证缺省配置下两个开关为 true
- [x] 2.2 改造 `load_v2_models_if_needed`：CAM++ 由 `open_v2_role` 单独控制、emotion2vec 由 `open_v2_emotion` 单独控制，两者互不依赖，验证只关闭其中一个时另一个仍正常加载
- [x] 2.3 保证模型按需懒加载：请求未携带 `showSpk` / `showEmotion` 时不加载对应模型，验证纯转写请求的显存占用不含 CAM++ 与 emotion2vec
- [x] 2.4 验证关闭开关后响应分段中对应的 `role` / `emotion` 字段为 `null` 且转写结果正常返回

## 3. 配置项清理与重建

- [x] 3.1 按设计决策 3 重写 `config.toml`：只保留 `instance_count`、`log_path`、`[log]`、`[model_paths]` 三个模型路径、`[speech_rate]`、`[features]`、`[itn]`、`[qwen_asr]`、`[model_device]`，验证所有键在代码中均有读取方
- [x] 3.2 以同一结构同步 `config.docker.toml`，验证容器默认配置的模型路径指向 `/app/model/` 且 GPU 落卡为 `cuda:0`
- [x] 3.3 在 `config.toml` / `config.docker.toml` 中显式配置 `[itn].remove_interjections = true`（继续过滤语气词），验证默认配置下语气词被剔除且其产生的标点副作用由标点卫生规则清理
- [x] 3.4 验证 `instance_count` 生效且默认为 1：缺省配置下启动 1 个 uvicorn 实例；配置为 2 时并行 2 个实例，且各实例独立持有请求处理槽位与模型加载能力

## 4. 日志保留策略

- [x] 4.1 在 `core/logging.py` 实现按天 + 按大小双轮转的自定义 handler：单文件超过 `[log].max_size_mb`（默认 100）时切出 `.1`、`.2` 序号文件并覆盖最旧，验证连续写入可产出多个历史文件
- [x] 4.2 每天首次写入前切分带日期的历史文件，并实现启动时与写入过程中清理超过 `[log].max_days`（默认 7）的日志文件，验证过期文件被删除
- [x] 4.3 补充单元测试覆盖「超限切分」「按天切分」「过期清理」「缺省 7 天 / 100MB」，验证测试全部通过

## 5. ITN 与标点卫生规则

- [x] 5.1 实现标点卫生工具函数：分段文本 `strip()` 后只剩标点则判定为无效段；段首为标点则移除开头连续标点序列，验证 `。` / `！` 段被识别为无效、`，，就。` 变为 `就。`
- [x] 5.2 在 v2 响应组装链路中接入该工具：无效段从 `segments` 数组移除且不参与整段文本拼接，验证 `segments` 长度与拼接文本一致
- [x] 5.3 保证规则在 ITN 关闭时同样执行（与 ITN 开关解耦），验证 `[itn].enabled=false` 时段首标点仍被清理
- [x] 5.4 补充单元测试覆盖「纯标点段」「段首标点」「连续标点」「正常文本不受影响」「ITN 关闭时仍生效」，验证测试全部通过

## 6. 语种占比简写

- [x] 6.1 在 `utils/qwen_response.py` 增加「模型语种名 → 简写」映射表（zh/en/yue/de/it/pt/es/fr/th/ms/ja/ko/ru/hi/ar/vi/id/nl/tr/pl/el/he），命中替换键、未命中原样透传，验证 `Chinese` → `zh`、`English` → `en`
- [x] 6.2 保持语种判定逻辑与百分比计算不变，验证中英混合音频的 `language_ratio` 键为 `zh`/`en`、各值之和为 100.0
- [x] 6.3 补充单元测试覆盖「已知语种映射」「未知语种透传」「单语种 100%」「无有效文本返回空字典」，验证测试全部通过

## 7. 镜像与依赖清理

- [x] 7.1 修改 `Dockerfile` 的必需模型目录校验：从五个减到 VAD、CAM++、emotion2vec 三个，删除 `faster-whisper-large-v3` 的条件校验分支，验证缺少已下线模型目录时构建不报错、缺少任一必需模型时构建失败并指出缺失目录名
- [x] 7.2 从 `requirements-pip-extra.txt` 删除 `faster-whisper` / `ctranslate2` / `importlib_resources`，若该层只剩 WeTextProcessing 则并入基础依赖层的既有安装步骤，验证依赖安装步骤保持缓存命中
- [x] 7.3 保留 WeTextProcessing 与 pynini（ITN 仍需要），验证镜像内 `[itn].enabled=true` 可正常完成数字规范化、依赖缺失时按 `passthrough` 降级不影响请求成功

## 8. 进程命名与本地模型显存可观测

- [x] 8.1 在服务启动入口调用 `setproctitle.setproctitle("asr-offline-model")` 覆写进程命令行首项，验证 `nvidia-smi --query-compute-apps=pid,process_name,used_memory` 中本服务显示为 `asr-offline-model`
- [x] 8.2 保持 VAD、CAM++、emotion2vec 三个模型在同一服务进程内加载（不引入子进程与跨进程 RPC），验证显存中只有一份 CUDA context、无额外约 0.8 GB 开销
- [x] 8.3 保留并验证懒加载语义：对应开关关闭或请求未触发时不加载权重，首次触发时加载后复用，验证 `showSpk`/`showEmotion` 全关时显存中只有 VAD
- [x] 8.4 在 `/get_status` 中暴露三个本地模型各自的加载状态与显存占用明细，验证未加载模型显存明细为 0、已加载模型可读出占用
- [x] 8.5 在依赖清单中加入 `setproctitle` 并在镜像中安装，验证容器内 `nvidia-smi` 同样能显示 `asr-offline-model`
- [x] 8.6 验证 `instance_count = 2` 时出现两行 `asr-offline-model` 且本地模型显存为单实例的 2 倍，文档中说明该放大关系

## 9. 文档

- [x] 9.1 更新 README / 接口文档：移除 v1.1.8 相关说明，补充 `/v2/seacraft_asr` 的 `language_ratio` 简写键、`open_v2_role` / `open_v2_emotion` 开关与新的配置项清单
- [x] 9.2 在部署文档中记录 Qwen 容器的方案 A 扩容参数（`--api-server-count 4 --max-model-len 1024 --max-num-seqs 128 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.45`）与启动后需核对的 `GPU KV cache size`、`Maximum concurrency` 两行日志
- [x] 9.3 在运维文档中说明用 `nvidia-smi` 的进程名 `asr-offline-model` 定位本服务显存的方法，以及「按模型区分显存」需改用 `/get_status` 的模型显存明细
- [x] 9.4 在发布说明中标注 v1.1.8 下线属破坏性变更及回滚方式（镜像回滚至 `v3.3`）
- [x] 9.5 产出 `docs/npu-910b-feasibility-report.md`：VAD / CAM++ / emotion2vec 在华为昇腾 910B（torch_npu）上的可行性调研、改造点、风险与结论（项目能否在 NPU 与 GPU 上双跑通）

## 10. 集成验证

- [x] 10.1 在宿主环境用 `config.toml` 启动服务，验证 `/v2/seacraft_asr` 在 `showSpk`/`showEmotion` 四种组合下的响应正确，且 `/v1.1.8/seacraft_asr` 返回 404
- [x] 10.2 用 `test/audio_file` 下的音频做回归：对照本 change 之前的转写结果，确认文字内容无回退、`language_ratio` 简写键正确、标点脏输出已消除
- [x] 10.3 验证 `nvidia-smi` 中本服务进程名为 `asr-offline-model`，且 `/get_status` 中三个本地模型的加载状态与显存明细符合预期
- [x] 10.4 运行 `tests/` 全量测试，验证全部通过
- [x] 10.5 用 `test/audio_file` 下的音频做效果比对：与本 change 之前的转写结果逐段对照，确认文字内容无回退、`language_ratio` 简写键正确、段首/连续标点与纯标点段已消除

## Workflow follow-up

- **本次不构建镜像、不部署上线**（业务明确要求）：本 change 只完成后端运行、测试与效果比对；镜像构建需单独确认后执行。
- Qwen 容器按方案 A 参数扩容属部署侧动作，不在本 change 的实现范围内。
- 发布上线与镜像归档在评审通过后进行。
