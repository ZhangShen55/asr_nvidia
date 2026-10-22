# Tasks

## 1. 配置与 Qwen 客户端

- [x] 1.1 修改 `core/config.py`：新增 `[qwen_asr]` 段属性（`enabled`、`base_url`、`model`、`segment_concurrency`（单音频内同时发往 Qwen 的段请求数，默认 4）、`request_concurrency`（全部音频合计请求数，默认 32）、`audio_concurrency`（同时处理的音频数，默认 8）、`queue_max_size`、`queue_wait_timeout_s`、`request_timeout_s`、`connect_timeout_s`、`retry_count`、`min_segment_duration_s`、VAD 参数）与 `[model_device]` 段属性（`vad`、`emotion2vec`、`speech_campplus`），缺项回落到默认值 4 / 32 / 8 / 队列满立即拒绝 / GPU 0；用单测断言默认值与配置覆盖两种情形
- [x] 1.2 修改 `config.toml` 与 `config.docker.toml`：写入 `[qwen_asr]` 段（本地 `base_url` 指向 `http://127.0.0.1:8806`，容器配置指向 `http://192.168.29.12:8806`）与 `[model_device]` 段（三个模型统一 GPU 0），验证两份配置可被 `core/config.py` 正确解析
- [x] 1.3 新增 `core/qwen_asr_client.py`：实现 `/v1/models` 模型发现（配置了 `model` 则跳过）、`/v1/audio/transcriptions` multipart 调用、超时与重试；按「`auto`/未提交则不下发 `language`、具体语种原样透传」的规则构造请求体；用打桩 HTTP 客户端的测试验证请求字段不含鉴权头、`auto` 时不含 `language` 字段、模型名为空时走发现流程
- [x] 1.4 新增 `utils/qwen_response.py`：解析 `language <语种><asr_text>` 前缀，输出文本与语种（前缀不进入 `segment_text`），多片段拼接、空文本与 `None` 语种判定，并按各段语种与文本字符数统计语种占比；用单测覆盖单片段、多片段、无语音与空串四种输入，以及语种占比计算

## 2. VAD 切分与分段策略

- [x] 2.1 修改 `core/models.py`：新增独立 VAD 模型单例（`vad_model_dir` + `vad_kwargs`），仅在 v2 链路需要时懒加载；用测试确认加载后 `get_vad_model()` 非空且不影响 v1.1.8 的模型实例
- [x] 2.2 新增 `core/vad.py`：实现「音频 → VAD 段列表（毫秒）→ 过滤过短段（`min_segment_duration_s`）→ 转为秒级区间列表」，不做相邻段合并；用单测覆盖过短段过滤与段顺序保持
- [x] 2.3 实现从 VAD 段裁剪 16 kHz 单声道 WAV 的工具（复用 `utils/audio_utils.extract_audio_clip`），验证裁剪片段时长与 VAD 段时长一致且首尾对齐

## 3. v2 接口与响应封装

- [x] 3.1 新增 `api/routes/asr_v2.py` 与 `api/routes/asr_qwen_common.py`：实现「预处理 → VAD → 分段请求 Qwen → 时间轴回填 → 响应封装」主链路，**提交的语种不被 Qwen 支持时**返回 `4009`（未提交 / 空串 / `auto` 均不下发 language，不返回 `4009`）、无语音返回 `4008`、Qwen 不可用返回 `4011`、调用失败返回 `4012`；用打桩客户端的测试覆盖各错误分支
- [x] 3.2 在响应封装中复用 `utils/feature_utils` 的语速统计与 `utils/itn` 的文本规范化，保证 `text` 为规范化后分段文本按空格拼接；用测试断言 `text` 与 `segments` 文本一致
- [x] 3.3 固定分段字段结构：每个分段包含 `segment_text`、`bg`、`ed`、`speed`、`segment_words`（空数组）、`role`（`null`）、`emotion`（`null`）；顶层额外包含 `language_ratio` 字典；用测试断言字段集合与取值
- [x] 3.4 修改 `main.py` 挂载 v2 路由；用 `GET /openapi.json` 断言同时存在 `/v1.1.8/seacraft_asr` 与 `/v2/seacraft_asr`，并回归验证 v1.1.8 请求行为未变

## 4. 并发控制与排队

- [x] 4.1 新增 `core/qwen_asr_queue.py`：实现音频级处理槽位计数（`audio_concurrency`，默认 8），默认 `queue_wait_timeout_s = 0` 时第 N+1 个请求立即返回 `4013`（队列满），配置大于 0 时最多等待并在超时后返回 `4010`；用测试验证第 N+1 个请求被立即拒绝、等待超时分支与槽位释放后新请求可进入
- [x] 4.2 实现两级请求信号量：每音频一个段级信号量（`segment_concurrency`，默认 4）+ 一个全局信号量（`request_concurrency`，默认 32）；用测试验证单音频不超过 4 路、全部音频合计不超过 32 路
- [x] 4.3 把两级控制接入 v2 主链路；用并发测试验证 8 路并发下接口全部成功返回且无槽位泄漏（处理结束后信号量计数回到初值）、异常路径同样释放槽位并删除全部临时文件

## 5. 真实 role 与 emotion（默认关闭）

- [x] 5.1 实现 CAM++ 说话人链路：对每个 VAD 段提取说话人嵌入并整段音频统一聚类，输出 `spk_N`；复用 `utils/feature_utils` 推断 `teacher` / `studentN`；用测试验证同一说话人跨段标签一致
- [x] 5.2 实现 emotion2vec 情绪链路：对 1–10 s 的 VAD 段判定情绪并经 `EMOTION_LABEL_MAP` 映射，长短越界段记为「平淡」；用测试覆盖长度越界与正常段
- [x] 5.3 用 `asyncio.gather` 把说话人与情绪计算与 Qwen 转写并行执行：每个 VAD 段的转写任务与该段的 role/emotion 任务同时发起，不得等全部转写完成后再计算；由请求参数 `showSpk` / `showEmotion` 触发计算、由配置 `open_spk` && `open_emotion` 决定是否加载模型；emotion2vec 以半精度加载；用测试验证开启后 `role` / `emotion` 为真实值，未开启时为 `null`，且单路失败不影响转写结果
- [x] 5.4 把 VAD、CAM++、emotion2vec 三个模型实例按 `[model_device]` 配置落卡（同卡、默认 GPU 0），并确认不影响 v1.1.8 链路的 `device` / `ngpu` 行为

## 6. 测试与性能验证

- [x] 6.1 新增 `tests/test_qwen_v2.py`：覆盖请求参数默认值、语言透传规则（`auto` 不下发）、`language_ratio` 计算、响应结构、错误码、并发与排队行为，全部使用打桩 Qwen 客户端，不依赖真实容器
- [x] 6.2 效果测试：用 `test/audio_file/` 三个音频对比 v1.1.8 与 v2 的转写文本，逐段核对漏字、重复与时间戳偏移，产出对比记录（**对比记录见 `docs/qwen-asr-v2-test-report.md` 第 2.5 节；因两条链路切分粒度不同，字符级整体比对会塌陷，改用「30 秒时间窗字符覆盖率 + 文本相似段的时间戳偏差」作为判定口径，脚本 `scripts/compare_v118_v2.py` 已同步补充这两个指标**）
- [x] 6.3 性能测试：在目标服务器上按 1 / 2 / 4 / 8 路并发压测，记录端到端时延、排队时延与 GPU 0 算力占用（`nvidia-smi` 采样），确认 8 路是否跑满算力
- [x] 6.4 边界测试：逐档位提高到出现明显幻觉或重复为止，记录出现幻觉/重复的并发边界与典型样本
- [x] 6.5 产出测试报告（`docs/` 下），写明准确度、时间戳准度、吞吐与幻觉边界结论，作为生产并发参数的取值依据（**报告见 `docs/qwen-asr-v2-test-report.md`，第 7 节给出 `audio_concurrency` / `segment_concurrency` / `request_concurrency` 等生产取值建议**）
- [x] 6.6 role/emotion 效果测试：同一音频、同一 VAD 参数下与 v1.1.8 的 `role`/`emotion` 逐段比对，role 语义一致率目标 ≥ 90%、emotion 标签一致率目标 ≥ 80%；对 CAM++ 聚类阈值做 sweep 选定生产取值（**原计划的 0.6/0.7/0.75/0.8 实测全部严重过分割，sweep 区间实际扩展到 0.15~0.8，结论见 design 决策 13**）；用 `教师1.wav` 全量验证同一说话人跨段标签不漂移
- [x] 6.7 role/emotion 性能测试：按「A 仅转写 / B 仅说话人 / C 仅情绪 / D 两者全开」四组开关，在 1 / 4 / 8 音频并发下记录端到端 P50/P95、每段耗时、吞吐、本机 GPU SM 利用率与显存增量、Qwen 侧时延；目标 D 组相对 A 组 P50 增幅 ≤ 30%，超出时如实记录数据并给出可选处置手段供决策，不改动并行方案；验证 D 组转写文本与 A 组逐段完全一致
- [x] 6.8 role/emotion 稳定性测试：8 音频并发 + 全开连续跑 30 分钟，确认无显存持续增长与文件/槽位泄漏；注入单段音频损坏、emotion2vec 推理失败、Qwen 超时三类异常，确认 role/emotion 回退为 `null` 且转写照常返回（**实测 16 轮 128 请求全部成功，P50 77.36 s / P95 115.65 s，后半程显存增长 0 MB，临时文件数无单调增长；三类异常由 `tests/test_qwen_v2.py::FaultInjectionTests` 覆盖，见报告第 5 节**）

## 7. 部署与文档

- [x] 7.1 修改 `README.md`：补充 `/v2/seacraft_asr` 接口说明（含 `language_ratio` 字段）、`[qwen_asr]` 与 `[model_device]` 配置项含义、Qwen 服务地址配置方式（IP + 8806）与模型加载条件
- [x] 7.2 修改 `model/README.md`：说明 v2 复用的 VAD 模型与 CAM++ / emotion2vec 需求，以及 Qwen 模型由独立容器提供、不进本镜像
- [x] 7.3 新增 `httpx` 依赖并在 `asr` conda 环境验证可导入（**`requirements.txt`（宿主机 conda 环境）直接新增 `httpx`；镜像侧新增独立的 `requirements-pip-runtime.txt` 并在 Dockerfile 运行层安装，`requirements-pip.txt` 保持不变以命中既有构建缓存，见 design 决策 15**）
- [x] 7.4 构建镜像（不清理缓存），用 `http://192.168.29.12:8806` 形式的 `base_url` 启动容器，验证容器内 `/v2/seacraft_asr` 可用；测试完成后删除测试容器（**镜像 `jy-algorithm-app-asr-offline:v3.2` 构建成功，依赖层全部命中缓存；容器内 `/openapi.json` 同时暴露 v1.1.8 与 v2，v2 真实音频转写、role/emotion、错误码均验证通过，见报告第 6 节**）

## Workflow follow-up

- 验证全部通过后归档本变更，并把 delta 同步到主规格。
- 依据测试报告确定生产环境的 `segment_concurrency`、`request_concurrency` 与 `audio_concurrency` 取值。
- 后续可选增强：引入 Qwen3-ForcedAligner 获取词级时间戳（本期不做，评估结论见 design 决策 10）。
