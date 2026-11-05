# Tasks

## 1. 配置结构与 VAD worker

- [x] 1.1 在 `core/config.py` 增加 `[vad]` 的唯一读取入口，迁移 `vad_device`、切分参数、全局 decibel/SNR 参数、三个段级 enable、阈值、抽帧比例和 `after_start/before_end`；实现类型与范围校验，并验证旧 `[qwen_asr]` VAD 键不会覆盖 `[vad]`。
- [x] 1.2 更新 `config.toml`、`README.md` 或部署配置文档，写入最终 `[vad]` 默认值和简要中文注释；验证示例可被 TOML 解析，且 `speaker_similarity_threshold` 仍位于 `[qwen_asr]`。
- [x] 1.3 修改 `core/vad_pool.py`、状态输出和相关调用方，使 VAD kwargs 与 worker 设备数组全部来自 `[vad]`；运行 VAD 池配置测试，验证 `vad_device=["cpu","cpu","cuda:0","cuda:0"]` 启动 4 个固定 worker，旧动态池参数不改变数量。
- [x] 1.4 更新配置契约和 VAD worker 单元测试，覆盖默认值、空数组、非法设备、`-1/0/3` 范围值、抽帧比例边界和 enable 类型；验证 `python -m pytest tests/test_config_contract.py tests/test_vad_pool.py` 通过。

## 2. 段级统计与过滤

- [x] 2.1 扩展 `core/vad_volume.py` 的段级统计，支持确定性 `vad_decibel_frame_sample_ratio` 抽帧并返回抽样帧数、低能量帧比例、RMS dBFS 和 SNR；使用 NumPy 向量化，验证全帧/半帧、短于 10 ms 段和零分母边界单元测试通过。
- [x] 2.2 在 `core/vad.py` 实现“短段→时间范围→段级规则”的固定顺序；按完整音频前后范围逐段判断，组合 decibel/SNR/RMS 三个 enable，任一启用规则命中即删除；验证不跨段合并、范围外段保留、`0/0` 全音频生效、`-1/-1` 不按位置限制。
- [x] 2.3 更新 VAD 统计与日志，区分短段、范围内候选、decibel/SNR/RMS 各类删除、合计删除和耗时；验证日志不包含音频内容，且 `get_vad_stats()` 数量与保留段列表一致。
- [x] 2.4 为段级过滤补充单元测试，覆盖低能量比例 `>=0.8`、比例低于阈值、RMS 或 SNR 任一命中、三个 enable 独立关闭、阈值 `<=-100` 短路和 `vad_snr_thres=-18.0` 的宽松行为；验证 `python -m pytest tests/test_vad_volume.py tests/test_vad.py`（或对应新增测试文件）通过。

## 3. Qwen 链路集成与回归

- [x] 3.1 确认 `/v3/seacraft_asr` 在音频张量缺失的兼容调用中仍能按约定取得段级统计，或返回明确处理错误；验证正常请求的段级过滤发生在 Qwen 请求调度前。
- [x] 3.2 更新 Qwen 链路测试，构造被 decibel/SNR/RMS 删除的候选段，验证这些段不创建 Qwen、CAM++ 或 emotion2vec 任务，过滤后的有效段数量作为进度和请求统计分母。
- [x] 3.3 在 `test/audio_file/音频文件` 选取前后静音、正常语音和短促噪声样本，分别运行默认 `after_start=3/before_end=3`、全音频 `0/0` 和关闭范围 `-1/-1`；保存原始段、过滤段、删除原因、Qwen 请求数和耗时对比报告。
- [x] 3.4 执行完整回归测试与配置解析检查，验证现有 Qwen 段时间戳、请求并发、CAM++/emotion2vec 模型池和响应协议未因过滤配置迁移而改变；记录失败样本和可回滚配置。

## Workflow follow-up

- 评审测试报告和默认阈值后，再运行 `$openspec-apply-change` 实施本 change。
- 实施并验收完成后，按项目流程归档该 change；本 change 不包含 FireRedVAD 后端替换。
