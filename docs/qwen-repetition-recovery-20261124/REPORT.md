# 异常复读恢复与配置审计（2026-11-24）

## 结果

修复continuous已确认异常尾部复读：保留原前文与一次完整循环，不再因恢复短句少于8汉字/4英文单词丢弃。例：`今天天气`+`很好`×80与整句循环，现在均保留`今天天气很好`；短英文、数字及半周期停止均有回归覆盖。仅处理已确认异常尾部，不对正常前文做通用全文折叠。按字符周期、真实分隔符及英文词界恢复，避免Listen/Within等前文词尾被吞入循环。

成功折叠仍是`completed_with_warnings`，警告`action=repetition_collapsed/recovered=true`，保留原段索引和时间边界；不等于人工核验正确。纯填充词、空正文和无效前缀仍按原异常处理；非循环length_limit的完整句恢复策略不变。正常EOS、在线检测阈值、VAD、N4/CAM4/emo1预加载、错误和资源所有权契约不变。

## 验证

- 针对性143项通过（11.78秒），含已有在线保护/常规折叠/RPC、真实continuous适配器的逐成员结束以及v3短句、原边界、警告和私有原文不外泄。
- 源码完整CPU回归：1340 passed / 379 subtests passed，2 skipped、1 deselected，pytest耗时250.32秒。2项跳过为既有保护构建专属；排除硬件模型探针文件及既有直接npu-smi探针。本次未做保护构建或NPU识别。
- CPU隔离先设置私有TMPDIR、config/log/stats路径；禁止真实torch_npu加载与模型资产IO，Transformers可选NPU包发现对CPU启动显式设为不可用；guard仍保留。测试结束无源/config漂移、无owned残留、共享cache写删尝试0。完整命令、环境和源SHA见verification.json，日志同目录保留。
- 三个差量能力、共四条修改Requirement的主规格合并预览：严格9/9通过。config.toml与85bd900逐字节一致，5个历史课程JSON的SHA全部不变。
- 前后差异明细见recovery-comparison.json（9个构造文本，不冒充真实模型输出）。

## 配置审计

当前根TOML有80个显式键。没有确认完全未读取的键；当前continuous/process模板可优先精简4项：`qwen_asr.batch_wait_timeout_ms`、`qwen_asr.batch_max_duration_ratio`仅legacy有效；两个增强`*_batch_size`只接受固定1、实际执行固定batch1。`retry_count`不控制continuous重试但仍写遥测，需要单独清理口径。queue_max_size是当前等待关闭下的条件项；queue_wait_timeout_s=0、VAD的false开关、两个v2命名增强开关和prefetch=0均有真实作用。Docker的instance_count也是部署入口有效配置。逐项证据、行号与删键回填默认值的影响见CONFIG-AUDIT.md。本次仅审计，没有删除配置或相关代码。

## 非通过轮次与修正

- target-v1在测试收集时被CPU guard拦住Transformers可选torch_npu导入，未执行目标测试。通过测试进程外CPU专用可选包发现替身解决，未改变业务依赖或放开设备保护。
- target-v2为134通过/2失败，发现带冒号的英文前文周期旋转截词；v3修正后136通过。
- target-v4为141通过/2失败，新增完整前文词与循环相同的冒号边界例再次暴露相位选择；最终按首单元内部布局和完整词界修正，v5为最终143全部通过。
- full-v1在新增边界发现后主动中断；该轮另有CPU启动未将tests加入搜索路径造成的新既有夹具导入错误，退出-2、无owned残留，不计通过。最终full-v2使用明确源码/tests路径完整通过。上述原始日志保留在对应/tmp/repetition-*目录，不覆盖失败记录。

## 适用范围与部署

原2.02秒异常段只有警告元数据、没有原始生成正文，本次不能证明它属于哪一种恢复拒绝，也不能宣称已找回该段。没有人工真值或完整课程重测，不能报告真实转写准确率/召回率、WER或性能改进。单凭循环文本也无法无歧义分辨前文恰好与周期相同的语义边界；保留模型文本和警告，不伪造确定性。受保护构建和镜像在下一次构建纳入；源码部署重启后生效，本次未启动或重启服务。
