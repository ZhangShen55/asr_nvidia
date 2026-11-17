# 910B迁移交付记录（2026-11-18）

实现已提交并推送：`2ce5c12803ef01d1ffcbf42c8e3eb6dd2b0b9ec2`（feat(NPU)）。实际开发和推理仅用xjtu910b7物理npu:7、conda asr。v3契约保留；Qwen原生BF16/单副本，CAM++与emo原生FP32，最终segment/request为1/1。

验收：最终449项CPU回归/146子断言；同八条完整WAV的g1/g2各8/8；完整长MP3；最终30扩展stage三次预热+16轮NPU稳态及资源检查通过。CPU逻辑测试不冒充NPU推理；质量无人工标签限制保留。测试自有服务、四worker和18083端口全部释放，保留既有PID1601397。原始run和B3参考副本位于logs/npu-migration-20261118，SHA见evidence-retention.json，不删除原件。

九份主规格实际同步完成，严格校验9/9；当前变更76/76任务已归档至`openspec/changes/archive/2026-11-17-npu-910b-local-qwen-asr`，随后独立chore(openspec)提交推送。CLI2保留已有日期前缀，不再重复日期；--skip-specs只跳过已执行的同步，实际9文件SHA均与审阅输出相符。详见spec-sync-protection.json与spec-sync-archive-validation.json。

全部历史归档检查为17/19：两个旧归档分别遗留7和1项未勾选任务，任务文件与迁移前HEAD逐字相同。本次归档通过且未改写旧任务；不将全历史检查描述为全部通过。

实际Docker镜像构建、容器NPU承载和上线按用户要求后置，本轮未执行。源码/配置的最终指纹与验收证据保持关联；归档提交仅同步规格、更新文档计数并留交付记录，不更改推理代码。
