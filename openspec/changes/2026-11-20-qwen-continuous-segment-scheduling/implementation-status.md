# 实施检查点（2026-11-21）

进度20/25，change保持open，不能归档或推广。用户选择的段级恢复已实现：可信非循环前缀保留，无法恢复明确warning并继续其余段；fatal超时/设备错误不降级。root默认仍legacy48/384，候选continuous8/16/128、3ms。

独立真实v3八并发最终轮：logs/qwen-continuous-20261121-a8-drain3-clean。2/8完成（a5 389.080083s、a8 411.050558s），6条420秒删失；完成率/严格无警告率25%，两完成子集P50=400.065321s、P95=409.952034s，墙钟420.959650s，8/8目标未通过。两返回无warning；worker闸门快照有4警告（3不可恢复、1前缀恢复），不能说全轮无异常。源码冻结无变化，未并行CPU压力或额外Qwen模型。

服务normalexit0、无强制回收，TERM至parent退出54.699801s，所有登记出生身份退出、端口空闲5秒、私有PCM空、原PID1601397保留，见cleanup-verification.json。旧strict强制关闭失败记录不改写。

source真正guarded1135pass/380subtests，保护40.so151pass/17subtests，保护NPU三段一致+动态补位，source NPU语种/取消/故障验证通过。旧plainCPU fixture曾漏隔离预加载/共享stats，已修复并公开更正；与其并行的早期3ms性能不作干净对照。

两异常段保护开156/200tokens停止，配对16.61→13.09s，均不可恢复警告。896既有token回放0触发不是人工召回；98%真值缺失。完整文本仅a8双方完成可比：56匹配段文本变化+1新独有段，含明显措辞变化，见full-v3-differences-drain3-clean.json/html及-summary.md。

剩余2.5数值/质量验收、5.4同源服务归因、6.3性能门槛、6.4完整八文件质量、6.6全部必需项验收均不勾选。性能失败后停止升压，不采用/tmp有界sender原型；独立fatal-release清理修复另作源码/保护CPU增量验证。原融合结果删除状态保存在stash，生成产物不入提交。按用户要求提交推送本change，工作区保持干净。

最终fatal-release清理修复：source13pass、actual.so31pass，审计passedTrue，业务源码无漂移；修复后未另跑NPU/吞吐。严格change1/1与临时合并预览spec9/9通过，主规格未同步。
