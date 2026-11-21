# 段级异常恢复与复读强路径（CPU阶段）

用户确认策略调整：仅段级生成异常尝试保留结构上可恢复的原模型前缀；无法恢复标警告并为空，继续其它段。该“可保留/可信前缀”不是人工语音真值，不保证模型前文本准确，不能把折叠出的一份口头词/循环当作可信内容。

## 新在线双路径

保持最少32token、连续两次新正文增长观测确认。路径一仍需超过max(120,ceil(40×时长))异常长度、≥8循环和64循环字符。路径二绕过异常长度，但要求循环尾部至少96个非分隔字符、占整个正文非分隔字符≥95%，且循环次数≥ceil(时长×4)并满足原8次/64字符下限。新字段strong_tail_min_chars=96、strong_tail_min_ratio=95、strong_tail_repeats_per_second=4均配置化；ratio为整数百分数1～100。这是针对已发现两异常轨迹的保守候选，不是准确率最优标定。

真实8.01秒的两个故障段原静态输出都是短口头词循环，256token前没有超过约321字符的旧长度门槛。新策略的CPU前缀回放分别在重编码token150/197触发。原静态结果仅存raw_text未存token IDs，因此采用原tokenizer.json重编码，最终均256token；这是CPU重构轨迹，不等同新NPU真实生成，实际停止步与时间须由主Agent复验。证据recovery-offline-failed-replay.json。

已对原固定128真实VAD、HF四档/OM三档首稳态共896个已记录token输出重新逐token回放：896最终解码匹配，触发0/896（0%）。仅同128来源×7配置，不是896独立音频，且无人工真值，只报告触发率，不称误杀率、召回或准确率。证据recovery-guard-recorded-token-replay.json。

## 恢复接口

`utils.qwen_segment_recovery.recover_segment_output(raw_text,reason,duration_s,config=None)`仅允许repetition_stop、length_limit、invalid_output_prefix；OOM、设备死亡、取消、超时及其它原因拒绝进入此恢复分支，调用者继续原失败处理。

有有效language/asr_text且循环前存在足够非口头信息时，优先保留最后一个完整句子；没有完整句子但有较长非循环前缀时可保留前缀，始终warning标recover_prefix/recovered=true。不会把循环单元折叠一份后无条件称可信。纯循环、短口头词或无有效前缀返回language None<asr_text>、空正文，warning标discard_unrecoverable/recovered=false。原始raw由调用方仅留授权测试证据，不入在线状态或HTTP正文。

## 验证边界

CPU用例覆盖教师强调、结巴、数字字母、朗读/歌词、时长相容的长重复、强路径早停、每段隔离、正常EOS与其它异常边界、可保留完整句、切断后续半句、纯循环和短口头词丢弃。真实两故障原文离线重放均为空警告，不再冒充完整成功；未运行NPU、未测真实服务8并发，不能据此宣布时间或端到端验收完成。
