# 增量复读组件与输入重放验证

2026-11-20，本项仅CPU，不加载模型、不启动NPU或服务。

## 接口与阈值

`IncrementalRepetitionGuard(duration_s, config=None)`每段独立实例；对每个新token累计解码结果调用`update(raw_text, generated_tokens=N)`。返回None或counts-only的`RepetitionDecision(reason="repetition_stop")`；只发停止建议，不修改正文、不负责资源释放或错误映射。runtime必须只停止该段，在设备安全边界清理KV并按4012传播，不把异常停止当EOS。

正文仅取`<asr_text>`后内容，语言前缀不参与。默认：至少32个生成token，正文超过max(120, ceil(40×音频秒数))，最多512字符尾窗内存在至少8次、合计至少64字符的连续重复，连续2次正文增长观测确认。扫描unit长度1～32，只在检测副本去掉分隔标点/空白，不改变原文，不做大小写或数字归一化。原40字符/秒与120绝对下限沿用现有后处理的保守异常长度经验；8次/64字符/两次增长是本轮候选合成保护阈值，尚非真实循环语料标定结论。长期重复但未超过异常长度可能不触发，需在真实语料评估召回与误杀，不能把本单元测试说成质量验收通过。

## 已执行

命令（独立短TMPDIR）：`python -m pytest tests/test_incremental_repetition.py tests/test_compare_segment_texts.py tests/test_replay_qwen_segments.py -q`，结果 **30 passed in 0.31s**。

其中增量检测24项：中英教师强调、结巴、数字/字母串、歌词朗读/教学重复保护；4种合成无限循环提前触发；前缀不误判、相同回调不累积确认、每序列状态隔离、关闭模式与参数校验。全部为构造文本，不是新录制音频或真实生成故障样本。

差异工具4项：按来源/段ID匹配、不按列表顺序；缺失及重复ID明确报出；保留大小写/标点差异；HTML转义防止正文变为标签。工具`scripts/compare_segment_texts.py BASE.json CAND.json --output REPORT`产出JSON与可搜索HTML，支持原始文本、tokens和finish_reason差异。原始记录含来源、边界或修复字段时全部保留；输入仅为授权测试产物，不接在线状态正文日志。

重放工具2项：固定PCM输入完整性和生成参数、损坏SHA/缺失选择拒绝。实际执行`replay_qwen_segments.py`对bench128-20261120的manifest.json与samples.npz校验 **128/128**；float32/mono、有限值、每段SHA、样本数和时长一致。产物replay128.json固定语种None、greedy、max_new_tokens256、EOS151643/151645和独立FP32前端契约，供主Agent真实模型配对使用。

## 已记录真实token的CPU回放

对既有bench128真实VAD结果的首稳态轮，HF batch16/32/64/128及OM batch16/32/64共896条记录，使用原始tokenizer.json逐token前缀解码，EOS前依次送入新检测器；全部896条最终解码与原记录raw_text逐字相同，检测器触发0次。结构化证据为repetition-recorded-token-replay.json。这是既有生成轨迹的离线CPU重放，不是重新运行模型、独立896段或人工标注准确率；其来源只有同128个段的七种配置。样本中没有已标注异常循环，不能据此评估复读召回，也没有设备耗时收益结论。

首轮尝试用Transformers AutoTokenizer因未source CANN触发torch_npu自动导入失败（libhccl.so）；该进程退出，无NPU初始化。改用tokenizers直接读取同一tokenizer.json后完成回放，未修改环境或依赖。

## 未完成

未执行真实生成token逐步关/开复读保护配对、真实复读召回或听音复核，未证明耗时收益。未执行本change完整服务8并发。历史慢批的逐成员PCM/提示映射本工具没有恢复，replay128.json明确标记为固定新同源输入，不宣称历史根因闭环。task4.1仅组件/合成保护部分完成，不能替代后续真实质量与时间验收。
