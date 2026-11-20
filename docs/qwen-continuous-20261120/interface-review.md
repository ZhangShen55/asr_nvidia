# 后端接口评审与CPU验证

## 核对结论

已核对interface.md与core/qwen_continuous_process.py、core/qwen_continuous_runtime.py当前实现。LocalQwenContinuousProcessRuntime位于独立模块core.qwen_continuous_process；constructor支持max_active、max_batch_size、prefill_batch_size、online_repetition_enabled、max_batch_tokens、max_batch_audio_frames，默认startup deadline120秒。continuous模式在READY前完成模型加载；submit本身不懒加载，调用前必须ensure_loaded。

submit返回concurrent.futures.Future，重复同(request_id,segment_id)复用future；attempt_id为接纳重试审计标识，不参与有效执行唯一键。cancel/cancel_segment只申请取消，必须等待原future安全终态；release_request只能在整请求全部future结束后调用，清理父future与worker去重集合，父请求结束后不得复用原稳定ID。发送成功仅意味着消息进入私有Pipe，不等于已收到显式接纳ACK；当前只有明确capacity_rejected结果可重试，未知通信状态不能当未接收。

## 真实CPU小模型及RPC测试

命令：source CANN后，独立短TMPDIR、TORCH_DEVICE_BACKEND_AUTOLOAD=0、OMP_NUM_THREADS=1，以conda asr运行pytest tests/test_qwen_continuous_adapter.py tests/test_qwen_continuous_rpc.py -q。

结果：2 passed，8.89秒。该验证使用CPU，不执行NPU模型。

- 小型真实Qwen3文本模型两层、hidden32，独立prompt长度3/5，与左padding混长KV packing对照四步；greedy token一致，逐层K/V在atol1e-6、rtol1e-5内一致。
- CPU Pipe fake adapter验证重复ID只执行一次、短段返回时长段仍active、取消目标段、release后新父请求可重用ID、正常CLOSE及线程退出。
- 初次测试暴露F.pad参数长度问题，已修正为(0,0,left_pad,0)后通过；最初未source CANN的导入失败不算模型失败。

此CPU证据不替代真实音频前端、多模态prefill、NPU数值、容量、稳定性或真实HTTP性能验收。真实NPU实验由主Agent单独串行执行，结果以其输出为准。

## 已核对异常与资源边界

首版prefill和decode在KV clone完成后执行设备同步；当前只读视图优化见kv-view-optimization.md，仍在forward/视图状态提交后执行设备同步，再判断结束/释放/回传；取消只在owner步边界进行且同步完成。KV只留设备，序列完成/取消退出active，不用EOS成员继续占固定整批屏障。

submit Pipe发送异常经_unrecoverable_rpc确认owned进程死亡后才允许终结未完成future；close发送OSError/EOF经显式abort_owned回收并记录异常关闭，不能称正常退出。reader未知/失信响应也走owned回收。正常close可重复调用。

worker在接收前以活跃加待prefill成员的保守token估计预留预算，单段帧超限拒绝；prefill chunk同时按max(frames)*B限制聚合帧预算。实际KV bytes可在snapshot中观察，预算是估算而非HBM保证；动态packing存在额外瞬时设备内存。

语言校验在入prefill队列前逐成员执行，LocalUnsupportedLanguage与LocalModelUnavailable的跨进程类别通过_restore_error保持，不统一改成4012。在线复读/长度耗尽是ContinuousGenerationError(4012)，不直接原样重试。

## 已知限制与待实机核对

- 动态KV每步padding、cat及有效块clone有明显复制/分配成本；正确补位不等于吞吐一定优于静态批。必须真实16/128与静态配对，再决定是否进入完整HTTP升压。
- 当前每步安全边界使用设备同步，保守保证资源可复用；不能将零等待或连续补位理解为零物理开销。
- 结果消息最多每秒携带一次有界adapter快照，最终active0强制刷新；在线事件环保留最近256条。高并发时可能截断早期事件，不能把它当无损轨迹；质量测试另存token/文本，状态snapshot不含正文。
- 连续future临时携带generated_tokens用于同源质量比较；完整父请求release_request负责清除，不把其写入生产状态接口。
- 复读保护只保证检测逻辑按段隔离；真实音频误杀和节省时间须保护开/关配对验证。
- CPU测试没有模拟所有Pipe故障/重复关闭竞态，真实故障收尾和NPU正常退出仍是主Agent必需验收。

当前优化版CPU复验：5 passed，9.36秒；online_repetition_config字典透传加入后相同5项再次通过（最终时间以主Agent日志为准）。详见kv-view-optimization.md。
