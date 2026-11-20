# 段级接口与所有权

LocalQwenContinuousProcessRuntime继承原process兼容加载/关闭/snapshot接口。submit(audio_bytes, language, request_id=..., segment_id=...)返回concurrent.futures.Future；cancel(request_id,segment_id)只申请取消。上游以asyncio.wrap_future异步等待，每段RESULT独立返回，不能持legacy独占租约等待整段。稳定request_id/segment_id关联接收确认、取消与结果，attempt用于拒绝重试审计；重复已接收段复用同一任务，不重做forward。

单worker一份权重，设备step串行；新prefill在安全边界插入，独立KV/position/mask按段归属，完成后实际读取/清理才回传结果并只归还一次credit。client负责逻辑16/128，worker负责实际max_active/max_batch_size及内存预算；拒绝仅在明确未接收时允许容量重试。生产快照有界、不存正文、不等待忙RPC。

异常EOS之外的repetition_stop与length_limit按4012传播；同音频其它取消需记录为连带取消，不重复计故障。模型整体busy不等于128序列全部占满。后端Agent A负责runtime/process，Agent B负责client/queue/scheduler/pipeline，Agent C独立复读组件；配置、集成与NPU由主Agent负责。

评审记录：A后端见interface-review.md；B调度见../qwen-continuous-scheduler-contract.md；C复读模块已完成接口对接与30项CPU验证，真实896条已记录token回放无触发。
