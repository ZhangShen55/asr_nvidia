# Persistent decode bundle draft

此目录是独立CPU候选，不修改项目运行源码。bundle.patch仅修改adapter和其CPU测试，NPU/HTTP由主Agent独立验证后再决定采纳。

同keys序列集合和顺序不变时，复用整批DynamicCache及有效attention_mask，追加一列有效mask，逻辑position仍由各段有效KV长度决定。成员结束、取消、插入或重排时，按旧bundle和新prefill组进行整组index_select、宽度padding及cat，必要时一次行重排；不逐段pad。prefill新段保持独立组，旧组上的只读valid slice用于归属和测试。

HF DynamicLayer以cat更新keys/values，不会原位改写旧KV行；稳定bundle的cache对象由单设备owner推进。成员变化时从每段最新group获取cache/row/mask，不使用陈旧的逻辑slice替代新KV。左padding继续由mask置0；position_ids按真实有效长度计算，物理cache宽度不能当logical position。pending token仍是不在当前KV中的最后采样token。

新增cache_reuses/repacks有界计数用于确认真实复用比例。代际存储的释放由原有StorageImpl弱引用测试覆盖；新测试覆盖稳定reuse、插入短新成员、跨组行重排、取消成员、再次reuse，以及每步相对独立模型token/各层KV对齐。

CPU测试命令：source CANN；cd /tmp/qc-bundle-draft；PYTHONPATH=/tmp/qc-bundle-draft:/home/zhangs/asr_nvidia:$PYTHONPATH TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 conda-asr-python -m pytest test_qwen_continuous_adapter.py -q。

不承诺吞吐收益或消除全部开销：频繁结束/新prefill仍会repack，序列Python管理/输出detokenize/安全同步仍存在；真实单卡16/128及8HTTP必须再次测。

补充：成员变化时每组trim到所选成员最大有效KV长度，mask同步取右端，避免跨代沿用旧physical width无界增长。30代持续插入/取消CPU测试中物理width始终≤8，且token/逐层KV对独立参考保持一致。
