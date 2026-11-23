# 关闭HBM停止线：六实例六音频正式轮

六个完整音频全部返回HTTP200，总墙钟194.809秒；处理完成率6/6，P50=169.546秒、P95=193.336秒。无warning严格成功3/6，另外3条明确带warning，不能称无损6/6。

|音频|耗时秒|状态|warning/恢复/不可恢复|输出段/非空段|文本字符|
|---|---:|---|---|---|---:|
|a1|157.574|complete|0/0/0|728/728|20553|
|a2|174.377|completed_with_warnings|4/2/2|569/569|18824|
|a3|194.531|completed_with_warnings|1/1/0|659/659|30784|
|a4|189.750|completed_with_warnings|11/0/11|614/614|22947|
|a5|115.836|complete|0/0/0|431/431|12017|
|a6|164.714|complete|0/0/0|512/512|16059|

16个warning原因：{'length_limit': 1, 'repetition_stop': 15}；3段恢复可信前缀，13段明确丢弃不可恢复文本（a2两段、a4十一段）。所有六条输出全文非空；这并未验证识别准确率。warning完整边界、reason、action保存在JSON。

正式状态流共140个样本，最低已观测卡7free=6323MiB，发生在起跑后194.811秒；低于8192共有47次观测，首次123.854秒、最后194.811秒，期间继续执行并最终6/6完成。是缓存采样最低值，不是真实连续最低值。

存储状态的failed_instances最大0、execution_failures最大0、死亡worker=[]，连续推理error计数={}；service.log命中推理OOM异常签名0处。未观察到推理OOM或worker失败，文本重复/长度warning单独统计。

实际run flag=True、首次startup资源记录hbm_free_stop_enabled=False/effective阈值None；记录的业务文件/配置SHA与基线均匹配=True，本轮source_changed=False。6worker×1线程、6音频×16段/global96及原预算未变。

唯一完整配对a5：上一轮431段/12024字符，本轮431段/12017字符；边界匹配431段、文本变化32段、仅字段变化0段、旧独有0段/新独有0段。完整差异：nohbm-a5-all-text-differences.json。其他五条上一轮未完成，差异为未知，绝不计零。

- 0.00–1.16秒：旧「Last session, I'm for this.」；新「Ask the session off today.」。
- 195.10–201.66秒：旧「充钱的，人，，人民币玩家，对你就可以在里面，，进入了之后呢，。」；新「充钱的，，人民币玩家，对你就可以在里面，，进入了之后呢，。」。
- 299.00–300.40秒：旧「你们自己的事情。」；新「你们自己的视频。」。
- 321.93–329.94秒：旧「是用 Python 组这个这个部分组成的。然后呢，这个它的好处就是它可以可可视化，就是你运行了一个代码之后呢，它」；新「是用 Python 组这个这个部分组成的，然后呢，这个它的好处就是它可以可可视化，就是你运行了一个代码之后呢，它。」。
- 346.48–354.49秒：旧「function，describe that word，我们可以看一个视频，这个是一个，James professor。」；新「function，describe that word，我们可以看一个视频，这个是一个，James Professor。」。
- 466.83–470.51秒：旧「得到它对应的一个结果。我们看这样一个例子。」；新「得到它对应的一个结果，我们看这样一个例子。」。

文本变化可能来自运行时组批/数值/自然输出差异；未有参考稿，不能判定哪版正确。关闭资源线本身不改变生成算法。

收尾当前记录：{'exit_code': 0, 'forced': False, 'normal_exit0': True, 'remaining_pids': [], 'remaining_cache_files': [], 'term_to_parent_exit_s': 26.044364910572767, 'passed': True}。root独立负责最终资源清理审计。

结论仅适用于本轮六音频；不能推断八音频200秒达标或声称完整配对提速。
