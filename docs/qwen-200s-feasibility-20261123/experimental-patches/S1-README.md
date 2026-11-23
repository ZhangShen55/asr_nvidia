# S1供给隔离实验接线（仅隔离副本）

正式业务目录未修改。S1仅更换父进程供给，不改变段窗口16/global128、NPU所有权、KV/生成/prefill策略。

- submit仍由原client调用，但仅在内存有界FIFO登记，立即返回原concurrent Future。唯一非daemon sender负责Pipe写入，不以128个默认pool线程等待_send_lock。
- data占用至多128个队列槽；command上限3×128+16=400，cancel去重且控制消息有预留。FIFO保持submit先于同key cancel/close。
- parent active容量由有界Future identity set维护，真实Future终态callback只减一次，完成历史不参与逐段扫描；未接收拒绝不进入去重。
- cancel快速入队后仍由原client等待该段设备终态。release_request每音频结束才调用一次，等待sender发送ticket确认，健康通道错误保留原同步契约，reap/closed不掩盖原4011/4012。
- reader/sender故障及explicit abort先确认owned死亡再结束Future；正常close发送FIFO关闭、收ACK后join sender。非daemon线程不遗留。
- 仅api内_cut中的裁剪/编码调用转入独立pool，默认4线程、最多128份已获段窗口的待执行调用（QWEN200_CUT_THREADS/CAPACITY为实验env，root应记录冻结值）。ITN/增强继续原路径。取消等待实际线程结束，atexit等待pool退出。
- 状态不含正文；process公共计时把原submit包络拆成锁等待/实际写包络，S1另列队列等待、命令数、字节和cut pool累计时间。并行累计值不等于整轮墙钟。

## CPU证据

最终s1-cpu-frozen：33passed/5.02s，完整r4+candidate+support guard，真实spawn Pipe覆盖正常关闭、取消、4MiB反压、故障死亡确认、reader故障、队列控制空间、裁剪pool隔离和取消收尾。原fatal-release13项断言未修改；额外pytest plugin仅清理其人为无worker fake通道创建的sender线程，防止私有fixture缺close挂住测试。API warning4及timeout6项通过。设备/权重/cache拦截0，源码无漂移，观察到的owned子进程退出。既有Starlette deprecation仅警告，未升级依赖。

这些是正确性与CPU隔离证据，不代表NPU吞吐收益。等待root串行真实v3准入与原八完整音频对照。所有CPU工作已停止。

## 补丁

s1-supply-full.patch相对a1de07f，仅包含S1拥有的process/API/helper/tests及公共process计时。shared runtime.py/asr_process.py由root另行保留；不要重复应用process-submit-timing.patch。patch reverse-check及每个文件SHA见s1-supply-manifest.json。
