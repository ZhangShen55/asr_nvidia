# 无效VAD配置清理

日期：2026-11-19。服务器 xjtu910b7，项目 /home/zhangs/asr_nvidia，conda asr。

## 删除内容与证据

删除 vad_max_single_segment_time_ms、vad_max_end_silence_time_ms、vad_speech_noise_thres、vad_decibel_thres，移除Settings的四项正名/四项qwen别名和VAD池自动nested转发。SNR仍是有效段级参数，只清理其无效nested转发。三个标定/验证脚本不再引用已删属性。

NativeVadRunner由模型原始model_conf构造FsmnVADStreaming，nested vad_kwargs只放入inference options；FunASR AutoModel构造合并model_conf与顶层kwargs，FSMN init_cache读取顶层max_end_silence_time，不读取这些nested阈值。既有真实选项证据为8000ms/800ms/0.6/-100/-100，见原classroom-threshold-review.md，原模型文件不改。

六份当前模板同步删除死键；复读五项从误放的vad表移到实际读取的qwen_asr表。旧误放值与getter有效缺省一致，功能保留。instance_count被docker/start.sh使用；model用于显示；能量过滤阈值、窗口、开关虽当前可关闭但真实代码使用，均保留。根配置已明确采用384，容器维持旧配置；修正之前同步根配置后未更新的测试断言，不改容器值。

## 验证

- 与Git原版Settings/各配置逐项对比：六份配置各85个保留property全部相同，见dead-vad-config-properties-20261119.json。
- 三个更新脚本AST语法通过，运行源码/脚本无已删除属性引用；git diff --check与OpenSpec严格校验通过。
- 首轮CPU：1021通过、3失败、2保护专属skip、1硬件探针排除、280子测试通过，238.50秒。失败为两个根配置旧默认断言及一个旧脚本属性引用；均已修正。原始证据保留在/tmp/asr-throughput-20261118-xvior99k/source-full-cpu-dead-config-20261119/。
- 最终完整CPU：1023 passed, 2 skipped, 1 deselected, 2 warnings, 281 subtests passed in 236.68s (0:03:56)。自有PID/出生身份全部退出、源码与profile冻结一致、共享cache写删尝试0；证据目录为/tmp/asr-throughput-20261118-xvior99k/source-full-cpu-dead-config-20261119-r2，精简结果见dead-vad-config-cpu-result-20261119.json。

完整CPU使用既有r4设备/权重guard及共享cache写删guard，导入前设置私有短TMPDIR；使用既有独立CPU profile，根配置另做属性等价验证。测试拥有者按PID/出生身份记录和核对；未启动真实模型或占用NPU，未重启既有服务，未修改fusion_result.json已有变更。旧保护产物仅做隔离预检，不作为本次新保护构建验收。未宣称新的NPU/八音频/准确率/吞吐结论。
