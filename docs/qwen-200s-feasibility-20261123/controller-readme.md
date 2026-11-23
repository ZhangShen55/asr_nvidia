# 200秒实验控制器

仅存在隔离副本，正式应用代码不变。基线 `a1de07f` 的 `benchmark_asr_throughput.py` 保持不动，新 `scripts/benchmark_qwen200.py` 复用其OwnedService/真实HTTP/资源收尾逻辑。

## 文件

将B0中的以下两个文件复制到对应候选的同一路径即可，ROOT依脚本所在副本解析：

- `scripts/benchmark_qwen200.py`
- `core/qwen200_watchdog.py`

CPU目标测试 `tests/test_qwen200_controller.py` 共19项，最终完整r4+candidate+support审计在 `assets/controller-cpu-v3/`。v1失败日志保留：最初50ms合成截止包含HTTP客户端构造，导致八请求未起跑；现在共同t0在客户端建好后、八任务提交前，包含真实上传和后续完整链路。v2 18项通过，v3追加旧真实轨迹回放后19项通过。

## 准入命令示例

在对应CASE目录，先加载CANN（保留已有PYTHONPATH），所有env/CPU亲和性由root统一设置：

```sh
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/zhangs/conda-envs/asr/bin/python scripts/benchmark_qwen200.py \
  --manifest /tmp/qwen200-20261123-j9r2czoi/assets/inputs.json \
  --output /ABS/UNIQUE-RUN-DIRECTORY \
  --concurrency 8 --mode A --launch-service \
  --config /ABS/CASE_CONFIG.toml \
  --warmup-manifest /tmp/qwen200-20261123-j9r2czoi/assets/warmup-manifest.json \
  --expected-qwen-workers 1 --warmup-only \
  --url http://127.0.0.1:18083
```

M2改expected-qwen-workers为2。正式八完整文件轮去掉`--warmup-only`，其它暖身条件保持一致。八条暖身清单的path均为原短clip，upload_filename分别为八源SHA前20字符，确保各worker真实forward；`warmup-final-status.json`保存每worker暖身后基线。准入轮性能不是正式性能证据。

## 固定边界

- 默认目标200秒，整个八请求共同起点hard336秒；CLI禁止放宽200/336、改成非A或非八并发。
- 默认owned HTTP drain30秒；模型grace由原config保持120；整体关闭210秒，端口观察至少5秒。
- cached driver snapshot仅检查index7，可用HBM<8192MiB停止；过旧snapshot标未知。主机内存<16GiB、三次不同快照持续换页、Qwen/owned服务死亡亦停止。其它卡占用不触发本轮红线。
- ETA默认仅报告，禁止本轮添加`--enable-eta-stop`。旧348快照无法验证自然段/token/尾部加速证据，可信窗口0，故不能拿平均段速定论。合成测试只验证机制，不能为真实可信度背书。
- 旧真实轨迹按新资源规则于222.326秒触发HBM红线；ETA独立回放去掉NPU行仅用于检查预测，不是资源重测，hard336.590秒停。
- 结束一定finally OwnedService.close；不接管已有端口、不按名字kill共享服务。强制回收或残留缓存/端口会使complete=False。
- `target_passed`才是时间目标及正常收尾的组合判断，仍不等于质量已复核。exit0仅表示全请求完成（可高于200）或准入完成，不能单凭exit码判目标达成。
- 若未完成，all_request_latency和exact_end_to_end_speedup为null；完成子集P50/P95带n。完整八条才给all_request_latency。
- 运行前后校验业务源身份，以及控制器、输入manifest、实验配置、暖身manifest/clip SHA和正式源文件stat；任何漂移使结果无效。
