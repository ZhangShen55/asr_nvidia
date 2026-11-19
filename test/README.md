# v3接口请求测试

`v3_request.py` 是类似Postman的独立HTTP客户端，只请求你已经启动的服务，不加载模型、不启动或终止服务。运行目录为项目根 `/home/zhangs/asr_nvidia`，解释器使用 conda asr。

单文件A类（纯转写）：

```bash
/home/zhangs/conda-envs/asr/bin/python test/v3_request.py \
  --audio '/完整/音频路径.wav' --url http://127.0.0.1:8083 --mode A
```

目录递归选择前八条wav/mp3、八路完整音频请求（A8基线）：

```bash
/home/zhangs/conda-envs/asr/bin/python test/v3_request.py \
  --audio-dir test --limit 8 --mode A --concurrency 8 --dry-run
# 核对计划后去掉 --dry-run 才发送请求
```

目录模式默认只选择1条，8条必须显式 `--limit 8`。`--dry-run`（同 `--list-only`）列出sample_id、原路径字节base64及文件字节数，不发送HTTP。历史音频目录采用GBK文件名字节，建议用ASCII路径 `--audio-dir test` 递归选择，或在终端用Tab补全实际路径；脚本保留Path／os.fsencode往返，不重命名原音频。目录选择仅是便捷接口测试，不等同OpenSpec固定同八SHA文件的最终性能矩阵；严格A/B比较请使用完全相同的 `--audio 文件1 文件2 ...` 列表。

B类：`--mode B` 同时开启 `showSpk` 和 `showEmotion`。单开可用 `--mode A --show-spk` 或 `--mode A --show-emotion`；`showRoleIdentify` 固定false。可用 `--language auto`、重复 `--hot-word 词语`、`--word-timestamps`、`--show-speed`。B8需要服务端独立B8配置；默认总8／B4下第5个B可能返回4017，不能把拒绝当速度收益。

`--concurrency` 为完整音频请求数，仅1／2／4／8，**不是Qwen语音段并发128**；Qwen每音频窗口／全局128与实际物理批大小由服务端config.toml决定。请求完成立即补下一条，不按整波等待；不自动重试业务失败。

输出默认放在ignored `logs/v3-request-时间戳/`，每条记录HTTP状态、业务码、段数、墙钟和原路径，summary保留失败／取消／未提交项。控制台不显示转写正文，默认不存原响应；需要像Postman查看原响应时加 `--save-responses`，生成 `response-NNNN.body`（通常为JSON），仅存logs且勿提交正文。`--output` 必须指定logs下不存在的新目录，避免覆盖证据。

脚本不自动暖身。计时包含实际上传、服务排队、首次懒加载／首次forward编译以及处理和响应读回，不应把首轮当成已预热的稳态数据。服务预加载权重也不等于所有输入shape已经暖身。

`--timeout` 默认420秒（最大420），包含上传到响应完整读回的单请求墙钟。超时立即停止追加、取消其他HTTP请求、有界获取 `/get_status` 诊断，退出码2；全成功退出0，Ctrl+C退出130。客户端取消不证明服务端NPU工作已结束。该工具不拥有既有服务，无法也不会关闭它；测试启动者仍须向自己的服务父进程发TERM，等待模型工作／子进程收尾并观察端口释放至少5秒。若需要自动启动与关闭owned服务、120秒无进展诊断和完整矩阵，请使用 `scripts/benchmark_asr_throughput.py`，不要用本便捷客户端替代最终验收。

Postman中相同请求为：`POST http://127.0.0.1:8083/v3/seacraft_asr`，Body选择form-data，`audioFile`类型File，其余字段Text：`language=auto`、`showSpk=false`、`showEmotion=false`、`showRoleIdentify=false`、`wordTimestamps=false`。让Postman自动生成multipart Content-Type和boundary。

## 使用单Qwen预加载128候选配置

保持项目根 `config.toml` 可继续使用原兼容默认。需要本轮候选时，服务启动者在服务器项目根执行：

```bash
cd /home/zhangs/asr_nvidia
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CONFIG_PATH=./test/config.qwen128.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py \
  --host 127.0.0.1 --port 8083 --workers 1
```

该profile为一个Qwen实例、每音频16／全局128逻辑段窗口、物理批上限128；总完整音频8／B子上限4。完整B8实验改用 `CONFIG_PATH=./test/config.qwen128-b8.toml`。两份配置仍是待完整性能验收的候选，不保证128是最优拐点。启动预加载只加载权重和processor，不执行generate，不等于真实输入shape预热完成。服务启动者测试后负责TERM和完整收尾；本请求脚本不改变该所有权。

## 本机Postman经SSH访问

Tunnelblick已连接时，在本机终端建立隧道：

```bash
ssh -L 18083:127.0.0.1:8083 -N xjtu910b7
```

本机Postman请求 `http://127.0.0.1:18083/v3/seacraft_asr`。`xjtu910b7`按本机SSH配置经跳板连接；仅Tunnelblick未连接时改用 `xjtu910b7-tailscale` 别名。以上仅为使用示例，不自动建立监听。完成后退出自己建立的隧道；退出隧道不会关闭服务器上的ASR服务。
