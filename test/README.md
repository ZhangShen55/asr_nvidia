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

目录模式默认只选择1条，8条必须显式 `--limit 8`。`--dry-run`（同 `--list-only`）列出sample_id、原路径字节base64及文件字节数，不发送HTTP。当前语料目录为 `test/test_audio`（旧GBK目录已由用户迁移）；建议用ASCII路径 `--audio-dir test/test_audio` 或 `--audio-dir test` 递归选择。脚本兼容旧GBK surrogateescape及当前UTF8路径，保留Path／os.fsencode往返，不重命名原音频。目录选择仅是便捷接口测试，不等同OpenSpec固定同八SHA文件的最终性能矩阵；严格A/B比较请使用完全相同的 `--audio 文件1 文件2 ...` 列表。

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


### 单Qwen八音频48／384段候选（2026-11-19）

参数已在`config.toml`与`docker/config.toml`显式列出，配置重启后生效。依用户此前选择，根配置保留1／1、物理batch1、预加载false及原小模型池行为；384只在独立候选`test/config.qwen384.toml`和`test/config.qwen384-b8.toml`启用。128测试配置仍保留。

| [qwen_asr]参数 | 384候选值 | 含义 |
| --- | ---: | --- |
| local_instance_count | 1 | 常驻Qwen权重副本数 |
| preload_on_startup | true | 权重/processor就绪后才开放HTTP，不做生成暖身 |
| segment_concurrency | 48 | 每条完整音频在途Qwen段窗口 |
| request_concurrency | 384 | 所有音频共享逻辑段窗口 |
| max_inference_batch_size | 384 | 每次物理批最多384，校验范围1～384；不开放512 |
| batch_wait_timeout_ms | 100 | 等待有限，到期发已有段，不强凑384 |
| batch_max_duration_ratio | 32 | 按最长/最短有效特征帧限制同批跨度 |
| max_batch_audio_frames | 345600 | 按每成员900padding帧×384估计 |
| max_batch_tokens | 240000 | 按(ceil(900/8)+256+256)×384估计 |
| pending_segment_limit | 512 | 独立活动PCM段数预算，旧256会先反压384供给 |
| pending_audio_bytes_limit | 268435456 | 256MiB原PCM预算，仍可能先反压；不含模型/前端/IPC临时内存 |
| audio_concurrency | 8 | A+B完整音频准入总上限 |
| audio_spk_emo_concurrency | 4或独立8 | B4默认保留；纯B八路使用-b8候选 |

八条各48为逻辑上限384；两条最多96、四条最多192，不等待凑满。B4纯B只准四条、最多192逻辑段，第5B仍4017；总8满时原4013优先级不变。四A四B也可供384，真实批仍取决于各音频VAD就绪、切片/WAV供给、长短/帧/token/PCM预算与增强汇合，不保证总是满384。VAD八CPU、CAM八process/emo八thread继续各自独立排队、batch1/预取0，不因Qwen384增加NN实例或跳过分支。

在远程项目根目录加载CANN后启动测试候选：

```bash
CONFIG_PATH=./test/config.qwen384.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py --host 127.0.0.1 --port 8083 --workers 1
# 八条纯B另用CONFIG_PATH=./test/config.qwen384-b8.toml；不会自动扩大根配置B4。
```

请求客户端仍用`test/v3_request.py --audio-dir test --limit 8 --mode A --concurrency 8`，其并发8是完整HTTP音频，不是384个HTTP请求。客户端不拥有服务；测试服务启动者须正常TERM并确认worker/端口关闭，不杀既有服务。

已有纯Qwen组件384是固定原128段各3份：中位7.415秒、同期128×3为11.389秒，峰allocated21.69GiB，27/384原始输出SHA不同，未验准确率。512首次暖身在Conv2D分配额外11.72GiB时OOM；该结论限当时共享卡/缓存条件。384在八CAM/八emo共驻和八完整音频上的稳定性、OOM、输出质量及吞吐尚未验收，不能以纯组件成绩推广正式默认。基于100ms/900帧的预算是供给/拆批门槛，不能当作HBM容量保证。
