# 昇腾 910B v3 交付

本目录准备 CANN 8.1.RC1、Linux aarch64、Python 3.10 的候选交付文件。文件检查、环境metadata、镜像构建、真实NPU前向与上线是不同证据等级。当前迁移要求先把完整服务跑稳，之后再实际构建与上线；本文的命令不能视为已经执行。

## 环境和依赖

宿主开发使用 `/home/zhangs/conda-envs/asr/bin/python`、物理 `npu:7`。不要原地升级现有环境，也不要把其site-packages通过PYTHONPATH叠加到系统Python。

- `requirements-pip.txt`：通用业务依赖；`requirements-pip-runtime.txt`：进程/进度依赖。
- `requirements-npu.txt`：torch/torch-npu/torchaudio 2.5.1、transformers 5.13.0等目标版本。emotion2vec的NPU实例使用该torch-npu提供的GELU V2 `npu_gelu(..., approximate="none")`，保持CPU原始GELU/erf语义；目标wheel必须保留此算子schema，不能静默回退默认tanh近似。该适配不新增pip依赖；HF32设置与模型数值一致性仍需真实设备验证。
- `requirements-pip-extra.txt`：ITN；先由conda安装av=14.2.0、pynini=2.1.7。
- `requirements-dev.txt`：测试客户端和pytest；`requirements-build.txt`：独立构建环境工具。

`requirements-conda-linux-aarch64.txt`和`requirements-lock-npu.txt`记录本轮参考环境的完整conda/pip来源；`environment-reference.json`保存pip元数据基线。rebuild脚本使用显式conda清单与pip-owned锁，避免接管conda的av/pynini，并严格比较pip list。

创建不存在的独立前缀（会安装软件）：

```bash
ASR_CONDA_EXE=/data/anaconda3/bin/conda ASR_WITH_DEV_TOOLS=1 ASR_WITH_BUILD_TOOLS=1 \
  ./scripts/rebuild_npu_env.sh /tmp/asr-build-env
```

脚本拒绝覆盖已有前缀，记录pip/conda快照、包差异和import结果；Cython/conda-pack仅安装于独立构建前缀。解析版本并不表示NPU前向通过。源运行前需要CANN共享库和编译器Python模块路径；source后不能用项目路径覆盖PYTHONPATH：

```bash
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
/home/zhangs/conda-envs/asr/bin/python scripts/check_npu_environment.py --config config.toml
```

`--metadata-only`不初始化NPU，不能作为硬件验收。CANN路径可用`ASR_CANN_ENV`覆盖，但必须存在。未经验证的硬件/运行时不宣称兼容。

## 镜像构建（服务稳定后）

默认底座为构建机已有的 `jy-algorithm-app-asr-ascend_cann8.1.rc1-ub2204py310:v1.1.9_260709`。允许`BASE_IMAGE`指定已验证的等价CANN底座，`--pull=false`不会升级或清除本机镜像。该历史镜像的系统Python可能有旧包，因此运行层仅用独立环境解释器。

```bash
ASR_ENV_PREFIX=/tmp/asr-build-env \
ASR_IMAGE=asr_offline:v3_20261117120000 ./docker/build.sh
```

build.sh使用临时staging，校验并复制VAD、CAM++、emotion2vec原始权重/配置，排除旧engine目录；不修改`model/`，不把样本、报告或Qwen大模型打进镜像。Qwen由运行时只读挂载提供。环境用conda-pack归档，解包至`/opt/asr`并执行conda-unpack；构建/运行Python ABI和目标aarch64必须匹配。也可通过`ASR_ENV_ARCHIVE`传入已制备的归档，归档需含`bin/python`和`bin/conda-unpack`。

缺少conda-pack或锁定的Cython会明确失败，不自动pip升级。Cython只在带`.asr-build-stage`标记的临时工作区运行，保留构建metadata，并在不自动初始化设备的构建层做扩展导入检查；这不是NPU验收。不能把macOS或不同Python ABI的环境归档用于此镜像。

## 启动入口与设备编号

Compose配置的是宿主绑定：`--privileged --runtime=ascend --ipc=host`的等价设置和`ASCEND_VISIBLE_DEVICES=7`。只映射物理7号卡时，容器内通常为`npu:0`，须按实际可见设备核对；不能直接在容器配置写宿主`npu:7`。

```bash
cd docker
export ASR_IMAGE=asr_offline:v3_20261117120000
export ASR_CONFIG=./config.toml
export QWEN_MODEL_DIR=/home/model/Qwen3-ASR-1.7B-hf
export ASCEND_VISIBLE_DEVICES=7
export ASR_PORT=9000
export ASR_CONTAINER_NAME=asr-npu-v3-test
export ASR_COMPOSE_PROJECT=asr-npu-v3-test
docker compose -p "$ASR_COMPOSE_PROJECT" up -d
curl --fail http://127.0.0.1:9000/get_status
```

三个小模型从`/app/model/`加载；Qwen挂载为`/app/qwen-model`，不依赖外部HTTP模型服务。`CONFIG_PATH=/config.toml`区分配置挂载与模型路径。`start.sh`是容器内entrypoint，负责source CANN、使用`/opt/asr/bin/python main.py --host 127.0.0.1 --port <端口> --workers 1`、预检和nginx；不能在其中添加`docker run`参数冒充设备绑定。

当前默认采用原生运行器、`instance_count=1`、Qwen `local_instance_count=1`，`segment_concurrency=1`、`request_concurrency=1`，容器设备保持`npu:0`。宿主物理7号共享卡以相同12秒片段、每档2轮4B+4A共16个测量请求标定：

| Qwen副本 | segment/request门限 | P50（秒） | P95（秒） |
|---:|:---:|---:|---:|
| 1 | 1/1 | 20.0201 | 23.5629 |
| 1 | 2/2 | 14.1411 | 23.7260 |
| 2 | 2/2 | 14.4533 | 22.8981 |

完整课堂对照选择单副本与1/1门限，最终数据与原始SHA见主迁移报告；上面的短音频三档保留作历史参考。每增加一个完整服务进程也会复制Qwen及增强模型池。上述是宿主原生FP32/BF16实际路径的重复容量工作负载，不是8份独立质量样本，也不能替代容器实际承载验收；镜像构建和容器NPU实测仍按后置流程执行。容器SNR=18.0及3/3过滤窗口对齐本轮现行宿主业务配置，非新增调参。

## API与检查

仅提供`POST /v3/seacraft_asr`、`GET /get_status`、`DELETE /clear_tasks_list`；旧路由不恢复。保持v3请求默认值、错误码、字段类型与`gpu_time_ms`原计时口径。状态查询不触发Qwen或增强模型懒加载，VAD按启动契约预热。

健康检查只表示服务就绪，不能代替转写、聚类、情绪或错误路径验收。完整测试矩阵和质量证据见本次OpenSpec change中的`test-plan.md`。

## 测试结束必须停止

Compose默认`restart: "no"`、`init: true`。在测试成功、失败或取消后的统一收尾中执行本次项目的关闭，并核验进程和监听端口释放：

```bash
docker compose -p "$ASR_COMPOSE_PROJECT" down
```

不要删除模型卷、镜像、缓存或日志；不要按端口或进程名终止服务器已有uvicorn。入口以独立进程组管理自己创建的进程，先向自有`main.py`服务父进程发TERM，让Uvicorn先通过lifespan关闭VAD与模型池；统一入口把随后重放的TERM转为Python SystemExit，从而执行NPU运行时atexit。NPU初始化后INT可能失效，正式入口不能换回`python -m uvicorn main:app`。nginx master接收QUIT。默认150秒宽限后才向未退出的自有组发TERM，再等10秒后KILL；entrypoint以0.2秒有界轮询监督自有子进程，关闭时逐PID收集退出码；非零退出、无法归属的127或超时均记失败。外部TERM且全部正常收尾时返回0，INT保留130且不会被后续TERM覆盖。Compose给予180秒停止时间。重启策略不得在测试后留下后台接口。TIME_WAIT不等于仍在LISTEN，原有服务继续占用HBM也不属于本次泄漏。保留日志、配置、样本哈希、退出与端口检查证据，关闭失败不宣称整批验收完成。
