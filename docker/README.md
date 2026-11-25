# 昇腾 910B Cython 保护镜像部署

本版为单 HTTP 服务，内部默认 Qwen4 / CAM++4 process / emotion2vec1，Qwen 每音频窗口16，总音频4 / 全局逻辑段64，VAD8 CPU。每个容器直接启动一个 HTTP 服务。Docker 构建在独立阶段编译业务模块，运行镜像交付 `.so`。增强设备数组非空开启、每元素一实例，`[]`关闭；旧 `[features]` 开关不可继续使用。`instance_count` 只能为1，模型数量配置在 `[qwen_asr].local_instance_count`。

本目录仅保留 Dockerfile、docker-compose.yml、config.toml、README.md。配置修改需重启；首次加载和真实语音暖身分开，HTTP就绪不等于已经执行前向。旧源码镜像的 B4 结果记录在 `../docs/docker-source-cann82-20261125/`。新保护镜像的验证见 `../docs/docker-cython-cann82-20261125/`。

## 构建前置条件

- Linux arm64 / 910B，Docker27及Compose2、Ascend Docker Runtime，驱动与CANN8.2 RC1配套。物理卡默认7；`ASCEND_VISIBLE_DEVICES`选择runtime目标设备，`ASCEND_RT_VISIBLE_DEVICES`进一步约束进程可见设备，单张物理7确认映射后配置逻辑npu:0。
- 本机底座 `cann:8.2.rc1-910b-ubuntu22.04-py3.10`；本轮核验ID为 `sha256:634118c33bcc879ee709510a532f3e8f90cb0d6a884922b5996367f9a6a7d864`。先核对实际toolkit/kernel及架构，禁止从公网拉同名镜像冒充本机底座。
- 容器运行环境为独立Python3.10、torch2.5.1、torch-npu2.5.1.post1；其余业务依赖按容器专用锁。宿主asr的torch-npu2.5.1/CANN8.1环境不升级。精确版本由容器专用入口检查，不只比较主次版本。
- 三个小模型原资产在仓库 `model/`，构建仅取必要权重/配置；Qwen原始HF权重不入镜像，运行时另行只读挂载。
- Buildx插件和独立docker-container builder。独立builder看不到daemon已有base，需要任务私有loopback registry传输并核对base config digest；仅该registry允许本地HTTP，不修改全局daemon设置、不对外发布。

## 准备最小构建材料

所有材料在本目录外，禁止打入测试、文档、项目脚本、样本、Git/OpenSpec数据和旧业务扩展。主上下文由根 `.dockerignore` 白名单控制，其中精确放行两项编译入口。环境、模型和 Cython 分走命名上下文。

```bash
cd /home/zhangs/asr_nvidia
export WORK_DIR=/home/zhangs/asr-docker-build-$(date +%Y%m%d-%H%M%S)
mkdir -p "$WORK_DIR"
/home/zhangs/conda-envs/asr/bin/python scripts/prepare_container_context.py \
  --workspace "$WORK_DIR" \
  --tool-python /tmp/asr-rebuild-clean-build-20261117-qr9yhpvu/bin/python
```

工具环境提供conda-pack0.8.1，只打包运行前缀，不把Cython/conda-pack工具环境作为运行环境。脚本生成 `runtime_env/`、`model_assets/` 和输入摘要；缺依赖或资产必须先定位，不用全仓库COPY兜底。准备脚本仅在宿主运行，不进入镜像。其它机器请将解释器、原资产位置及工具路径改为已核验的对应位置。

保护构建使用 Cython 3.0.12 的官方源码包。在独立构建阶段运行 Cython 前端，用镜像的 Python 3.10 和原生 GCC 编译扩展。

```bash
mkdir -p "$WORK_DIR/cython_tools"
curl --fail --location --retry 3 \
  https://files.pythonhosted.org/packages/5a/25/886e197c97a4b8e254173002cdc141441e878ff29aaa7d9ba560cd6e4866/cython-3.0.12.tar.gz \
  -o "$WORK_DIR/cython_tools/cython-3.0.12.tar.gz"
```

Dockerfile 核对源码包 SHA256。Cython、生成的 C 和构建产物只在独立阶段处理，最终复制业务扩展和包入口，保留 `main.py`、`model_status_runtime.py`、`telemetry_runtime.py`。容器运行时标识 `cann82-source` 沿用 CANN 版本和单卡预检契约，镜像标签 `org.seacraftasr.code-format=cython` 标记代码格式。Cython 扩展仍可逆向，保护目标是交付时移除可直接阅读的业务源码。

## 构建镜像和保留缓存重试

先登记本次创建资源及原有镜像/容器清单，再准备任务私有插件和构建器。下面只创建独立资源，不修改全局Docker配置；`WORK_DIR`沿用上一步。端口15087必须空闲，任务ID须唯一。所有pull均固定辅助镜像摘要，CANN底座始终来自本机。

```bash
export TASK_ID=$(basename "$WORK_DIR")
export DOCKER_CONFIG="$WORK_DIR/docker-cli"
export ASR_BUILDER="asr-cann82-${TASK_ID}"
export REGISTRY_NAME="asr-registry-${TASK_ID}"
export REGISTRY_VOLUME="asr-registry-${TASK_ID}"
export REGISTRY_PORT=15087
export BASE_LOCAL_ID=sha256:634118c33bcc879ee709510a532f3e8f90cb0d6a884922b5996367f9a6a7d864
export BASE_ALIAS="127.0.0.1:${REGISTRY_PORT}/asr-cann82-base:${TASK_ID}"
export REGISTRY_IMAGE=registry@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373
export BUILDKIT_IMAGE=moby/buildkit@sha256:14aa1b4dd92ea0a4cd03a54d0c6079046ea98cd0c0ae6176bdd7036ba370cbbe
export FRONTEND_SOURCE=docker/dockerfile@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
export FRONTEND_ALIAS="127.0.0.1:${REGISTRY_PORT}/asr-dockerfile:${TASK_ID}"
mkdir -p "$DOCKER_CONFIG/cli-plugins" "$WORK_DIR/tools"
docker image ls --no-trunc --digests > "$WORK_DIR/images-before.txt"
docker ps -a --no-trunc > "$WORK_DIR/containers-before.txt"
docker volume ls > "$WORK_DIR/volumes-before.txt"
# 下载只放入本任务目录，不覆盖用户全局插件。
curl --fail --location --retry 3 \
  https://github.com/docker/buildx/releases/download/v0.20.1/buildx-v0.20.1.linux-arm64 \
  -o "$WORK_DIR/tools/buildx-v0.20.1.linux-arm64"
curl --fail --location --retry 3 \
  https://github.com/docker/buildx/releases/download/v0.20.1/checksums.txt \
  -o "$WORK_DIR/tools/checksums.txt"
(cd "$WORK_DIR/tools" && grep ' buildx-v0.20.1.linux-arm64$' checksums.txt | sha256sum -c -)
install -m 0755 "$WORK_DIR/tools/buildx-v0.20.1.linux-arm64" "$DOCKER_CONFIG/cli-plugins/docker-buildx"
docker buildx version
# 先检查空闲端口；不能使用或停止其它任务的同名资源。
REGISTRY_PORT="$REGISTRY_PORT" /home/zhangs/conda-envs/asr/bin/python -c 'import os,socket; s=socket.socket(); s.bind(("127.0.0.1",int(os.environ["REGISTRY_PORT"]))); s.close()'
docker volume create --label "asr.task=$TASK_ID" "$REGISTRY_VOLUME"
docker run -d --name "$REGISTRY_NAME" --label "asr.task=$TASK_ID" \
  -p "127.0.0.1:${REGISTRY_PORT}:5000" \
  -v "$REGISTRY_VOLUME:/var/lib/registry" "$REGISTRY_IMAGE"
docker image inspect "$BASE_LOCAL_ID" --format '{{.Id}} {{.Architecture}}'
docker image tag "$BASE_LOCAL_ID" "$BASE_ALIAS"
docker image push "$BASE_ALIAS"
# 同时验证manifest的config身份及固定传输后的manifest digest。
BASE_IMAGE=$(/home/zhangs/conda-envs/asr/bin/python -c 'import json,os,urllib.request; alias=os.environ["BASE_ALIAS"]; repo,tag=alias.rsplit(":",1); req=urllib.request.Request("http://"+repo.rsplit("/",1)[0]+"/v2/"+repo.rsplit("/",1)[1]+"/manifests/"+tag,headers={"Accept":"application/vnd.docker.distribution.manifest.v2+json"}); r=urllib.request.urlopen(req,timeout=30); m=json.load(r); assert m["config"]["digest"]==os.environ["BASE_LOCAL_ID"]; digest=r.headers["Docker-Content-Digest"]; assert digest.startswith("sha256:"); print(repo+"@"+digest)')
export BASE_IMAGE
printf '%s\n' "$BASE_IMAGE" > "$WORK_DIR/base-image-ref.txt"
# BuildKit未必能直达Docker Hub；frontend也由daemon pull后经私有registry传输。
docker pull --platform linux/arm64 "$FRONTEND_SOURCE"
export FRONTEND_LOCAL_ID=$(docker image inspect --format '{{.Id}}' "$FRONTEND_SOURCE")
docker image tag "$FRONTEND_LOCAL_ID" "$FRONTEND_ALIAS"
docker image push "$FRONTEND_ALIAS"
FRONTEND_IMAGE=$(/home/zhangs/conda-envs/asr/bin/python -c 'import json,os,urllib.request; alias=os.environ["FRONTEND_ALIAS"]; repo,tag=alias.rsplit(":",1); req=urllib.request.Request("http://"+repo.rsplit("/",1)[0]+"/v2/"+repo.rsplit("/",1)[1]+"/manifests/"+tag,headers={"Accept":"application/vnd.docker.distribution.manifest.v2+json"}); r=urllib.request.urlopen(req,timeout=30); m=json.load(r); assert m["config"]["digest"]==os.environ["FRONTEND_LOCAL_ID"]; digest=r.headers["Docker-Content-Digest"]; assert digest.startswith("sha256:"); print(repo+"@"+digest)')
export FRONTEND_IMAGE
printf '%s\n' "$FRONTEND_IMAGE" > "$WORK_DIR/frontend-image-ref.txt"
cat > "$WORK_DIR/buildkitd.toml" <<EOF
[registry."127.0.0.1:${REGISTRY_PORT}"]
  http = true
  insecure = true
[worker.oci]
  gc = false
  max-parallelism = 4
EOF
docker buildx create --name "$ASR_BUILDER" --driver docker-container \
  --driver-opt network=host --driver-opt "image=$BUILDKIT_IMAGE" \
  --buildkitd-config "$WORK_DIR/buildkitd.toml" --bootstrap
docker buildx inspect "$ASR_BUILDER" > "$WORK_DIR/builder.txt"
docker inspect "$REGISTRY_NAME" > "$WORK_DIR/registry-container.json"
docker volume inspect "$REGISTRY_VOLUME" > "$WORK_DIR/registry-volume.json"
```

构建器内部使用host网络访问上述loopback registry；registry宿主端口仍只绑定127.0.0.1。记录实际builder容器和cache volume ID，不能按名称猜测删除。原有基础/最终镜像和共享资源列为保护项。上一步验证成功的`BASE_IMAGE`用于下面构建：

```bash
export DOCKER_CONFIG="$WORK_DIR/docker-cli"
# BASE_IMAGE必须是已核验的私有地址@sha256:manifest_digest，不是示例短标签。
: "${BASE_IMAGE:?先设置本地底座经私有registry传输后的不可变引用}"
export CANDIDATE_IMAGE="asr_offline_910b:candidate-${TASK_ID}"

docker buildx build --builder "$ASR_BUILDER" --load --progress plain \
  --provenance=false --sbom=false --platform linux/arm64 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg BUILDKIT_SYNTAX="$FRONTEND_IMAGE" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-context runtime_env="$WORK_DIR/runtime_env" \
  --build-context model_assets="$WORK_DIR/model_assets" \
  --build-context cython_tools="$WORK_DIR/cython_tools" \
  -f docker/Dockerfile -t "$CANDIDATE_IMAGE" .
```

构建参数 `BUILDKIT_SYNTAX` 将Dockerfile前端固定到已核验的私有引用，避免独立BuildKit尝试访问不可达的Docker Hub。首次实际构建在解析公网前端时超时的记录必须保留，不计为镜像构建成功。前端覆盖机制见[Docker官方文档](https://docs.docker.com/build/buildkit/frontend/)。`worker.oci.gc=false`用于避免失败重试期间自动GC删除本次cache，成功验收后仍须清理整个本轮私有builder缓存。

使用同一WORK_DIR/builder重试，不加`--no-cache`，失败时保留已完成层和原始日志。长构建应由远端持久后台控制器登记PID/出生、日志和退出文件，SSH断开不等于构建失败；确认旧任务已退出才重试，不并行构建。构建成功后先用候选image ID测试，验收通过后再给同一ID添加 `asr_offline_910b:v3_YYYYMMDD`，不重新构建。

## 启动服务

以下命令从本目录执行，不需要.env或额外启动脚本。选择空闲宿主端口和独立Compose项目名，避免接管已有容器。

```bash
cd /home/zhangs/asr_nvidia/docker
export ASR_IMAGE=asr_offline_910b:v3_20261125
export QWEN_MODEL_DIR=/home/model/Qwen3-ASR-1.7B-hf
export ASCEND_VISIBLE_DEVICES=7
export ASR_RT_VISIBLE_DEVICES=7 # Compose将该覆盖值传为ASCEND_RT_VISIBLE_DEVICES
export ASR_PORT=9000
export ASR_COMPOSE_PROJECT=asr-910b-v3
export ASR_CONFIG="$PWD/config.toml"
test -f "$ASR_CONFIG"
test -d "$QWEN_MODEL_DIR"
docker compose -p "$ASR_COMPOSE_PROJECT" config
docker compose -p "$ASR_COMPOSE_PROJECT" up -d --no-build
docker compose -p "$ASR_COMPOSE_PROJECT" ps
```

Compose设置环境变量 `CONFIG_PATH=/config.toml`，主服务实际从该路径读取宿主只读挂载的配置。配置内容修改后需使用下文 `--force-recreate` 重新创建容器，确保文件重新绑定。配置和Qwen均为只读bind，缺失源不自动建目录。三个小模型在镜像 `/app/model/`，Qwen在 `/app/qwen-model`；配置只来自挂载的 `/config.toml`，不内置模板兜底。不要把宿主物理npu:7写进仅见单卡的容器配置。`ASCEND_VISIBLE_DEVICES`单独不能证明隔离：本机实测仅设置该变量时torch仍可见8卡，已在创建张量前拦截。Compose同时设置`ASCEND_RT_VISIBLE_DEVICES`，缺省沿用同一物理卡选择7；只有实测设备计数为1且映射确认后才执行NPU运算。`ASR_RT_VISIBLE_DEVICES`用于显式覆盖进程级编号：其它runtime若已重映射设备，须先只读核验再人工校准，不能凭卡号猜测。两变量含义见[昇腾环境变量文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0028.html)。父进程OMP/MKL=4、Qwen owner_threads=1分别生效，不因容器化更改调度/精度/复读/过滤策略。

预加载耗时可能超过三分钟，健康检查预留360秒start_period。它验证HTTP生命周期就绪及状态响应，不代表四owner都做过真实语音forward；正式压测前另做有界暖身及身份覆盖。`restart: "no"`避免测试结束后台自动拉起。

```bash
curl --fail --silent "http://127.0.0.1:${ASR_PORT}/get_status"
docker compose -p "$ASR_COMPOSE_PROJECT" logs --tail 100 -f
```

## 请求与完整B4验收

```bash
export AUDIO=/absolute/path/to/full-audio.wav
curl --fail --silent --show-error \
  -F "audioFile=@${AUDIO}" -F 'language=auto' \
  -F 'showSpk=true' -F 'showEmotion=true' -F 'showRoleIdentify=false' \
  "http://127.0.0.1:${ASR_PORT}/v3/seacraft_asr" > /tmp/asr-result.json
```

正式验收使用 `../docs/qwen-multi-instance-20261124/test-inputs.json` 中canonical a1～a4的四条独立完整WAV，SHA、原文件名编码和全长核对后共同起跑。全部Qwen owner真实暖身覆盖后计时，不能用短片、离线段或缓存结果代替。记录完整JSON、客户端逐请求耗时及P50/P95、整批墙钟、HBM、CAM/emo真实forward/skip和段级警告/丢弃。HTTP200、4/4处理完成、严格无警告和人工质量是不同口径。任一完整请求超过336秒或真实OOM/设备故障则停止该轮、有界诊断；B4无200秒硬门槛。

验收控制器在宿主执行，不进入镜像。先用候选镜像启动独立Compose项目，宿主端口设18083。脚本仅操作传入的本轮容器，并在完成/失败的finally正常停止它；必须提供精确候选image ID及本轮task ID，并核对容器的asr.build.task标签一致；不可核验或不匹配时不得stop。不要把既有生产容器ID传入。本轮实际task ID为`asr-docker-cann82-20261125-pwn3bhmy`，不能使用`standalone`代替验收归属。

```bash
cd /home/zhangs/asr_nvidia
export ASR_IMAGE="$CANDIDATE_IMAGE"
export ASR_PORT=18083
export ASCEND_VISIBLE_DEVICES=7
export ASR_RT_VISIBLE_DEVICES=7
export ASR_COMPOSE_PROJECT="asr-b4-${TASK_ID}"
export ASR_BUILD_TASK_ID="$TASK_ID"
export EXPECTED_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$CANDIDATE_IMAGE")
export ASR_CONFIG="$PWD/docker/config.toml"
export QWEN_MODEL_DIR=/home/model/Qwen3-ASR-1.7B-hf
docker compose -p "$ASR_COMPOSE_PROJECT" -f docker/docker-compose.yml up -d --no-build
export TEST_CONTAINER=$(docker compose -p "$ASR_COMPOSE_PROJECT" -f docker/docker-compose.yml ps -q asr-offline)
export WARMUP_WAV=/tmp/qc-warmup.wav
# 该clip只用于有界真实暖身；正式B4始终来自完整manifest。
test -f "$WARMUP_WAV"
PYTHONDONTWRITEBYTECODE=1 /home/zhangs/conda-envs/asr/bin/python scripts/validate_source_container.py \
  --container "$TEST_CONTAINER" --expected-image-id "$EXPECTED_IMAGE_ID" --task-id "$TASK_ID" \
  --base-url http://127.0.0.1:18083 \
  --manifest docs/qwen-multi-instance-20261124/test-inputs.json \
  --config docker/config.toml --output "$WORK_DIR/validation-b4" --clip "$WARMUP_WAV"
```

`--output`每轮使用新目录。检查result.json的整体验收及cleanup，不只看控制器退出或单条HTTP200；失败保留该轮原件及构建缓存。暖身WAV需是事先核验可读的真实语音，不能把暖身短片成绩作为四完整音频性能。

## 修改配置与停止服务

编辑宿主config后重建容器以重新绑定文件，尤其编辑器通过替换inode保存时不能只依赖旧bind。无需重建镜像：

```bash
docker compose -p "$ASR_COMPOSE_PROJECT" up -d --no-build --force-recreate
```

日志、统计和上传缓存均在该容器的 `/tmp/asr-offline` 下，删除容器前将需要保留的证据导出到本目录外：

```bash
cd /home/zhangs/asr_nvidia/docker
export RUN_DIR=/home/zhangs/asr-runs/$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR"
docker compose -p "$ASR_COMPOSE_PROJECT" logs --no-color > "$RUN_DIR/container.log"
docker compose -p "$ASR_COMPOSE_PROJECT" cp asr-offline:/tmp/asr-offline "$RUN_DIR/runtime"
docker compose -p "$ASR_COMPOSE_PROJECT" stop
docker compose -p "$ASR_COMPOSE_PROJECT" ps -a
```

Compose正常停止宽限240秒；主程序处理TERM并关闭本轮Qwen/CAM/VAD及在途设备工作。先核对退出码、全部owned worker、端口连续空闲至少5秒及私有PCM清空，异常退出/强制kill不能当正常收尾。保留停止后的日志和统计，再移除本项目容器：

```bash
docker compose -p "$ASR_COMPOSE_PROJECT" logs --no-color > "$RUN_DIR/container-after-stop.log"
docker compose -p "$ASR_COMPOSE_PROJECT" cp asr-offline:/tmp/asr-offline "$RUN_DIR/runtime-after-stop"
docker compose -p "$ASR_COMPOSE_PROJECT" down
```

## 验收后发布和清理

只有完整B4、镜像内容审计及正常关闭全部通过后，给已测候选image ID添加最终日期tag：

```bash
export TESTED_ID=$(docker image inspect --format '{{.Id}}' "$CANDIDATE_IMAGE")
export FINAL_IMAGE=asr_offline_910b:v3_$(date +%Y%m%d)
docker image tag "$TESTED_ID" "$FINAL_IMAGE"
docker image inspect "$FINAL_IMAGE" --format '{{.Id}}'
```

清理前按本次资源登记逐项列出镜像/container/volume/builder、核对出生归属及容器引用。移除本次独立builder及其cache、registry/底座和frontend传输alias、候选多余tag、无引用的本次新增helperimage（含frontend源镜像）和临时构建材料。保留最终及基础镜像和它们必需的共享层、所有原有镜像/共享服务、日志报告。失败阶段保留缓存用于继续修复；不得执行全局docker system/image/builder prune或按名称批量kill。具体本次清理ID及前后核验写入验收报告，不让通用命令误删其它项目。

## 2026-11-25实测交付

镜像`asr_offline_910b:v3_20261125`（ID `sha256:ef7b8add6a42c0297dc66eaa58718a67df8fba688d1738f1c314fe59b8565c5a`）已通过真实完整B4和正常退出。冷加载至HTTP就绪约416～419秒，四路短clip暖身30秒单列；up -d返回不表示模型就绪，以状态/health确认。B4完成4/4，P50/P95/墙钟182.44/196.25/198.65秒，严格无警告2/4；16警告含9恢复/7丢弃，非人工准确率认证。详见[报告](../docs/docker-source-cann82-20261125/REPORT.md)。本轮中间镜像、私有构建缓存及辅助资源已清，测试容器已释放。
