# Design

## Context

动机和范围见 proposal。源码当前为 `8b8bfa12a144f008c5a20a00f3f6fadd1cc0fbc1`，已完成增强数组配置精简；本次仍须按实际构建提交重新冻结。根配置另有用户未提交的 `vad_segment_decibel_thres=-100.0` 修改，不能覆盖或混入本 change 提交。

已只读核验：

| 项目 | 现状及约束 |
| --- | --- |
| 指定底座 | 本机 `cann:8.2.rc1-910b-ubuntu22.04-py3.10`，linux/arm64，image ID `sha256:634118c33bcc879ee709510a532f3e8f90cb0d6a884922b5996367f9a6a7d864`，约13GB；尚未在本次运行 |
| 宿主运行环境 | 独立 asr Python3.10，torch/torch_npu2.5.1，现用 CANN8.1 配套；不原地升级 |
| Docker | client/server27.5.1，Compose2.39.2，Ascend runtime已注册；无Buildx插件，不能假定现代构建命令已可用 |
| 驱动 | version.info显示25.0.rc1.1；仍需按官方矩阵与实际容器 forward核验，版本号本身不是兼容通过 |
| 存储 | DockerRootDir为 `/data/docker`，/data约3TB可用、/home约3.2TB可用；/tmp为大容量tmpfs。系统根分区余量约648MB，需监测，但不得误判Docker数据目录没有空间 |
| 当前服务 | 单HTTP、Qwen4 continuous/process/BF16、每owner S16、CAM4 process、emo1、VAD8 CPU；增强已启动预加载 |
| 性能参考 | 历史最终B4约200.10秒，发生在最新复读恢复等提交前；不能直接当本次同版本性能基线 |
| 共享资源 | 卡7及既有容器仍有外部服务，保留所有既有进程；包括共享PID1601397/出生1791963491.5及其网络命名空间8007 |

现有Docker目录含build.sh/start.sh/nginx.conf、旧compose、忽略规则和历史报告；现有Cython构建工具在仓库及独立构建前缀中，不能因此要求本版执行保护构建。

## Goals / Non-Goals

**Goals:** 从本地底座形成可复现的源码镜像；证明配置挂载、多进程NPU和真实完整B4可用；失败可定位并复用缓存，成功能清掉本轮构建资源且保留可运行最终镜像。

**Non-Goals:** 本版不改变推理算法、实例组合、VAD或复读阈值，不新增推理后端，不做Cython，不改宿主依赖/驱动，不干扰其它项目，不推送镜像到外部registry，不自动长期上线，不把处理完成当人工召回率认证。

## Decisions

### 1. 容器专用依赖与宿主分离

官方 Ascend PyTorch 配套表将 CANN8.2.RC1/torch2.5.1 对应到 torch-npu2.5.1.post1，而宿主原组合对应8.1。

来源：[Ascend PyTorch v2.5.1-7.1.0 官方说明](https://raw.githubusercontent.com/Ascend/pytorch/v2.5.1-7.1.0/README.zh.md)。固定该来源及访问日期2026-11-24，apply时保存配套证据。

在仓库Docker目录外新增容器专用依赖锁/重建说明，记录实际分发来源、wheel摘要、pip/conda区分、Python3.10/aarch64、CANN toolkit/kernel/驱动和模型revision。运行环境可通过只读打包现有asr前缀复用，但须在独立构建层重定位，按锁仅替换torch-npu为post1并完整核对依赖；不得用系统Python叠加asr的site-packages。其它已锁版本保留，新增必要依赖必须追溯真实缺项，不使用pip install -U。

conda-pack可由现有独立工具环境调用，打包的对象是运行环境，不能把Cython/conda-pack工具环境整体作为运行环境。本版不安装项目保护工具；运行包自带必要原生扩展保留。归档、wheelhouse和构建说明/锁作为目录外材料，必要时通过命名上下文提供，不能将宿主整个环境或项目全目录直接挂入运行容器。

先验证底座实际组件、解释器/包来源、依赖一致性和导入，再在物理7对应逻辑0执行矩阵计算与真实Qwen prefill/decode、CAM/emo forward。现有core/deployment.py版本归一化会去掉.post，不能用此结果证明post1配套；本版需容器专用精确distribution版本门禁，记录实际构建后缀/来源，错误2.5.1须在worker创建前拒绝，同时宿主2.5.1仍合法。原检查脚本若固定torch_npu精确等于2.5.1，不得通过删检查或运行时忽略混用；应使用容器专用配套检查，并保留主入口实际预检。宿主脚本和锁不因容器post1改变。真实驱动不兼容则给出明确阻塞，不换8.1冒充本版交付，也不升级宿主驱动。

### 2. 四文件部署目录及最小源码内容

最终目录恰好为：

```
docker/
  Dockerfile
  docker-compose.yml
  config.toml
  README.md
```

旧验证报告迁至日期明确的docs子目录保留，旧Docker脚本和Nginx配置按本次授权移除；根README/AGENT的入口说明同步。不删除仓库独立Cython工具或历史保护记录。

根 `.dockerignore` 采用先全部排除再白名单放行，主上下文仅开放Dockerfile和以下源码；依赖归档、锁及必要小模型通过独立严格筛选的命名上下文输入。检查真实发送清单，不使用 `COPY . .`，也不先复制全仓库再删除。

| 允许输入 | 核验方式 |
| --- | --- |
| main.py、model_status_runtime.py、telemetry_runtime.py | 冻结提交及SHA，确认实际运行引用 |
| core/api/utils/entity必要.py及__init__.py | 递归import/运行检查，不能复制旧同名.so、pyc或缓存 |
| 三个小模型必要权重、配置和确有运行用途的附属资产 | 每个目录清单与摘要，忽略其无关说明/实验资料，不改原始权重 |
| 已锁定运行环境及必要第三方二进制/规则数据 | 包来源/依赖审计；不以“.so都是禁止”误删torch、pynini等必需扩展 |
| 构建追溯元数据 | 源码revision、基础身份、依赖清单等，不含密钥、正文或宿主私有配置 |

项目test/tests/docs/scripts/openspec/.git/.codex、样本、日志报告、setup_cython.py、构建C文件、旧业务扩展均不得进入任一构建阶段。Dockerfile/Compose/README也无需进入应用目录；配置仅挂载。第三方分发包、基础OS和CANN自带的必要文件与项目禁入目录分开审计，不机械删除依赖。底座若带有其它项目代码/样本，必须报告并处理其实际遗留，不能只审查本次COPY后宣称镜像干净。

### 3. 一个HTTP服务，内部模型进程保持原架构

Dockerfile内联入口在 `set +u` 下加载CANN环境，检查CONFIG_PATH及本版HTTP1限制，然后 `exec /opt/asr/bin/python /app/main.py --host 0.0.0.0 --port 9000 --workers 1`，保留主程序预检和TERM正常清理。仅使用必要内联命令，不把项目start.sh/scripts目录复制进镜像。Compose使用init、桥接端口映射、runtime=ascend、privileged和ipc=host；仅物理7可见并实证映射逻辑0。

容器入口在模型创建前拒绝instance_count!=1。loader的一般HTTP字段语义不重命名为Qwen数量，不在本版实现多HTTP/代理。Qwen数量仍由配置local_instance_count控制，默认4；S16，派生总4/global64、B4，CAM数组四份逻辑npu:0、emo一份。VAD8 CPU及有效推理/过滤参数沿当前冻结配置，父OMP/MKL4、Qwen owner_threads1保持相应参考，避免镜像CPU线程数变化成为隐蔽性能变量。

### 4. 必需的只读配置及模型挂载

默认Compose将自身同目录 `./config.toml` 以bind只读挂载到 `/config.toml`，设置CONFIG_PATH；可用ASR_CONFIG覆盖宿主来源。使用长语法禁止缺失源自动建目录，容器再验证是可读普通文件。镜像不COPY配置，也无根开发配置兜底；修改配置后restart/recreate生效，README明确配置inode替换时采用recreate重新绑定。

三个小模型保留镜像内/app/model；QWEN_MODEL_DIR必须指向存在、完整的本地HF权重，独立只读挂到/app/qwen-model。日志、统计、上传缓存和可写编译缓存路径由现有支持项与环境明确隔离，必要宿主目录在Docker目录外创建；不以增加docker目录内的脚本或.env文件解决。用户根配置只读记录，不改写、不提交。新的Docker配置与实测有效值对应，但将设备/路径转换为容器语义。

预加载历史约206秒，健康检查不能继续用旧120秒窗口。先以360秒start_period容纳正常加载，记录实测后有依据调整；健康必须在lifespan加载完成后检查状态，不在健康查询中加载。按现有HTTP drain30秒、模型关闭grace120秒、外层回收210秒分层，Compose stop_grace_period使用240秒覆盖外层收尾，并用实际关闭证据确认。强制kill或非零退出不能当正常通过。

### 5. 构建缓存有独立归属，失败保留、验收后清理

选用任务私有Buildx docker-container builder，避免共享daemon缓存无法证明归属。主机无Buildx，apply时下载固定版本arm64插件并校验官方摘要，放入任务私有Docker CLI配置目录；不升级daemon、不覆盖用户全局配置。BuildKit及本地传输辅助镜像也固定来源和digest并登记。

该builder看不到daemon中已有CANN镜像，不能直接FROM短标签并假定沿用本地底座。使用本轮临时registry，仅发布宿主127.0.0.1空闲端口；为builder配置host网络，仅该localhost registry允许HTTP/insecure，不修改daemon全局registry设置。将已核验本机base ID临时tag/push到该本地registry，记录manifest digest与config digest，构建FROM该任务私有digest；核对其config digest仍为指定本地base ID。该传输不对外发布镜像。

首步登记镜像/容器/volume/builder/端口/磁盘快照和任务ID，再创建任何资源。资源清单放Docker目录外，记录归属、创建时间、ID、引用、base/final保护项；基底本来存在的13GB镜像和原有114GB可回收镜像不能被纳入本轮删除范围。

同一任务的依赖层、源码层和模型层分别布局，失败后保持同一builder及缓存，不用--no-cache、不重建builder。依赖/模型未变时复用层；内容失败修复后再构建，构建与NPU测试均只允许运行一个控制器。长构建在远端持久后台记录PID/出生/退出状态及日志，SSH断连不等于构建失败；卡住时取IO、进程与构建日志定位后再操作，不能无限重复同一失败。

构建load候选image到daemon测试，验收后将确切image ID打最终日期tag，不二次build。清理顺序：停止并核验测试资源→最终tag/ID/文件清单确认→删除本轮candidate多余tag、无引用中间image→关闭/移除私有builder及其缓存volume→移除registry/container/volume、本地base临时alias、任务新增且无引用helperimages与临时context/归档。每一步先输出清单并核对引用，只在本轮授权范围执行；日志/报告保留。最终image ID和底座ID仍须可inspect。

不得全局docker system/image/builder prune、强删被容器引用镜像、移除已有helper镜像或删最终image需要的共享层。Builder移除不使用keep-state；若版本/存储驱动仍留下任务cache，按精确登记volume补查清理并记录。清理前后disk/cache/owned资源清单证明完成，不要求最终image物理大小不含其必要层。

参考：[Docker Buildx移除builder及state选项](https://docs.docker.com/reference/cli/docker/buildx/rm/)。

### 6. 用真实完整B4证明可用

从现有 `docs/qwen-multi-instance-20261124/test-inputs.json` 读取canonical a1～a4文件名原始编码、路径、完整时长及SHA，不手工重构特殊Unicode文件名。四条完整约52～55分钟音频共同起跑，参数showSpk=true/showEmotion=true/showRoleIdentify=false；不预切VAD、不复用结果、不用短片替代。宿主控制器向已load候选镜像的真实v3接口上传，私有测试端口默认18083且先验证空闲。

验证分层：静态Compose/内容/依赖→NPU矩阵及真实短请求→全四owner真实prefill/forward覆盖及增强真实调用→活动/PCM归零并冻结暖身后计数→完整B4。有限暖身不得无限重复请求；身份覆盖以PID出生及执行证据判断，不按发送四次就假定覆盖四owner。冷启动和首forward耗时分开。

正式记录：源码/底座/候选ID与tag、配置SHA和有效参数、依赖/驱动、样本SHA/时长、每请求monotonic start/end/elapsed、HTTP和业务状态、processing_status/segment_warnings、逐段异常/恢复/丢弃、CAM/emo真实forward/skip/null、各owner请求归属、队列/阶段、父/worker及卡聚合HBM、外部占用、完整JSON。输出客户端线性插值P50/P95（N=4小样本）、从共同发起到最后完整响应墙钟、完成数/4与严格无警告数/4。报告与JSON在docs/logs的日期目录，不进镜像或docker目录。

通过条件：四个完整请求成功完成，无真实OOM/设备故障，无新增未解释的缺段、接口或文本恢复行为回归，增强实际执行及skip口径正确，输入/输出归属、预算和资源收尾成立；段级既有保守恢复警告单列，不把HTTP200或completed_with_warnings称为准确率通过。若发现可疑文本/丢段或性能回退，按同源码/配置宿主路径对照定位，历史不同提交成绩不能充当单因素基线；人工真值不足时明确质量未认证。

任一完整请求墙钟超过336秒或真实故障，停止该轮追加、有界取证、正常回收，不丢弃慢请求算整体分位数。失败轮保留缓存和全部证据，修复容器相关根因后重验；若根因超出本change且需改变业务策略/宿主驱动，明确阻塞，不擅自扩大范围。B4无200秒硬门槛，历史目标及实测差距如实列出。

结束正常停止本轮容器，按宿主/容器PID映射及出生身份核对所有owner/CAM/VAD退出、无孤儿、私有PCM为空、测试端口连续空闲至少5秒，共享PID/容器不变。所有验收通过再清缓存；最后交付最终镜像和四文件，不留下本轮测试服务常驻。

### 7. 并行划分与单一执行控制

按AGENT的并行要求可拆：A负责Dockerfile、上下文及容器依赖；B负责Compose/config/README和启动语义；C只读审核规格、内容白名单和验收工具。各文件唯一作者，依赖变更互相通报。主agent整合、提交并独占构建器/registry/NPU7/测试端口/清理，不并行构建或压力测试；各agent不得停止共享服务。规划任务本身已有独立只读交叉评审。

## Risks / Trade-offs

- [8.2配套或驱动不兼容] → 先验证真实组件和forward，post1仅在容器修改；主机阻塞明确报告。
- [最小内容漏掉动态导入/ITN数据] → 依赖清单、导入来源及真实三模型请求联合检查，不能靠把整个仓库复制回去修复。
- [底座有遗留代码、依赖包带无关开发内容] → 分别审计基础层和新增上下文，保留运行必需项，报告无法消除的基础内容边界。
- [独立builder获取不到本地base或helper下载失败] → 固定摘要、私有registry传输及身份比对；保留日志/cache，不静默改用另一底座。
- [长预加载被误判不健康或关闭被提前kill] → 启动/暖身/正常关闭分开计时，健康及停止窗口覆盖已测过程。
- [失败后误清缓存/共享镜像] → 归属快照、专属builder及引用检查，只有全部成功才执行范围内清理。
- [CANN/post1改变性能或输出] → 完整B4记录并有条件执行同版本宿主对照，不承诺容器必然提速。

## Migration Plan

1. 保存当前Git/用户配置、镜像、共享服务、磁盘和输入清单；实现并校验四文件及容器专用依赖，不修改宿主环境。
2. 创建独立构建资源，完成配套探针、真实构建及内容审计；失败保留缓存并逐因修复。
3. 使用候选ID完成暖身覆盖与完整B4，正常释放服务，冻结报告。
4. 通过后标记最终日期tag并清理本轮临时构建资源；核对可inspect和可部署的最终镜像及Compose引用，Git提交/推送仅含本change内容，保留用户根配置差异。
5. 所有任务有证据完成后同步本change规格、归档；旧open change和历史成绩不改写。生产长期运行另行执行README命令。

回退：候选失败不更新最终tag、不删除已有镜像或本轮可用缓存；停止本轮容器、保留报告及变更版本。原宿主环境与原有服务未修改，无需通过改依赖才能恢复。已存在的同日不同最终tag应保留其原ID记录，不能为让新验收通过悄悄覆盖。

### 实测补充：特权容器的进程可见卡限制

2026-11-25候选镜像实测仅设置ASCEND_VISIBLE_DEVICES=7时，torch.npu.device_count()仍为8，已在张量创建前拒绝继续。补齐ASCEND_RT_VISIBLE_DEVICES=7后实际仅1卡，逻辑0的BF16矩阵通过；宿主npu-smi物理7进程表与容器PID/NSpid出生身份相符。Compose同时传入两项，镜像入口补齐进程可见性缺省，源码容器预检强制count=1及逻辑npu:0；不改变宿主多卡配置语义。证据physical-mapping-proof.json和gpu-probe-r2-result.json。
