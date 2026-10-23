# 规格增量

## MODIFIED Requirements

### Requirement: 配置 v2 链路模型的落卡
系统 MUST 支持通过配置指定 v2 链路本地模型的运行设备：CAM++（speech_campplus）与 emotion2vec 的 GPU 索引 MUST 由 `[model_device]` 配置并绑定同一张卡，默认 GPU 0；VAD 的运行设备 MUST 由独立的 `[qwen_asr].vad_device` 指定，默认 `"cpu"`，MUST NOT 再随 `[model_device]` 的 GPU 索引一同迁移。VAD 的设备配置 MUST NOT 影响 CAM++ 与 emotion2vec 的落卡。

#### Scenario: 修改落卡配置
- **当** 运维人员把 `[model_device]` 的 GPU 索引改为 1 并重启服务
- **则** CAM++ 与 emotion2vec 均加载到 GPU 1

#### Scenario: 配置缺省
- **当** 配置文件中缺少落卡配置
- **则** CAM++ 与 emotion2vec 均加载到 GPU 0，VAD 在 CPU 上运行

#### Scenario: VAD 单独切换到 GPU
- **当** 运维人员把 `vad_device` 设为 `"cuda:1"`
- **则** VAD 加载到 GPU 1，而 CAM++ 与 emotion2vec 仍按 `[model_device]` 加载

### Requirement: 通过配置外置运行参数
Qwen 适配链路的参数 MUST 全部来自配置文件，服务地址、模型名、请求并发度、音频并发度、队列容量与超时、VAD 参数 MUST NOT 硬编码在业务代码中。VAD 的运行设备、worker 池规模、等待队列水位、扩缩容阈值与超时 MUST 同样由配置提供，MUST NOT 以常量形式写死在代码里。

#### Scenario: 调整 VAD 池规模无需改代码
- **当** 运维人员把 `vad_pool_size` 从 8 改为 4 并重启服务
- **则** 服务以 4 个 VAD worker 运行，行为由配置决定

#### Scenario: 缺省配置下的 VAD 参数
- **当** 配置文件中缺少全部 VAD 进程池相关配置
- **则** 系统使用默认值：设备为 CPU、常驻 worker 8 个、扩容上限 16 个、排队水位 4、空闲回收 120 秒、排队超时 120 秒

#### Scenario: 修改并发参数后重启生效
- **当** 运维人员修改配置中的 Qwen 请求并发度与音频并发度并重启服务
- **则** 服务按新的并发参数运行

#### Scenario: 配置缺省时使用默认值
- **当** 配置文件中缺少相关配置项
- **则** 服务使用默认值：单音频内同时发往 Qwen 的段请求数 4、全部音频合计请求数 32、同时处理的音频数 8、队列满立即拒绝

## ADDED Requirements

### Requirement: VAD 阶段排队超时以独立错误码返回
系统 MUST 在请求等待 VAD worker 超过配置项 `vad_queue_timeout_s` 时终止该请求，`POST /v2/seacraft_asr` 的响应 MUST 返回错误码 4014 与可读的错误描述，且 MUST 立即释放该请求占用的音频处理槽位并清理其全部临时文件。

#### Scenario: VAD 排队超时的响应
- **当** 请求等待 VAD worker 超过 `vad_queue_timeout_s`
- **则** 响应返回 `{"msg": ..., "code": 4014}`，且不返回分段结果

#### Scenario: 排队超时不影响其他请求
- **当** 某个请求因 VAD 排队超时而失败
- **则** 其余在途请求与后续新请求仍能正常完成转写
