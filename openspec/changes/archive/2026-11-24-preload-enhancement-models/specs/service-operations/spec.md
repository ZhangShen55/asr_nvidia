## MODIFIED Requirements

### Requirement: 启动前校验部署前置条件
昇腾部署 MUST 在就绪前校验设备可见性、必需资产、torch_npu运行时及CONFIG_PATH可读性，失败明确报告并阻止就绪。三个必需模型资产与Qwen权重策略分开：preload=false时Qwen缺失在转写触发按4011处理，不为预检加载；正式默认preload=true时全部N份权重及processor须顺序加载成功且健康后才就绪，任何部分启动失败统一回收。前置校验本身不得加载模型或绕过功能开关；校验通过后，HTTP就绪前须完成已开启且非空数组的增强池预加载尝试，失败按增强池降级契约处理。

#### Scenario: GPU 或资产不可用
- **WHEN** 容器无可用NPU、必需模型目录缺失或配置不可读
- **THEN** 启动指出具体前置条件并失败，不接受请求；默认Qwen预加载失败同样阻止就绪并回收已启动owner，显式懒加载的Qwen缺失仍按原4011处理

#### Scenario: 前置条件通过
- **WHEN** 设备、目录和配置检查通过，且按当前preload策略完成必要加载
- **THEN** 服务就绪并报告镜像tag、实际配置路径、后端和N份实际健康加载状态；不把预检、READY或模型名当真实语音forward暖身
