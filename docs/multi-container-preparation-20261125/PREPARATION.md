# 五卡多容器离线部署准备（2026-11-25）

用户明确只准备文件，由用户在另一台服务器自行启动，禁止运行验证。本轮没有docker create/run/compose up，没有创建任何容器或network；现有容器与network前后ID/元数据保持。

交付/data/asr_offline_910b/config2.toml、docker-compose-multi.yml、nginx.conf、nginx.tar。三份配置的入库副本位于本目录；nginx.tar仅在服务器上，未将二进制归档入Git。Compose project为asr-offline-910b-multi，与此前单容器部署分开。

五个ASR分别使用物理0/1/5/6/7，通过两个Ascend可见性变量限制单卡，容器统一逻辑npu:0。各容器HTTP1/Qwen5/S16/global80，A+B共享总音频5、B子集5、等待超时0，ITNfalse。CAM4 process/emo1沿用现有部署，未按Qwen数复制；模型共享只读/data/asr_offline_910b/Qwen3-ASR-1.7B-hf。每容器独立IPC模型/准入，Nginx least_conn依据活跃连接选择后端，不是跨实例全局业务任务计数；不能据此保证25个请求一定全部准入。4013不自动重放到另一后端。get_status/clear_tasks_list也只路由一个实例，不是集群汇总/广播。

官方当前mainline为1.31.10，稳定分支1.30.8；选择官方nginx:1.31.10-alpine的linux/arm64镜像。版本对应近期官方安全公告的修复线，不承诺没有未知漏洞或完整OS依赖零CVE。来源：https://nginx.org/en/download.html 和 https://nginx.org/en/security_advisories.html 。server_tokens off隐藏版本号，后端Server/X-Powered-By不透传；不启用本次不需要的HTTP2/3、TLS/mail/stream监听。Nginx非root、只读rootfs、仅tmp可写；只有8081映射容器8080，后端不发布宿主端口。

使用共享zone保证两个Nginx worker共享least_conn状态；Docker resolver/resolve支持容器换IP；HTTP1.1 upstream keepalive，长音频上传限制关闭，600秒读写超时，关闭请求/响应buffer和POST自动重试。五后端全部healthy才启动Nginx，Nginx自身health不代表各池实际推理成功。

## 由用户在目标机执行

目标机应具备同架构910B驱动和Ascend容器runtime，8081可用。文件及Qwen模型位于/data/asr_offline_910b；若未加载ASR镜像，先load已有ASR归档。

```sh
cd /data/asr_offline_910b
docker load -i asr_offline_910b_v3_20261125.tar
docker load -i nginx.tar
docker compose -f docker-compose-multi.yml up -d
docker compose -f docker-compose-multi.yml ps
```

访问POST http://目标机IP:8081/v3/seacraft_asr。修改config2后需要重启后端；修改nginx.conf后需要重启Nginx。修改Compose/挂载后可再次up -d使配置生效。

## 已完成和未完成的检查

完成：Compose静态解析、六服务网络/端口/物理卡/只读路径展开核对、TOML字段、镜像arm64及精确版本元数据、tar manifest/tag/config SHA与全部层diff_id核对。配置及镜像归档SHA见metadata.json。原始审计日志路径：/home/zhangs/asr_nvidia/logs/multi-deployment-prep-20261125-112249。

未执行：nginx -t、任何容器启动/网络创建、HTTP代理/版本隐藏实测、25 owner共驻的HBM/性能/输出验收。N5/B5/CAM4/emo1在新机器的实际可用性由用户后续验证，不以本次静态准备冒称通过。

初次network前后比较直接比JSON文本，Docker标签顺序变化导致误判；改为解析字段并对标签排序后，实际network身份/内容完全一致。原失败工具输出保留；没有因此操作网络或容器。

部署目录相关挂载统一相对Compose文件目录：./config2.toml、./Qwen3-ASR-1.7B-hf、./nginx.conf。目标目录可整体搬迁；Ascend宿主系统路径和容器内路径仍为必要绝对路径。本次只做静态解析，未启动任何服务。
