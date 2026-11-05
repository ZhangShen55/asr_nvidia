# VAD 日志 RMS 与 SNR 折线图

来源：`/home/zhangs/.codex/attachments/1f48900f-1ef0-4535-910c-b86546575920/已粘贴的文本.txt`。共解析 **665** 条“丢弃低音量段”日志。

- `low_volume:snr`：614 段
- `low_volume:rms_dbfs,snr`：51 段
- RMS 范围：-47.32～-9.20 dBFS，中位数 -22.89 dBFS
- SNR 范围：3.95～39.86 dB，中位数 24.25 dB
- RMS 低于 -41 dBFS：51 段
- SNR 低于 28 dB：418 段；SNR 不低于 28 dB：247 段

图中 SNR=28 dB 是对比线，SNR=40 dB 只是根据日志中最高被丢弃值 39.86 dB 推断的参考线；原日志没有配置快照，不能仅凭日志严格确认实际配置值。
