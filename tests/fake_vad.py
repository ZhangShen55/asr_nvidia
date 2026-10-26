"""VAD 进程池单测用的轻量替身。

真实 FSMN-VAD 的加载要数秒、数百 MB 内存，单测不该承担这份成本。这里用纯
Python 的「模型」与「任务」把 worker 进程、消息流转、扩缩容与失败语义跑通。

任务指令由音频路径的文件名承载：

- ``ok``            正常返回两段区间
- ``sleep:0.5``     模拟耗时
- ``error``         抛异常（单次任务失败）
- ``crash``         让 worker 进程直接退出（模拟崩溃）
- ``procname``      返回 worker 自己的进程名
"""

from __future__ import annotations

import os
import time


def build_fake_model(model_dir: str, device: str, vad_kwargs: dict) -> dict:
    return {"model_dir": model_dir, "device": device, "vad_kwargs": vad_kwargs}


def _process_title() -> str:
    try:
        with open("/proc/self/cmdline", "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
    except OSError:
        return ""


def run_vad_task(model, audio_path: str, vad_kwargs: dict):
    command = os.path.basename(audio_path)
    if command == "crash":
        os._exit(17)
    if command == "error":
        raise RuntimeError("模拟 VAD 任务失败")
    if command == "procname":
        return [{"key": _process_title(), "value": [[0.0, 1000.0]]}]

    if command.startswith("sleep:"):
        time.sleep(float(command.split(":", 1)[1]))
    return [
        {
            "key": command,
            "value": [[0.0, 1000.0], [2000.0, 3500.0]],
        }
    ]
