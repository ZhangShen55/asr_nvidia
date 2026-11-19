"""历史实验导入兼容；原型完整源码/SHA保留在RUN artifact，正式协议只维护core一份。"""
from core.qwen_asr_process import (
    LocalQwenProcessRuntime,
    QwenWorkerUnavailableError,
    RemoteRuntimeError,
)

ProcessQwenRuntime = LocalQwenProcessRuntime
WorkerProcessError = QwenWorkerUnavailableError

__all__ = ["ProcessQwenRuntime", "WorkerProcessError", "RemoteRuntimeError"]
