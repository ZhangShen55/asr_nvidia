"""旧实验名仅做兼容别名检查；真实spawn/故障矩阵集中test_local_qwen_process。"""
from core.qwen_asr_process import LocalQwenProcessRuntime, QwenWorkerUnavailableError
from scripts.qwen_process_candidate import ProcessQwenRuntime, WorkerProcessError


def test_legacy_process_candidate_names_are_formal_core_aliases():
    assert ProcessQwenRuntime is LocalQwenProcessRuntime
    assert WorkerProcessError is QwenWorkerUnavailableError
