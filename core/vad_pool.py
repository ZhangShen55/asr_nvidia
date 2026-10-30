"""固定设备数组的 FSMN-VAD 进程池。

每个设备数组元素对应一个独立进程和一份 VAD 模型。池在服务启动时一次性
预热，任务按 FIFO 分派；运行期间不动态扩容或回收，数组长度就是完整音频
VAD 的有效并发上限。
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing as mp
import os
import queue as queue_stdlib
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

VAD_WORKER_PROCESS_NAME = "asr-offline-vad"
_MSG_IDLE = "idle"
_MSG_STARTED = "started"
_MSG_FINISHED = "finished"
_STOP_POLL_INTERVAL_S = 0.5
_MONITOR_INTERVAL_S = 1.0
_DEFAULT_STARTUP_TIMEOUT_S = 300.0


class VadPoolError(RuntimeError):
    code = 4014

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class VadQueueTimeoutError(VadPoolError):
    """保留错误类型以兼容调用方；固定池默认不对等待队列设置超时。"""


def _set_process_name(name: str) -> None:
    try:
        import setproctitle

        setproctitle.setproctitle(name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("设置 VAD worker 进程名失败: %s", exc)


def _import_target(spec: str) -> Callable:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"目标规格必须形如 module:function，实际为 {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def build_vad_model(model_dir: str, device: str, vad_kwargs: dict):
    from funasr import AutoModel

    return AutoModel(
        model=model_dir,
        device=device,
        vad_kwargs=vad_kwargs,
        disable_update=True,
        disable_pbar=True,
    )


def run_vad_task(model, audio_path: str, vad_kwargs: dict) -> Any:
    return model.generate(input=audio_path, cache={}, vad_kwargs=vad_kwargs)


def _write_warmup_wav() -> str:
    import math
    import struct
    import tempfile
    import wave

    rate = 16000
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate)))
        for i in range(rate // 2)
    )
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


def _worker_memory_mb(device: str) -> float:
    if not str(device).startswith("cuda"):
        return 0.0
    try:
        import torch

        return torch.cuda.memory_allocated(torch.device(device)) / (1024**2)
    except Exception:  # noqa: BLE001
        return 0.0


def _worker_main(
    worker_id: int,
    model_dir: str,
    device: str,
    vad_kwargs: dict,
    builder_spec: str,
    runner_spec: str,
    task_queue,
    result_queue,
    stop_event,
) -> None:
    _set_process_name(VAD_WORKER_PROCESS_NAME)
    try:
        builder = _import_target(builder_spec)
        model = builder(model_dir, device, dict(vad_kwargs))
        runner = _import_target(runner_spec)
        warmup_path = _write_warmup_wav()
        try:
            runner(model, warmup_path, dict(vad_kwargs))
        finally:
            try:
                os.remove(warmup_path)
            except OSError:
                pass
        allocated_mb = _worker_memory_mb(device)
    except BaseException:  # noqa: BLE001
        result_queue.put((_MSG_IDLE, worker_id, worker_id, "error", traceback.format_exc()))
        return

    result_queue.put((_MSG_IDLE, worker_id, worker_id, "ok", {"allocated_mb": allocated_mb}))
    while not stop_event.is_set():
        try:
            message = task_queue.get(timeout=_STOP_POLL_INTERVAL_S)
        except queue_stdlib.Empty:
            continue
        except (EOFError, OSError):
            break
        if message is None:
            break
        task_id, audio_path = message
        result_queue.put((_MSG_STARTED, task_id, worker_id, "ok", None))
        try:
            payload = runner(model, audio_path, dict(vad_kwargs))
            result_queue.put((_MSG_FINISHED, task_id, worker_id, "ok", payload))
        except BaseException:  # noqa: BLE001
            result_queue.put((_MSG_FINISHED, task_id, worker_id, "error", traceback.format_exc()))
        result_queue.put((_MSG_IDLE, worker_id, worker_id, "ok", None))


@dataclass
class _Worker:
    worker_id: int
    process: Any
    stop_event: Any
    task_queue: Any
    device: str
    ready: bool = False
    failure: Optional[str] = None
    busy_task_id: Optional[int] = None
    last_finished_at: Optional[float] = None
    allocated_mb: float = 0.0


@dataclass
class _PendingTask:
    task_id: int
    audio_path: str
    started: asyncio.Event
    finished: asyncio.Future
    worker_id: Optional[int] = None


class VadPool:
    """固定 worker 池；一个 worker 同一时刻只处理一个完整音频。"""

    def __init__(
        self,
        devices: Optional[list[str]] = None,
        *,
        # 这些参数仅保留显式测试/外部调用的兼容签名，不再参与动态扩缩容。
        pool_size: Optional[int] = None,
        pool_max: Optional[int] = None,
        watermark: Optional[int] = None,
        collapse_timeout_s: Optional[float] = None,
        queue_timeout_s: Optional[float] = None,
        call_timeout_s: Optional[float] = None,
        device: Optional[str] = None,
        model_dir: Optional[str] = None,
        vad_kwargs: Optional[dict] = None,
        builder_spec: str = "core.vad_pool:build_vad_model",
        runner_spec: str = "core.vad_pool:run_vad_task",
        start_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S,
    ):
        if devices is None:
            configured = settings.vad_device
            if device is not None:
                count = max(1, int(pool_size or 1))
                configured = [device] * count
            devices = configured
        self._devices = [str(item).strip().lower() for item in devices]
        self._model_dir = model_dir if model_dir is not None else settings.vad_model_dir
        self._vad_kwargs = {
            "max_single_segment_time": settings.qwen_vad_max_single_segment_time_ms,
            "max_end_silence_time": settings.qwen_vad_max_end_silence_time_ms,
            "speech_noise_thres": settings.qwen_vad_speech_noise_thres,
            "decibel_thres": settings.qwen_vad_decibel_thres,
            "snr_thres": settings.qwen_vad_snr_thres,
        }
        if vad_kwargs:
            self._vad_kwargs.update(vad_kwargs)
        self._builder_spec = builder_spec
        self._runner_spec = runner_spec
        self._start_timeout_s = float(start_timeout_s)
        self._ctx = mp.get_context("spawn")
        self._workers: Dict[int, _Worker] = {}
        self._tasks: Dict[int, _PendingTask] = {}
        self._queue: Deque[Tuple[int, str]] = deque()
        self._next_task_id = 0
        self._next_worker_id = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._result_queue = None
        self._reader_thread: Optional[threading.Thread] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._start_lock = asyncio.Lock()
        self._state_lock = threading.RLock()
        self._started = False
        self._stopping = False

    async def start(self) -> None:
        if self._started:
            return
        if not self._devices:
            self._refresh_vad_status()
            raise VadPoolError("qwen_asr.vad_device 不能为空，至少配置一个 VAD worker")
        async with self._start_lock:
            if self._started:
                return
            self._loop = asyncio.get_running_loop()
            self._stopping = False
            self._result_queue = self._ctx.Queue()
            self._reader_thread = threading.Thread(
                target=self._reader_loop, name="vad-pool-reader", daemon=True
            )
            self._reader_thread.start()
            logger.info("[VAD 池] 启动固定 worker：设备=%s，配置实例=%d", self._devices, len(self._devices))
            for configured_device in self._devices:
                self._spawn_worker(configured_device)
            try:
                await self._wait_core_ready()
            except Exception:
                await self.stop()
                raise
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            self._started = True
            self._refresh_vad_status()

    async def stop(self, timeout_s: float = 30.0) -> None:
        self._stopping = True
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None
        for worker in list(self._workers.values()):
            self._stop_worker(worker, timeout_s=timeout_s)
        self._workers.clear()
        if self._result_queue is not None:
            self._result_queue.close()
            self._result_queue = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout_s)
            self._reader_thread = None
        for pending in self._tasks.values():
            pending.started.set()
            if not pending.finished.done():
                pending.finished.set_exception(VadPoolError("服务停止中，VAD 请求被终止"))
        self._tasks.clear()
        self._queue.clear()
        self._started = False
        self._refresh_vad_status()

    async def detect(self, audio_path: str) -> Any:
        await self.start()
        task_id = self._next_task_id
        self._next_task_id += 1
        pending = _PendingTask(
            task_id=task_id,
            audio_path=audio_path,
            started=asyncio.Event(),
            finished=asyncio.get_running_loop().create_future(),
        )
        self._tasks[task_id] = pending
        self._queue.append((task_id, audio_path))
        self._dispatch()
        try:
            await pending.started.wait()
            return await asyncio.shield(pending.finished)
        except asyncio.CancelledError:
            await self._discard(task_id)
            raise

    def stats(self) -> Dict[str, Any]:
        workers = list(self._workers.values())
        ready = sum(1 for worker in workers if worker.ready)
        failed = [worker.failure for worker in workers if worker.failure]
        busy = sum(worker.busy_task_id is not None for worker in workers)
        return {
            "loaded": ready > 0,
            "device": self._devices[0] if self._devices else None,
            "devices": list(self._devices),
            "configured_instances": len(self._devices),
            "loaded_instances": ready,
            "available_instances": max(0, ready - busy),
            "allocated_mb": round(sum(worker.allocated_mb for worker in workers), 2),
            "failed_instances": len(failed),
            "failures": failed,
            "busy": busy,
            "waiting": len(self._queue),
        }

    def _spawn_worker(self, device: str) -> _Worker:
        with self._state_lock:
            worker_id = self._next_worker_id
            self._next_worker_id += 1
            stop_event = self._ctx.Event()
            task_queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=_worker_main,
                name=VAD_WORKER_PROCESS_NAME,
                args=(
                    worker_id,
                    self._model_dir,
                    device,
                    dict(self._vad_kwargs),
                    self._builder_spec,
                    self._runner_spec,
                    task_queue,
                    self._result_queue,
                    stop_event,
                ),
            )
            worker = _Worker(
                worker_id=worker_id,
                process=process,
                stop_event=stop_event,
                task_queue=task_queue,
                device=device,
            )
            self._workers[worker_id] = worker
            process.start()
            logger.info("[VAD 池] 拉起固定 worker#%d（设备=%s）", worker_id, device)
            return worker

    def _pick_idle_worker(self) -> Optional[_Worker]:
        idle = [
            worker
            for worker in self._workers.values()
            if worker.ready and worker.busy_task_id is None and worker.process.is_alive()
        ]
        return min(idle, key=lambda worker: worker.worker_id) if idle else None

    def _dispatch(self) -> None:
        while self._queue:
            worker = self._pick_idle_worker()
            if worker is None:
                return
            task_id, audio_path = self._queue.popleft()
            pending = self._tasks.get(task_id)
            if pending is None:
                continue
            pending.worker_id = worker.worker_id
            worker.busy_task_id = task_id
            worker.task_queue.put((task_id, audio_path))

    def _stop_worker(self, worker: _Worker, timeout_s: float = 30.0) -> None:
        self._workers.pop(worker.worker_id, None)
        try:
            worker.stop_event.set()
            worker.task_queue.close()
        except Exception:  # noqa: BLE001
            pass
        if worker.process.is_alive():
            worker.process.join(timeout=timeout_s)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=5)

    def _reap_crashed(self) -> None:
        crashed = [
            worker for worker in list(self._workers.values())
            if not worker.process.is_alive() and not worker.stop_event.is_set() and not worker.failure
        ]
        for worker in crashed:
            self._workers.pop(worker.worker_id, None)
            task_id = worker.busy_task_id
            pending = self._tasks.pop(task_id, None) if task_id is not None else None
            if pending is not None:
                pending.started.set()
                if not pending.finished.done():
                    pending.finished.set_exception(VadPoolError("VAD worker 异常退出，请重试"))
            logger.error("[VAD 池] worker#%d 异常退出（exitcode=%s），按原设备补位", worker.worker_id, worker.process.exitcode)
            if not self._stopping:
                self._spawn_worker(worker.device)
        if crashed:
            self._dispatch()
            self._refresh_vad_status()

    def _reader_loop(self) -> None:
        loop = self._loop
        while True:
            try:
                message = self._result_queue.get(timeout=0.2)
            except queue_stdlib.Empty:
                if self._stopping:
                    break
                continue
            except (EOFError, OSError):
                break
            try:
                loop.call_soon_threadsafe(self._on_message, *message)
            except RuntimeError:
                break

    def _on_message(self, kind: str, task_id: int, worker_id: int, status: str, payload: Any) -> None:
        worker = self._workers.get(worker_id)
        if kind == _MSG_IDLE:
            if worker is None:
                return
            if status != "ok":
                worker.failure = str(payload)
                logger.error("[VAD 池] worker#%d（设备=%s）预热失败：%s", worker_id, worker.device, payload)
                self._refresh_vad_status()
                return
            worker.ready = True
            if payload:
                worker.allocated_mb = float(payload.get("allocated_mb", 0.0))
            self._refresh_vad_status()
            self._dispatch()
            return
        if kind == _MSG_STARTED:
            if worker is not None:
                worker.busy_task_id = task_id
            pending = self._tasks.get(task_id)
            if pending is not None:
                pending.started.set()
            return
        if kind != _MSG_FINISHED:
            return
        if worker is not None:
            worker.busy_task_id = None
            worker.last_finished_at = time.monotonic()
        pending = self._tasks.pop(task_id, None)
        if pending is not None:
            if status == "ok":
                if not pending.finished.done():
                    pending.finished.set_result(payload)
            else:
                detail = str(payload).strip().splitlines()[-1] if payload else "未知错误"
                if not pending.finished.done():
                    pending.finished.set_exception(VadPoolError(f"VAD 切分失败：{detail}"))
        self._dispatch()
        self._refresh_vad_status()

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(_MONITOR_INTERVAL_S)
            try:
                self._reap_crashed()
            except Exception:  # noqa: BLE001
                logger.exception("[VAD 池] 池巡检异常")

    async def _wait_core_ready(self) -> None:
        deadline = time.monotonic() + self._start_timeout_s
        while True:
            workers = list(self._workers.values())
            for worker in workers:
                if not worker.process.is_alive() and not worker.ready and not worker.failure:
                    worker.failure = f"进程退出（exitcode={worker.process.exitcode}）"
            if workers and all(worker.ready or worker.failure for worker in workers):
                self._refresh_vad_status()
                if any(worker.ready for worker in workers):
                    return
                raise VadPoolError("所有 VAD worker 预热失败，请检查模型路径与设备配置")
            if time.monotonic() >= deadline:
                raise VadPoolError("VAD worker 预热超时")
            await asyncio.sleep(0.1)

    async def _discard(self, task_id: int) -> None:
        pending = self._tasks.pop(task_id, None)
        if pending is None or pending.worker_id is not None:
            return
        self._queue = deque(item for item in self._queue if item[0] != task_id)
        self._dispatch()
        self._refresh_vad_status()

    def _refresh_vad_status(self) -> None:
        workers = list(self._workers.values())
        try:
            from core.models import record_vad_runtime

            record_vad_runtime(
                device=self._devices,
                instances=sum(worker.ready for worker in workers),
                loaded=any(worker.ready for worker in workers),
                allocated_mb=sum(worker.allocated_mb for worker in workers),
                configured_instances=len(self._devices),
                successful_instances=sum(worker.ready for worker in workers),
                failed_instances=sum(bool(worker.failure) for worker in workers),
                devices=self._devices,
                busy=sum(worker.busy_task_id is not None for worker in workers),
                waiting=len(self._queue),
                failures=[worker.failure for worker in workers if worker.failure],
            )
        except Exception:  # noqa: BLE001
            logger.exception("VAD 状态登记失败")


_vad_pool: Optional[VadPool] = None


def get_vad_pool() -> VadPool:
    global _vad_pool
    if _vad_pool is None:
        _vad_pool = VadPool()
    return _vad_pool


async def start_vad_pool() -> None:
    await get_vad_pool().start()


async def stop_vad_pool() -> None:
    global _vad_pool
    if _vad_pool is not None:
        await _vad_pool.stop()
        _vad_pool = None


def reset_for_tests() -> None:
    global _vad_pool
    _vad_pool = None


__all__ = [
    "VAD_WORKER_PROCESS_NAME",
    "VadPool",
    "VadPoolError",
    "VadQueueTimeoutError",
    "build_vad_model",
    "get_vad_pool",
    "reset_for_tests",
    "run_vad_task",
    "start_vad_pool",
    "stop_vad_pool",
]
