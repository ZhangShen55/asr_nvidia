"""VAD 切分的 worker 进程池。

背景：FSMN-VAD 的 ``model.generate`` 里有大量 Python / ONNX 调用，受 GIL
限制，同进程内无论加载多少实例总吞吐都不增长（实测 16 实例的吞吐与 1 实例
相当，单请求耗时却线性放大）。因此把 VAD 整体下沉到一组独立进程：

- 每个 worker 持有一份 FSMN-VAD 实例，拥有独立解释器，多份音频真正并行；
- 主进程只投递音频路径、回收分段结果，worker 全程不写文件（零拷贝）；
- 无空闲 worker 时请求进入等待队列，按等待深度自适应扩缩容。

主进程侧的所有状态变更都发生在事件循环线程里；结果队列由一个常驻线程读出，
再通过 ``call_soon_threadsafe`` 回到事件循环，避免任何阻塞调用卡住服务。
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
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

# worker 进程名：运维在进程列表里据此区分 VAD worker 与主进程
VAD_WORKER_PROCESS_NAME = "asr-offline-vad"

# worker -> 主进程的消息类型
_MSG_IDLE = "idle"
_MSG_STARTED = "started"
_MSG_FINISHED = "finished"

# worker 空闲轮询 stop_event 的间隔（秒），决定 worker 响应回收的快慢
_STOP_POLL_INTERVAL_S = 0.5
# 主进程侧监控协程的巡检间隔（秒）：回收空闲 worker、探测崩溃
_MONITOR_INTERVAL_S = 1.0
# 启动阶段等待核心 worker 预热的默认上限（秒）
_DEFAULT_STARTUP_TIMEOUT_S = 300.0


class VadPoolError(RuntimeError):
    """VAD 进程池内部错误（worker 异常退出、切分失败等）。"""

    code = 4014

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class VadQueueTimeoutError(VadPoolError):
    """请求等待 VAD worker 超时。"""

    def __init__(self, message: str = "VAD 切分排队超时，请稍后再试"):
        super().__init__(message)


def _set_process_name(name: str) -> None:
    """覆写进程名；失败只降级为原进程名，不影响 worker 工作。"""
    try:
        import setproctitle

        setproctitle.setproctitle(name)
    except Exception as exc:  # noqa: BLE001 - 进程名只是运维标识
        logger.warning(f"[VAD 池] 设置 worker 进程名失败: {exc}")


def _import_target(spec: str) -> Callable:
    """按 ``模块:函数`` 规格加载目标函数（字符串跨进程传递，无需 pickle 函数对象）。"""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"目标规格必须形如 'module:function'，实际为 {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def build_vad_model(model_dir: str, device: str, vad_kwargs: dict):
    """worker 进程内新建一份 FSMN-VAD 实例。"""
    from funasr import AutoModel

    return AutoModel(
        model=model_dir,
        device=device,
        vad_kwargs=vad_kwargs,
        disable_update=True,
        disable_pbar=True,
    )


def run_vad_task(model, audio_path: str, vad_kwargs: dict) -> Any:
    """对一个音频文件跑一次 VAD。

    funasr 的 ``FsmnVADStreaming.inference`` 使用可变默认参数 ``cache={}``，
    并发调用会共享同一份内部状态，因此每次调用都显式传入独立 cache。
    """
    return model.generate(input=audio_path, cache={}, vad_kwargs=vad_kwargs)


def _write_warmup_wav() -> str:
    """合成 0.5 秒 16 kHz 正弦波，用于触发模型首次推理路径。"""
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
    """worker 进程内自测的显存增量（MB）；CPU 上恒为 0。"""
    if not str(device).startswith("cuda"):
        return 0.0
    try:
        import torch

        return torch.cuda.memory_allocated(torch.device(device)) / (1024 ** 2)
    except Exception:  # noqa: BLE001 - 显存统计失败不影响切分
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
    """VAD worker 进程入口：预热模型后循环消费任务队列。"""
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
    except BaseException:  # noqa: BLE001 - 初始化失败要把原因带回主进程
        result_queue.put(
            (_MSG_IDLE, worker_id, worker_id, "error", traceback.format_exc())
        )
        return

    result_queue.put(
        (_MSG_IDLE, worker_id, worker_id, "ok", {"allocated_mb": allocated_mb})
    )

    while not stop_event.is_set():
        try:
            message = task_queue.get(timeout=_STOP_POLL_INTERVAL_S)
        except queue_stdlib.Empty:
            # 空闲轮询：只为响应 stop_event；主进程已知它空闲，不再重复报到
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
        except BaseException:  # noqa: BLE001 - 单次任务失败不应让 worker 自杀
            result_queue.put(
                (_MSG_FINISHED, task_id, worker_id, "error", traceback.format_exc())
            )
        result_queue.put((_MSG_IDLE, worker_id, worker_id, "ok", None))


@dataclass
class _Worker:
    """一个 worker 进程在主进程侧的登记项。"""

    worker_id: int
    process: Any
    stop_event: Any
    task_queue: Any
    is_core: bool = True
    ready: bool = False
    busy_task_id: Optional[int] = None
    last_finished_at: Optional[float] = None
    allocated_mb: float = 0.0
    created_at: float = 0.0

    @property
    def idle_since(self) -> float:
        return self.created_at if self.last_finished_at is None else self.last_finished_at


@dataclass
class _PendingTask:
    """一个在途的 VAD 请求。"""

    task_id: int
    audio_path: str
    started: asyncio.Event
    finished: asyncio.Future
    worker_id: Optional[int] = None


class VadPool:
    """常驻 + 按需扩容的 VAD worker 进程池。"""

    def __init__(
        self,
        pool_size: Optional[int] = None,
        pool_max: Optional[int] = None,
        watermark: Optional[int] = None,
        collapse_timeout_s: Optional[float] = None,
        queue_timeout_s: Optional[float] = None,
        device: Optional[str] = None,
        model_dir: Optional[str] = None,
        vad_kwargs: Optional[dict] = None,
        builder_spec: str = "core.vad_pool:build_vad_model",
        runner_spec: str = "core.vad_pool:run_vad_task",
        start_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S,
    ):
        self._pool_size = max(1, int(pool_size if pool_size is not None else settings.qwen_vad_pool_size))
        self._pool_max = max(
            self._pool_size,
            int(pool_max if pool_max is not None else settings.qwen_vad_pool_max),
        )
        self._watermark = max(0, int(watermark if watermark is not None else settings.qwen_vad_queue_watermark))
        self._collapse_timeout_s = float(
            collapse_timeout_s
            if collapse_timeout_s is not None
            else settings.qwen_vad_collapse_timeout_s
        )
        self._queue_timeout_s = float(
            queue_timeout_s if queue_timeout_s is not None else settings.qwen_vad_queue_timeout_s
        )
        self._call_timeout_s = self._queue_timeout_s
        if device is None:
            # Validate and normalize the configured VAD device before workers start.
            from core.models import resolve_vad_device

            self._device = resolve_vad_device()
        else:
            self._device = device
        self._model_dir = model_dir if model_dir is not None else settings.vad_model_dir
        default_vad_kwargs = {
            "max_single_segment_time": settings.qwen_vad_max_single_segment_time_ms,
            "max_end_silence_time": settings.qwen_vad_max_end_silence_time_ms,
            "speech_noise_thres": settings.qwen_vad_speech_noise_thres,
            "decibel_thres": settings.qwen_vad_decibel_thres,
            "snr_thres": settings.qwen_vad_snr_thres,
        }
        if vad_kwargs:
            default_vad_kwargs.update(vad_kwargs)
        self._vad_kwargs = default_vad_kwargs
        self._builder_spec = builder_spec
        self._runner_spec = runner_spec
        self._start_timeout_s = float(start_timeout_s)

        self._ctx = mp.get_context("spawn")
        self._workers: Dict[int, _Worker] = {}
        self._tasks: Dict[int, _PendingTask] = {}
        # 主进程统一派活的等待队列：(task_id, audio_path)
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

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        """拉起核心 worker 并等待全部预热完成。"""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            self._loop = asyncio.get_running_loop()
            self._result_queue = self._ctx.Queue()
            self._reader_thread = threading.Thread(
                target=self._reader_loop, name="vad-pool-reader", daemon=True
            )
            self._reader_thread.start()

            logger.info(
                "[VAD 池] 启动：设备=%s 常驻=%d 上限=%d 水位=%d 空闲回收=%0.0fs",
                self._device,
                self._pool_size,
                self._pool_max,
                self._watermark,
                self._collapse_timeout_s,
            )
            for _ in range(self._pool_size):
                self._spawn_worker(is_core=True)
            await self._wait_core_ready()

            self._monitor_task = asyncio.create_task(self._monitor_loop())
            self._started = True
            self._refresh_vad_status()

    async def stop(self, timeout_s: float = 30.0) -> None:
        """回收全部 worker 与队列资源。"""
        self._stopping = True
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None

        for worker in list(self._workers.values()):
            self._stop_worker(worker, graceful=True, timeout_s=timeout_s)
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

    # ---------- 对外接口 ----------

    async def detect(self, audio_path: str) -> Any:
        """提交一次 VAD 切分，返回 funasr 的原始输出。

        排队等待超过 ``vad_queue_timeout_s`` 抛 :class:`VadQueueTimeoutError`；
        worker 崩溃或任务异常抛 :class:`VadPoolError`。
        """
        await self.start()

        with self._state_lock:
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
        self._scale_up_if_needed()
        self._dispatch()

        try:
            await asyncio.wait_for(pending.started.wait(), timeout=self._queue_timeout_s)
        except asyncio.TimeoutError:
            await self._discard(task_id)
            raise VadQueueTimeoutError() from None

        try:
            # shield：超时时由 _discard 负责清理，不让 wait_for 直接取消内部 future
            return await asyncio.wait_for(
                asyncio.shield(pending.finished), timeout=self._call_timeout_s
            )
        except asyncio.TimeoutError:
            await self._discard(task_id)
            raise VadQueueTimeoutError("VAD 切分超时，请稍后再试") from None

    def stats(self) -> Dict[str, Any]:
        """当前池规模，供状态接口暴露。"""
        workers = list(self._workers.values())
        return {
            "device": self._device,
            "workers": len(workers),
            "busy": sum(1 for w in workers if w.busy_task_id is not None),
            "waiting": len(self._queue),
            "pool_size": self._pool_size,
            "pool_max": self._pool_max,
        }

    # ---------- worker 管理 ----------

    def _spawn_worker(self, is_core: bool) -> _Worker:
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
                    self._device,
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
                is_core=is_core,
                created_at=time.monotonic(),
                task_queue=task_queue,
            )
            self._workers[worker_id] = worker
            process.start()
            logger.info(
                "[VAD 池] 拉起%s worker#%d（当前 %d/%d 个）",
                "核心" if is_core else "扩容",
                worker_id,
                len(self._workers),
                self._pool_max,
            )
            return worker

    def _pick_idle_worker(self) -> Optional[_Worker]:
        """挑一个已预热、且当前没有在跑任务的 worker。"""
        idle = [
            w
            for w in self._workers.values()
            if w.ready and w.busy_task_id is None and w.process.is_alive()
        ]
        if not idle:
            return None
        return min(idle, key=lambda w: w.worker_id)

    def _dispatch(self) -> None:
        """把等待队列里的任务派给空闲 worker。

        派活权在主进程手上：worker 完成或预热后向主进程报到，主进程再指定它
        下一个要处理的任务。这样扩容出的新 worker 能立刻接管已排队的请求，
        worker 崩溃时主进程也准确知道该让哪个请求失败。
        """
        while self._queue:
            worker = self._pick_idle_worker()
            if worker is None:
                return
            task_id, audio_path = self._queue.popleft()
            pending = self._tasks.get(task_id)
            if pending is None:
                continue  # 已超时丢弃
            pending.worker_id = worker.worker_id
            worker.busy_task_id = task_id
            worker.task_queue.put((task_id, audio_path))

    def _stop_worker(self, worker: _Worker, graceful: bool, timeout_s: float = 30.0) -> None:
        with self._state_lock:
            self._workers.pop(worker.worker_id, None)
        try:
            worker.task_queue.close()
        except Exception:  # noqa: BLE001 - 队列关闭失败不影响回收
            pass
        if not worker.process.is_alive():
            worker.process.join(timeout=0)
            return
        if graceful:
            worker.stop_event.set()
            worker.process.join(timeout=timeout_s)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=5)

    def _scale_up_if_needed(self) -> None:
        """等待队列深度超过水位时，一次性把在途请求全部补足到 worker 上。"""
        current = len(self._workers)
        waiting = len(self._queue)
        if waiting <= self._watermark or current >= self._pool_max:
            return
        target = min(self._pool_max, current + waiting)
        for _ in range(target - current):
            self._spawn_worker(is_core=False)
        self._refresh_vad_status()

    def _collapse_idle(self) -> None:
        now = time.monotonic()
        for worker in list(self._workers.values()):
            if worker.is_core or worker.busy_task_id is not None:
                continue
            idle = now - worker.idle_since
            if idle < self._collapse_timeout_s:
                continue
            self._stop_worker(worker, graceful=True)
            logger.info(
                "[VAD 池] 空闲 %0.fs 的扩容 worker#%d 已回收，当前 %d 个",
                idle,
                worker.worker_id,
                len(self._workers),
            )
        self._refresh_vad_status()

    def _reap_crashed(self) -> None:
        """worker 非正常退出：让在途请求失败，并按原角色补位。"""
        crashed = [
            worker
            for worker in list(self._workers.values())
            if not worker.process.is_alive() and not worker.stop_event.is_set()
        ]
        for worker in crashed:
            self._workers.pop(worker.worker_id, None)
            try:
                worker.task_queue.close()
            except Exception:  # noqa: BLE001 - 队列关闭失败不影响补位
                pass
            logger.error(
                "[VAD 池] worker#%d 异常退出（exitcode=%s），在途请求失败并补位",
                worker.worker_id,
                worker.process.exitcode,
            )
            task_id = worker.busy_task_id
            pending = self._tasks.pop(task_id, None) if task_id is not None else None
            if pending is not None:
                pending.started.set()
                if not pending.finished.done():
                    pending.finished.set_exception(
                        VadPoolError("VAD worker 异常退出，请重试")
                    )
            if not self._stopping:
                try:
                    self._spawn_worker(is_core=worker.is_core)
                except Exception:  # noqa: BLE001 - 补位失败由下次巡检继续
                    logger.exception("[VAD 池] worker 补位失败")
                self._dispatch()
        if crashed:
            self._refresh_vad_status()

    # ---------- 消息与巡检 ----------

    def _reader_loop(self) -> None:
        """常驻线程：从结果队列取消息，转交事件循环。"""
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
                break  # 事件循环已关闭

    def _on_message(self, kind: str, task_id: int, worker_id: int, status: str, payload: Any) -> None:
        worker = self._workers.get(worker_id)

        if kind == _MSG_IDLE:
            if worker is None:
                return
            if status != "ok":
                logger.error("[VAD 池] worker#%d 预热失败：%s", worker_id, payload)
                return
            first_ready = not worker.ready
            worker.ready = True
            if payload:
                worker.allocated_mb = float(payload.get("allocated_mb", 0.0))
            if first_ready:
                logger.info(
                    "[VAD 池] worker#%d 预热完成（设备 %s，当前 %d 个）",
                    worker_id,
                    self._device,
                    len(self._workers),
                )
                self._refresh_vad_status()
            self._dispatch()
            return

        pending = self._tasks.get(task_id)
        if kind == _MSG_STARTED:
            if worker is not None:
                worker.busy_task_id = task_id
            if pending is None:
                return  # 已超时丢弃；worker 继续跑完，结果会被丢弃
            if not pending.started.is_set():
                pending.started.set()
            return

        if kind == _MSG_FINISHED:
            if worker is not None:
                worker.busy_task_id = None
                worker.last_finished_at = time.monotonic()
            self._tasks.pop(task_id, None)
            if status == "ok":
                if pending is not None and not pending.finished.done():
                    pending.finished.set_result(payload)
            elif pending is not None:
                detail = str(payload).strip().splitlines()[-1] if payload else "未知错误"
                if not pending.finished.done():
                    pending.finished.set_exception(VadPoolError(f"VAD 切分失败：{detail}"))
            self._dispatch()

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(_MONITOR_INTERVAL_S)
            try:
                self._reap_crashed()
                self._collapse_idle()
            except Exception:  # noqa: BLE001 - 巡检异常不能让监控协程死掉
                logger.exception("[VAD 池] 池巡检异常")

    # ---------- 内部辅助 ----------

    async def _wait_core_ready(self) -> None:
        deadline = time.monotonic() + self._start_timeout_s
        while True:
            core = [w for w in self._workers.values() if w.is_core]
            dead = [w for w in core if not w.process.is_alive()]
            if dead:
                raise VadPoolError(
                    f"VAD worker#%d 预热失败（进程已退出），请检查模型路径与设备配置"
                    % dead[0].worker_id
                )
            if core and all(w.ready for w in core):
                return
            if time.monotonic() >= deadline:
                raise VadPoolError("VAD worker 预热超时")
            await asyncio.sleep(0.1)

    async def _discard(self, task_id: int) -> None:
        pending = self._tasks.pop(task_id, None)
        if pending is None or pending.worker_id is not None:
            return  # 已派出：worker 会跑完，结果因 pending 已消失而被丢弃
        for index, (queued_id, _audio_path) in enumerate(self._queue):
            if queued_id == task_id:
                del self._queue[index]
                break

    def _refresh_vad_status(self) -> None:
        workers = list(self._workers.values())
        loaded = any(w.ready for w in workers)
        try:
            from core.models import record_vad_runtime

            record_vad_runtime(
                device=self._device,
                instances=len(workers),
                loaded=loaded,
                allocated_mb=sum(w.allocated_mb for w in workers),
                pool_size=self._pool_size,
                pool_max=self._pool_max,
            )
        except Exception:  # noqa: BLE001 - 状态登记失败不影响切分
            logger.exception("[VAD 池] 登记运行状态失败")


_vad_pool: Optional[VadPool] = None


def get_vad_pool() -> VadPool:
    """取进程池单例（惰性创建，真正的启动在 ``start_vad_pool``）。"""
    global _vad_pool
    if _vad_pool is None:
        _vad_pool = VadPool()
    return _vad_pool


async def start_vad_pool() -> None:
    """服务启动时预热全部核心 worker。"""
    await get_vad_pool().start()


async def stop_vad_pool() -> None:
    """服务停止时回收 worker 进程。"""
    global _vad_pool
    if _vad_pool is not None:
        await _vad_pool.stop()
        _vad_pool = None


def reset_for_tests() -> None:
    """测试用：清掉进程级单例。"""
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
