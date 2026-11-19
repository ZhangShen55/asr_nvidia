#!/usr/bin/env python3
"""Owned offline Qwen NPU cancellation/serialization evidence; no HTTP port.

Use a benchmark_qwen_candidates JSON manifest containing one 4.1s real WAV
clip, or select --sample-id. The caller owns the hardware execution window.
Cancellation never attempts to stop a Python thread or an in-flight NPU kernel.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_qwen_candidates import (
    environment_metadata, error_details, load_samples, model_files,
    parsed_output, validate_device,
)


class Observer:
    def __init__(self):
        self.lock = threading.Lock()
        self.load_started = threading.Event()
        self.load_finished = threading.Event()
        self.call_started = threading.Event()
        self.call_finished = threading.Event()
        self.close_started = threading.Event()
        self.close_finished = threading.Event()
        self.started = self.finished = self.active = self.peak_active = 0
        self.events = []
        self.outputs = []

    def event(self, name, **details):
        with self.lock:
            self.events.append({"event": name, "monotonic_s": time.perf_counter(),
                                "thread_id": threading.get_ident(), **details})

    def begin(self):
        with self.lock:
            self.started += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            call_id = self.started
            self.events.append({"event": "call_started", "call_id": call_id,
                                "active": self.active, "monotonic_s": time.perf_counter(),
                                "thread_id": threading.get_ident()})
        self.call_started.set()
        return call_id

    def end(self, call_id, raw=None, error=None):
        with self.lock:
            self.active -= 1
            self.finished += 1
            self.events.append({"event": "call_finished_after_runtime_sync", "call_id": call_id,
                                "active": self.active, "monotonic_s": time.perf_counter(),
                                "thread_id": threading.get_ident(), "error": error})
            if raw is not None:
                self.outputs.append({"call_id": call_id, "raw": raw, "parsed": parsed_output(raw)})
        self.call_finished.set()

    def snapshot(self):
        with self.lock:
            return {"actual_started": self.started, "actual_finished": self.finished,
                    "actual_active": self.active, "peak_actual_active": self.peak_active,
                    "load_started": self.load_started.is_set(),
                    "load_finished": self.load_finished.is_set(),
                    "close_started": self.close_started.is_set(),
                    "close_finished": self.close_finished.is_set(),
                    "events": list(self.events), "outputs": list(self.outputs)}


def observed_runtime(base, observer):
    """Instrumentation only; real superclass retains loading/device/precision."""
    class ObservedRuntime(base):
        def ensure_loaded(self):
            cold = not self.loaded
            if cold:
                observer.event("cold_load_started")
                observer.load_started.set()
            try:
                return super().ensure_loaded()
            finally:
                if cold:
                    observer.event("cold_load_finished", loaded=self.loaded, failure=self.failure)
                    observer.load_finished.set()

        def transcribe(self, audio_bytes, language=None):
            if callable(getattr(base, "transcribe_batch", None)):
                return self.transcribe_batch([audio_bytes], [language])[0]
            call_id, raw, error = observer.begin(), None, None
            try:
                raw = super().transcribe(audio_bytes, language)
                return raw
            except BaseException as exc:
                error = error_details(exc)
                raise
            finally:
                # LocalQwenRuntime returns/raises only after its synchronize.
                observer.end(call_id, raw, error)

        def transcribe_batch(self, audio_list, languages=None):
            if len(audio_list) != 1:
                raise ValueError("该工具验证单实例单批取消；真正大批取消由批调度验证覆盖")
            if not callable(getattr(base, "transcribe_batch", None)):
                return [self.transcribe(audio_list[0], (languages or [None])[0])]
            call_id, raw, error = observer.begin(), None, None
            try:
                values = super().transcribe_batch(audio_list, languages)
                raw = values[0]
                return values
            except BaseException as exc:
                error = error_details(exc)
                raise
            finally:
                observer.end(call_id, raw, error)

        def close(self):
            observer.event("runtime_close_started")
            observer.close_started.set()
            try:
                return super().close()
            finally:
                observer.event("runtime_close_finished", loaded=self.loaded)
                observer.close_finished.set()
    return ObservedRuntime


async def wait_until(predicate, timeout_s, label):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"observation timeout: {label}; owned work will still be drained")
        await asyncio.sleep(.005)


async def cancelled_outcome(task):
    try:
        value = await task
    except asyncio.CancelledError:
        return {"status": "cancelled"}
    raise AssertionError(f"cancelled owner returned instead: {value!r}")


async def run_validation(sample, *, model_dir, device="npu:7", dtype="bfloat16",
                         max_new_tokens=256, request_timeout_s=240,
                         observation_timeout_s=120, runtime_base=None, report=None,
                         checkpoint=None, should_stop=None):
    from core.async_utils import await_completion
    from core.qwen_asr_client import QwenAsrClient
    if runtime_base is None:
        from core.qwen_asr_runtime import LocalQwenRuntime
        runtime_base = LocalQwenRuntime
    report = {} if report is None else report
    checkpoint = checkpoint or (lambda: None)
    should_stop = should_stop or (lambda: False)
    observer = Observer()
    client = QwenAsrClient(local_model_dir=model_dir, local_device=device, local_dtype=dtype,
                           local_instance_count=1, max_new_tokens=max_new_tokens,
                           request_timeout_s=request_timeout_s, retry_count=0,
                           max_inference_batch_size=1,
                           execution_mode="thread", runtime_factory=observed_runtime(runtime_base, observer))
    tasks = []
    report.update({"phases": [], "status": "running", "cleanup": {"status": "pending"},
                   "precision_policy": "production LocalQwenRuntime process helper; no independent HF32 override"})

    def save():
        report["observer"] = observer.snapshot()
        report["client"] = client.snapshot()
        checkpoint()

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    def start_request(name):
        task = asyncio.create_task(client.transcribe_bytes(sample.wav_bytes, sample.language,
                                                           filename=f"{sample.id}.wav"), name=name)
        tasks.append(task)
        return task

    def stop_check():
        if should_stop():
            raise InterruptedError("stop requested; draining owned work and closing model")

    try:
        save()
        stop_check()
        # Cold loading is real superclass work, not a simulated long sleep/gate.
        cold = start_request("qwen-cold-owner")
        await wait_until(observer.load_started.is_set, observation_timeout_s, "cold load started")
        require(not observer.load_finished.is_set(), "cold load finished before cancellation could be observed")
        cold.cancel()
        await asyncio.sleep(0)
        cold.cancel()
        await asyncio.sleep(0)
        before = client.snapshot()
        require(observer.snapshot()["actual_finished"] == 0, "cold work finished before busy observation")
        require(before["busy"] == 1 and not cold.done(), "cold cancellation released owner before device completion")
        phase = {"phase": "cold_load_cancel_repeated", "status": "running", "before_finish": before}
        report["phases"].append(phase)
        save()
        phase["cancel_outcome"] = await cancelled_outcome(cold)
        after = client.snapshot()
        require(observer.call_finished.is_set() and observer.snapshot()["actual_active"] == 0,
                "cancelled owner returned before actual runtime finished")
        require(after["busy"] == after["waiting"] == 0, "cold cancellation left busy/waiting capacity")
        require(observer.snapshot()["outputs"], "cancelled cold execution did not produce inspectable raw output")
        require(all(item["parsed"]["has_speech"] for item in observer.snapshot()["outputs"]),
                "cancelled cold execution raw output lacks speech/language contract")
        phase.update(status="passed", after_finish=after)
        save()
        stop_check()

        first = start_request("qwen-serial-first")
        second = start_request("qwen-serial-second")
        await wait_until(lambda: observer.snapshot()["actual_started"] >= 2 and client.snapshot()["waiting"] == 1,
                         observation_timeout_s, "same-instance request waiting")
        before = client.snapshot()
        require(before["busy"] == 1, "serial pair did not retain a busy instance")
        phase = {"phase": "two_requests_one_instance", "status": "running", "during_wait": before}
        report["phases"].append(phase)
        save()
        results = await asyncio.gather(first, second)
        outputs = [{"request": name, "raw": raw, "parsed": parsed_output(raw)}
                   for name, raw in zip(("first", "second"), results)]
        require(all(item["parsed"]["has_speech"] for item in outputs), "serial request output lacks speech/language contract")
        state = observer.snapshot()
        require(state["peak_actual_active"] == 1 and state["actual_started"] == state["actual_finished"] == 3,
                "same runtime executions overlapped or requests were lost")
        require(client.snapshot()["busy"] == client.snapshot()["waiting"] == 0,
                "serial requests did not return capacity")
        phase.update(status="passed", outputs=outputs, peak_actual_active=state["peak_actual_active"])
        save()
        stop_check()

        active = start_request("qwen-close-active")
        await wait_until(lambda: observer.snapshot()["actual_started"] == 4,
                         observation_timeout_s, "active execution before close")
        closer = asyncio.create_task(client.aclose(), name="qwen-close-owner")
        tasks.append(closer)
        await wait_until(lambda: client._close_task is not None, observation_timeout_s, "owned close task started")
        closer.cancel()
        await asyncio.sleep(0)
        closer.cancel()
        await asyncio.sleep(0)
        before = client.snapshot()
        require(observer.snapshot()["actual_finished"] == 3, "active execution finished before close cancellation observation")
        require(before["busy"] == 1 and not closer.done(), "cancelled close abandoned active owned work")
        phase = {"phase": "aclose_cancel_repeated_while_active", "status": "running", "before_finish": before}
        report["phases"].append(phase)
        save()
        raw = await active
        require(parsed_output(raw)["has_speech"], "close-time active request lacks valid speech output")
        outcome = await cancelled_outcome(closer)
        require(observer.close_finished.is_set(), "close cancellation returned before real runtime close")
        require(client._close_task.done() and client._close_task.exception() is None,
                "owned close task failed or remained pending")
        after = client.snapshot()
        require(after["busy"] == after["waiting"] == 0, "close left busy/waiting capacity")
        require(not after["loaded"], "runtime still loaded after close")
        phase.update(status="passed", after_close=after, active_raw=raw,
                     active_parsed=parsed_output(raw), cancel_outcome=outcome)
        report["status"] = "passed"
        save()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = error_details(exc)
        save()
        raise
    finally:
        try:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await await_completion(asyncio.gather(*tasks, return_exceptions=True))
            await await_completion(client.aclose())
            report["cleanup"] = {"status": "passed", "client": client.snapshot(),
                                  "actual_active": observer.snapshot()["actual_active"]}
            require(report["cleanup"]["actual_active"] == 0, "cleanup ended with active runtime")
            require(client.snapshot()["busy"] == client.snapshot()["waiting"] == 0,
                    "cleanup ended with unreleased client capacity")
        except BaseException as exc:
            report["cleanup"] = {"status": "failed", **error_details(exc)}
            report["status"] = "failed"
            raise
        finally:
            save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/home/model/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--request-timeout-s", type=float, default=240)
    parser.add_argument("--observation-timeout-s", type=float, default=120)
    args = parser.parse_args(argv)
    validate_device(args.device)
    samples = load_samples(args.manifest)
    if args.sample_id:
        samples = [sample for sample in samples if sample.id == args.sample_id]
    if len(samples) != 1:
        parser.error("select exactly one sample, using --sample-id for a multi-sample manifest")
    sample = samples[0]
    if abs(sample.seconds - 4.1) > 1 / 16000:
        parser.error("this cancellation smoke requires a 4.1s clip with explicit source boundaries")
    if args.max_new_tokens < 1 or min(args.request_timeout_s, args.observation_timeout_s) <= 0:
        parser.error("tokens/timeouts must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("use a new output path to preserve previous evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "configuration": vars(args),
              "sample": sample.metadata(), "model_files": model_files(args.model_dir),
              "declared_model_revision": args.model_revision, "environment": environment_metadata(),
              "scope": "offline owned model; no HTTP ports; no destructive injection; shared card"}

    def checkpoint():
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
        temporary.replace(output)

    stop_signals, old_handlers = [], {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, lambda number, frame: stop_signals.append(number))
    failed = False
    try:
        asyncio.run(run_validation(sample, model_dir=args.model_dir, device=args.device, dtype=args.dtype,
                                   max_new_tokens=args.max_new_tokens, request_timeout_s=args.request_timeout_s,
                                   observation_timeout_s=args.observation_timeout_s, report=report,
                                   checkpoint=checkpoint, should_stop=lambda: bool(stop_signals)))
    except BaseException as exc:
        report["fatal_error"] = error_details(exc)
        failed = True
    finally:
        report["stop_signals"] = stop_signals
        checkpoint()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({"report": str(output.resolve()), "status": report.get("status"),
                      "cleanup": report.get("cleanup"), "fatal_error": report.get("fatal_error")}))
    return 128 + stop_signals[0] if stop_signals else int(failed or report.get("status") != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
