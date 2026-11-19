"""CPU固定VAD池全语料统计；只复制完整源至本run缓存，不改源文件。"""
from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2)+"\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory_samples(inventory):
    data = json.loads(Path(inventory).read_text())
    rows = data.get("samples", data.get("files", [])) if isinstance(data, dict) else data
    selected, excluded, seen = [], [], set()
    for index, row in enumerate(rows):
        if str(row.get("format", "")).upper() != "WAV":
            excluded.append({"index": index, "reason": "not_wav", "sha256": row.get("sha256")})
            continue
        sha = row.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("WAV清单缺少SHA256")
        if sha in seen:
            excluded.append({"index": index, "reason": "duplicate_sha256", "sha256": sha})
            continue
        seen.add(sha)
        filename = os.fsdecode(base64.b64decode(row["path_bytes_b64"], validate=True)) if row.get("path_bytes_b64") else row["path"]
        duration = row.get("duration_s", row.get("audio_seconds"))
        if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("WAV duration_s必须为有限正数")
        selected.append(dict(row, path=filename, sample_id=row.get("sample_id", "audio-"+sha[:20]), audio_seconds=duration))
    if not selected:
        raise ValueError("清单无可统计的独立完整WAV")
    return selected, excluded


def prepare_copy(sample, destination):
    if digest(sample["path"]) != sample["sha256"]:
        raise ValueError("原始文件SHA与inventory不一致")
    # 完整源缓存是独立新文件，绝不写回原路径；ASCII名供FunASR/soundfile安全读取。
    import shutil
    with open(sample["path"], "rb") as source, open(destination, "xb") as target:
        shutil.copyfileobj(source, target, length=1024*1024)
    if digest(destination) != sample["sha256"]:
        raise ValueError("完整源缓存SHA不一致")


def segment_stats(raw, source_seconds, *, sample_rate=16000):
    from core.performance_watchdog import percentile
    spans = raw[0].get("value", []) if raw and isinstance(raw[0], dict) else (raw[0] if raw else [])
    valid, invalid = [], []
    for item in spans:
        try:
            start, end = float(item[0]), float(item[1])
            if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= source_seconds*1000+1:
                raise ValueError("invalid_boundary")
            valid.append([start, end])
        except (ValueError, TypeError, IndexError):
            invalid.append(repr(item))
    kept = [[start, end] for start, end in valid
            if round(end*sample_rate/1000)-round(start*sample_rate/1000) >= round(.3*sample_rate)]
    lengths = [(end-start)/1000 for start, end in kept]
    effective = sum(lengths)
    return {"raw_intervals_ms": spans, "kept_intervals_ms": kept, "invalid_spans": invalid,
            "raw_segment_count": len(valid), "short_dropped": len(valid)-len(kept),
            "segment_count": len(kept), "effective_seconds": effective,
            "speech_density": effective/source_seconds,
            "segment_duration_s": {"min": min(lengths) if lengths else None,
            "p50": percentile(lengths, .5), "p95": percentile(lengths, .95),
            "max": max(lengths) if lengths else None,
            "bins": {"0.3_to_lt_1": sum(.3-1e-9 <= x < 1 for x in lengths),
                     "1_to_lt_3": sum(1 <= x < 3 for x in lengths),
                     "3_to_lt_6": sum(3 <= x < 6 for x in lengths), "gte_6": sum(x >= 6 for x in lengths)}},
            "acoustic_language": None, "acoustic_language_scope": "not annotated; no language inference performed"}


def attach_dispatch_observer(pool):
    """公共stats缺少任务对应worker，旁路观察分派结果，不改变FIFO/执行逻辑。"""
    evidence = {"assignments": {}, "busy_peak": 0}
    if not hasattr(pool, "_dispatch"):
        return evidence
    original = pool._dispatch
    def observe():
        result = original()
        stats = pool.stats()
        evidence["busy_peak"] = max(evidence["busy_peak"], stats.get("busy", 0))
        for task_id, pending in pool._tasks.items():
            if pending.worker_id is not None and task_id not in evidence["assignments"]:
                worker = pool._workers.get(pending.worker_id)
                evidence["assignments"][task_id] = {"worker_id": pending.worker_id,
                    "worker_pid": worker.process.pid if worker else None,
                    "observed_monotonic": time.monotonic()}
        return result
    pool._dispatch = observe
    return evidence


def worker_identities(pool):
    import psutil
    identities = []
    for worker in getattr(pool, "_workers", {}).values():
        pid = worker.process.pid
        try:
            created = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        identities.append({"worker_id": worker.worker_id, "pid": pid,
                           "create_time": created, "device": worker.device})
    return identities


async def profile(pool, samples, output, args, *, timing_fn=None):
    from core.performance_watchdog import PerformanceWatchdog
    output = Path(output)
    cache = output/"owned-full-source-cache"; cache.mkdir()
    records = [dict(sample_id=s["sample_id"], sha256=s["sha256"], source=s, status="not_submitted") for s in samples]
    running, starts, cache_paths = {}, {}, set()
    observation = attach_dispatch_observer(pool)
    watchdog = PerformanceWatchdog(args.wall_timeout_s, args.no_progress_s)
    trigger = failure = None
    identities = []
    initial = {}
    began = time.monotonic()
    async def one(index):
        record, sample = records[index], samples[index]
        destination = cache/(sample["sha256"]+".wav")
        cache_paths.add(destination)
        record.update(status="copying", started_monotonic=time.monotonic())
        starts[sample["sample_id"]] = record["started_monotonic"]
        copy_task = asyncio.create_task(asyncio.to_thread(prepare_copy, sample, destination))
        try:
            # 取消时先等复制线程结束，不能让finally删文件后线程重新写出残留。
            try:
                await asyncio.shield(copy_task)
            except asyncio.CancelledError:
                while not copy_task.done():
                    try:
                        await asyncio.shield(copy_task)
                    except asyncio.CancelledError:
                        continue
                copy_task.result()
                raise
            record.update(status="detecting", copy_ended_monotonic=time.monotonic())
            task_id = getattr(pool, "_next_task_id", None)
            raw = await pool.detect(str(destination))
            record.update(segment_stats(raw, sample["audio_seconds"]), raw_vad_result=raw,
                          worker_assignment=observation["assignments"].get(task_id),
                          pool_timing=timing_fn() if timing_fn else None, status="completed")
            destination.unlink()
            cache_paths.discard(destination)
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            # pool.detect取消不会中断已启动worker；缓存留到pool.stop完成才删除。
            raise
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            record["ended_monotonic"] = time.monotonic()
            record["elapsed_s"] = record["ended_monotonic"]-record["started_monotonic"]
            starts.pop(sample["sample_id"], None)
            save(output/(sample["sha256"]+".json"), record)
    try:
        await pool.start()
        initial = pool.stats(); identities = worker_identities(pool)
        save(output/"initial-pool.json", {"stats": initial, "workers": identities})
        if initial.get("loaded_instances") != args.workers or initial.get("failed_instances", 0):
            raise RuntimeError("配置CPU worker未全部健康，不能以部分容量统计八worker结果")
        next_index = 0
        while True:
            current = pool.stats()
            if current.get("loaded_instances") != args.workers or current.get("failed_instances", 0):
                raise RuntimeError("VAD运行期间健康容量下降，停止追加而非继续声称八健康worker")
            completed = [row["elapsed_s"] for row in records if row["status"] == "completed"]
            trigger = watchdog.sample(completed, starts, tuple(row["status"] for row in records))
            if trigger:
                save(output/"diagnostic.json", {"trigger": asdict(trigger), "pool": pool.stats(), "records": records,
                                               "assignments": observation})
                break
            while len(running) < args.workers and next_index < len(samples):
                running[next_index] = asyncio.create_task(one(next_index)); next_index += 1
            if not running:
                break
            done, _ = await asyncio.wait(running.values(), timeout=args.monitor_interval_s,
                                         return_when=asyncio.FIRST_COMPLETED)
            for index, task in list(running.items()):
                if task in done:
                    task.result(); del running[index]
            observation["busy_peak"] = max(observation["busy_peak"], pool.stats().get("busy", 0))
            save(output/"progress.json", {"pool": pool.stats(), "records": records})
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for task in running.values():
            task.cancel()
        await asyncio.gather(*running.values(), return_exceptions=True)
        cleanup_error = None
        cleanup = asyncio.create_task(pool.stop(timeout_s=args.stop_timeout_s))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            cleanup.result()
        except Exception as exc:
            cleanup_error = repr(exc)
        import psutil
        survivors = []
        for identity in identities:
            try:
                process = psutil.Process(identity["pid"])
                if process.create_time() == identity["create_time"] and process.status() != psutil.STATUS_ZOMBIE:
                    survivors.append(identity)
            except psutil.NoSuchProcess:
                pass
        # 仅在owned worker均退出后回收完整源缓存；异常仍被进程持有则保留证据。
        if not survivors:
            for path in cache_paths:
                if path.is_file():
                    path.unlink()
        shutdown = {"initial_identities": identities, "remaining_owned": survivors,
                    "pool_shutdown_evidence": getattr(pool, "_shutdown_evidence", None),
                    "pool_shutdown_failures": getattr(pool, "_shutdown_failures", None),
                    "error": cleanup_error, "cache_remaining": [str(p) for p in cache.iterdir()]}
        shutdown["passed"] = not survivors and not cleanup_error and not shutdown["cache_remaining"] and not any(
            row.get("forced") or not row.get("cleanup_complete", True) for row in (shutdown["pool_shutdown_evidence"] or []))
        save(output/"shutdown.json", shutdown)
        passed = failure is None and trigger is None and shutdown["passed"] and all(row["status"] == "completed" for row in records)
        result = {"status": "diagnostic_aborted" if trigger else "completed" if passed else "incomplete", "complete": passed,
                  "configured_workers": args.workers, "initial_pool": initial, "busy_peak": observation["busy_peak"],
                  "elapsed_s": time.monotonic()-began, "trigger": asdict(trigger) if trigger else None,
                  "error": failure, "shutdown": shutdown, "records": records}
        save(output/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wall-timeout-s", type=float, default=420)
    parser.add_argument("--no-progress-s", type=float, default=120)
    parser.add_argument("--monitor-interval-s", type=float, default=1)
    parser.add_argument("--stop-timeout-s", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 0 < args.wall_timeout_s <= 420 or not all(math.isfinite(x) and x > 0 for x in (args.no_progress_s, args.monitor_interval_s, args.stop_timeout_s)):
        parser.error("workers需1～8，诊断不超过420秒，其余时间参数为有限正数")
    os.environ["CONFIG_PATH"] = str(Path(args.config).resolve())
    # spawn导入模块时也会读CONFIG_PATH，因此必须在导入任何core运行时之前设置。
    from core.vad_pool import VadPool, get_last_detect_timing
    samples, excluded = inventory_samples(args.inventory)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    save(output/"manifest.json", {"samples": samples, "excluded": excluded, "arguments": vars(args)})
    import shutil
    shutil.copy2(args.config, output/"config.toml")
    pool = VadPool(devices=["cpu"]*args.workers)
    async def run():
        task = asyncio.current_task(); loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
        return await profile(pool, samples, output, args, timing_fn=get_last_detect_timing)
    try:
        result = asyncio.run(run())
        return 0 if result["complete"] else 2
    except BaseException as exc:
        save(output/"runner-error.json", {"error": repr(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
