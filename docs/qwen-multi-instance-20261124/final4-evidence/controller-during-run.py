"""正式多实例真实HTTP验收：固定完整源、暖身分列、336秒与owned安全回收。"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import socket
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_asr_throughput import (
    OwnedService, service_command, write_json, source_identity, source_differences,
    progress_token, fields_for,
)
from core.async_utils import await_completion
from core.performance_watchdog import percentile

HARD_S = 336.0
TARGET_S = 200.0


class ExecutionProgress:
    """只认真实完成计数与已完成prefill/decode步，缓存age/耗时不构成进展。"""
    def __init__(self, started):
        self.last_progress = started
        self.high = {"controller.completed_requests": 0}
        self.step_observation = {"status": "unknown", "missing": []}

    def observe(self, status, now, completed_requests):
        counts = dict(progress_token(status))
        counts["controller.completed_requests"] = completed_requests
        missing = []
        for item in status.get("qwenModel", {}).get("instances", []):
            steps = item.get("continuous", {}).get("steps", {})
            for phase in ("prefill", "decode"):
                key = f"owner.{item.get('index')}.steps.{phase}"
                value = steps.get(phase)
                if type(value) is int and value >= 0:
                    counts[key] = value
                else:
                    missing.append(key)
        progressed = any(key in self.high and value > self.high[key] for key, value in counts.items())
        for key, value in counts.items():
            self.high[key] = max(self.high.get(key, value), value)
        if progressed:
            self.last_progress = now
        self.step_observation = {"status": "observed" if not missing and status.get("qwenModel", {}).get("instances") else "unknown", "missing": missing}

    def expired(self, now):
        return now - self.last_progress >= 120


def window_observation(status, *, owner_count=6):
    q, unknown, values = status.get("qwenModel", {}), [], []
    for item in q.get("instances", []):
        value = item.get("continuous", {}).get("active_sequences")
        if type(value) is int and value >= 0:
            values.append(value)
        else:
            unknown.append(f"owner.{item.get('index')}.active_sequences")
    running = q.get("batch_scheduler", {}).get("running_members")
    if type(running) is not int or running < 0:
        unknown.append("global.running_members")
    if not q.get("instances"):
        unknown.append("owners")
    violation = ("qwen_owner_active_limit" if any(value > 16 for value in values) else
                 "qwen_global_active_limit" if sum(values) > owner_count * 16 or type(running) is int and running > owner_count * 16 else None)
    return {"status": "unknown" if unknown else "observed", "unknown": unknown,
            "active_sequences_observed": values, "global_running": running, "violation": violation,
            "limits": {"owner_count": owner_count, "segments_per_owner": 16, "global_segments": owner_count * 16}}


def load_inputs(manifest):
    rows = json.loads(Path(manifest).read_text())
    if [r.get("sample_id") for r in rows] != [f"a{i}" for i in range(1, 7)]:
        raise ValueError("正式清单必须是原a1～a6，不截短、不自动换源")
    seen = set()
    for row in rows:
        path = os.fsdecode(base64.b64decode(row["path_bytes_b64"], validate=True))
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"] or row["sha256"] in seen:
            raise ValueError("原完整源SHA不匹配或重复")
        row["path"] = path
        seen.add(row["sha256"])
    return rows


def summarize(rows, wall_s, trigger=None, *, target_s=None):
    done = [r for r in rows if r.get("status") == "completed"]
    values = [r["elapsed_s"] for r in done]
    latency = {"n": len(values), "p50_s": percentile(values, .5), "p95_s": percentile(values, .95)}
    complete = len(done) == len(rows) and not trigger
    # 完整墙钟只使用HTTP响应接收时间；终止观察时长及取消收尾另列。
    received = [r.get("response_received") for r in rows]
    sent = [r.get("started") for r in rows]
    exact_wall = max(received) - min(sent) if complete and all(x is not None for x in received + sent) else None
    observed_wall = wall_s
    wall_s = exact_wall if exact_wall is not None else wall_s
    return {"expected": len(rows), "completed": len(done), "complete": complete, "passed": complete,
            "all_request_latency": latency if complete else None,
            "completion_subset_latency": latency, "wall_s": wall_s, "observation_wall_s": observed_wall,
            "wall_scope": "first_send_to_last_response" if exact_wall is not None else "stop_observation",
            "trigger": trigger,
            "warning_free": sum(r.get("warning_count") == 0 for r in done),
            "warning_segments": sum(r.get("warning_count", 0) for r in done),
            "discarded_segments": sum(r.get("discarded", 0) for r in done),
            "target_s": target_s,
            "target_200s_met": (complete and wall_s <= target_s) if target_s is not None else None,
            "records": rows, "hbm_free_stop_enabled": False}


class Safety:
    # HBM剩余值只记录，用户已取消提前停止线；不换一个free阈值。
    def __init__(self, *, owner_count=6):
        self.owner_count = owner_count
        self.stamp = self.swap = None
        self.swap_streak = 0

    def check(self, status):
        self.window = window_observation(status, owner_count=self.owner_count)
        if self.window["violation"]:
            return self.window["violation"]
        q = status.get("qwenModel", {})
        if q.get("failed_instances", 0) or q.get("execution_failures", 0):
            return "qwen_owner_failure"
        if status.get("vadPool", {}).get("failed_instances", 0):
            return "vad_worker_failure"
        for model in status.get("localModels", {}).values():
            if isinstance(model, dict) and model.get("failed_instances", 0):
                return "enhancement_worker_failure"
        diag = status.get("resourceDiagnostics", {})
        age = diag.get("sampling", {}).get("age_s")
        if isinstance(age, (int, float)) and age > 15:
            return None
        host = diag.get("host", {})
        if host.get("available") and host.get("available_bytes", 2**64) < 16 * 1024**3:
            return "host_available_below_16GiB"
        stamp = diag.get("timestamp")
        vm = diag.get("vmstat", {})
        swap = tuple(vm.get(k) for k in ("swap_in_bytes", "swap_out_bytes"))
        if vm.get("available") and stamp is not None and stamp != self.stamp and all(type(x) in (int, float) for x in swap):
            if self.swap is not None:
                self.swap_streak = self.swap_streak + 1 if any(a > b for a, b in zip(swap, self.swap)) else 0
            self.stamp, self.swap = stamp, swap
            if self.swap_streak >= 3:
                return "sustained_host_swap"
        return None


def covered_owners(status):
    result = {}
    for item in status.get("qwenModel", {}).get("instances", []):
        worker = item.get("model_worker", {})
        continuous = item.get("continuous", {})
        if (item.get("loaded") and not item.get("failure") and worker.get("worker_alive")
                and worker.get("ready_verified") and continuous.get("completed", 0) > 0):
            result[str(item["index"])] = {"pid": worker.get("worker_pid"),
                "birth": worker.get("worker_create_time"), "completed": continuous["completed"],
                "entry": worker.get("worker_entry_module_file"), "launch_config": worker.get("launch_config")}
    return result


def deployment_contract(config_bytes, *, instance_count=6):
    import tomli
    if type(instance_count) is not int or instance_count not in (4, 5, 6):
        raise ValueError("本控制器支持最终N4验收及原N5/N6历史对照")
    cfg = tomli.loads(config_bytes.decode())
    q, pools = cfg.get("qwen_asr", {}), cfg.get("model_device_pool", {})
    expected = {"generation_mode": q.get("generation_mode"), "execution_mode": q.get("execution_mode"),
                "N": q.get("local_instance_count"), "S": q.get("segment_concurrency"),
                "device": q.get("local_device"), "B": q.get("audio_spk_emo_concurrency", 4),
                "pools": {name: pools.get(name + "_devices") for name in ("speech_campplus", "emotion2vec")}}
    if (expected["generation_mode"], expected["execution_mode"], expected["N"], expected["S"], expected["device"], expected["B"]) != (
            "continuous", "process", instance_count, 16, "npu:7", 4):
        raise ValueError(f"控制器配置必须明确匹配continuous/process/N{instance_count}/S16/npu7/B4")
    if any(not isinstance(devices, list) or not devices for devices in expected["pools"].values()):
        raise ValueError("正式B4需明确独立增强设备数组，不能猜测池规模")
    if instance_count == 4 and (expected["pools"]["speech_campplus"] != ["npu:7"] * 4
                                or expected["pools"]["emotion2vec"] != ["npu:7"]):
        raise ValueError("最终N4验收要求CAM4/emo1，不能用较小池数替代")
    if cfg.get("instance_count", 1) != 1:
        raise ValueError("正式验收保持一个HTTP服务")
    expected["global"] = expected["N"] * expected["S"]
    return expected


def service_validation(status, expected, *, baseline=None, idle=False, resident=False):
    errors, unknown = [], []
    q = status.get("qwenModel", {})
    owners = q.get("owner_pool", {})
    checks = [(q.get("generation_mode"), "continuous", "generation_mode"),
              (q.get("execution_mode"), "process", "execution_mode"),
              (q.get("device"), expected["device"], "qwen_device"),
              (owners.get("configured"), expected["N"], "configured_owners"),
              (owners.get("healthy"), expected["N"], "healthy_owners"),
              (owners.get("segment_capacity_per_owner"), 16, "S"),
              (owners.get("configured_global_segments"), expected["global"], "global_segments"),
              (q.get("batch_scheduler", {}).get("capacity"), expected["global"], "global_scheduler"),
              (status.get("audioConcurrency", {}).get("configured_total"), expected["N"], "audio_total"),
              (status.get("spkEmoConcurrency", {}).get("configured"), 4, "B_configured"),
              (status.get("spkEmoConcurrency", {}).get("effective"), 4, "B_effective")]
    errors.extend(f"{name}: expected={want!r}, observed={got!r}" for got, want, name in checks if got != want)
    if len(q.get("instances", [])) != expected["N"]:
        errors.append(f"{expected['N']} instance records required")
    for item in q.get("instances", []):
        worker = item.get("model_worker", {})
        entry = worker.get("worker_entry_module_file")
        launch = worker.get("launch_config", {})
        if (type(worker.get("worker_pid")) is not int or type(worker.get("worker_create_time")) not in (int, float)
                or not isinstance(entry, str) or not Path(entry).resolve().is_relative_to(ROOT / "core")
                or launch.get("continuous_options", {}).get("max_active") != 16
                or launch.get("device") != expected["device"]):
            errors.append(f"owner {item.get('index')}: actual identity/import/S/device evidence missing or mismatched")
    if baseline is not None:
        actual = covered_owners(status)
        if set(actual) != set(baseline) or any((actual[k]["pid"], actual[k]["birth"]) !=
                (baseline[k]["pid"], baseline[k]["birth"]) for k in actual.keys() & baseline.keys()):
            errors.append("owner identity/real-forward coverage changed")
        peak = q.get("batch_scheduler", {}).get("peak_inflight")
        if type(peak) is not int or peak < 0:
            unknown.append("global.peak_inflight")
        elif peak > expected["global"]:
            errors.append(f"global peak_inflight exceeds {expected['global']}")
        for item in q.get("instances", []):
            peak = item.get("continuous", {}).get("peak_active")
            if type(peak) is not int or peak < 0:
                unknown.append(f"owner.{item.get('index')}.peak_active")
            elif peak > 16:
                errors.append(f"owner {item.get('index')} peak_active exceeds 16")
    if idle:
        pending = status.get("resourceDiagnostics", {}).get("segmentPipeline", {})
        values = {"audio_active": status.get("audioConcurrency", {}).get("active_total"),
                  "B_active": status.get("spkEmoConcurrency", {}).get("active"),
                  "owner_leased": owners.get("leased"), "logical_running": q.get("batch_scheduler", {}).get("running_members"),
                  "PCM_active": pending.get("active"), "PCM_bytes": pending.get("audio_bytes")}
        errors.extend(f"not_idle:{key}={value!r}" for key, value in values.items() if value != 0)
        for item in q.get("instances", []):
            active = item.get("continuous", {}).get("active_sequences")
            if active is None:
                unknown.append(f"owner.{item.get('index')}.active_sequences")
            elif active != 0:
                errors.append(f"owner {item.get('index')} active={active!r}")
    for name, devices in expected["pools"].items():
        pool = status.get("localModels", {}).get(name, {})
        if pool.get("configured_instances") != len(devices):
            errors.append(f"{name}: configured capacity changed")
        if resident and (not pool.get("loaded") or pool.get("instances") != len(devices)
                         or pool.get("called_instances") != len(devices) or pool.get("failed_instances", 0)):
            errors.append(f"{name}: residency/actual NN coverage missing")
    return {"passed": not errors and (baseline is None or not unknown), "errors": errors, "unknown": unknown}


def enhancement_validation(before, after, rows):
    errors, evidence = [], {}
    for name in ("speech_campplus", "emotion2vec"):
        old = before.get("localModels", {}).get(name, {})
        pool = after.get("localModels", {}).get(name, {})
        previous = {r["instance_index"]: r.get("calls", 0) for r in old.get("instance_metrics", [])}
        deltas = {r["instance_index"]: r.get("calls", 0) - previous.get(r["instance_index"], 0)
                  for r in pool.get("instance_metrics", [])}
        count = pool.get("configured_instances")
        if not pool.get("loaded") or pool.get("instances") != count or not count or len(deltas) != count or any(v <= 0 for v in deltas.values()):
            errors.append(f"{name}: every configured instance must execute real NN in this round")
        outcomes = {key: pool.get("inference_outcomes", {}).get(key, 0) - old.get("inference_outcomes", {}).get(key, 0)
                    for key in ("success", "empty", "failed", "cancelled")}
        if any(outcomes[k] for k in ("empty", "failed", "cancelled")) or outcomes["success"] <= 0:
            errors.append(f"{name}: NN degraded/empty/failed/cancelled or no success")
        if sum(outcomes.values()) != sum(deltas.values()):
            errors.append(f"{name}: outcomes do not conserve actual calls")
        evidence[name] = {"instance_call_deltas": deltas, "outcome_deltas": outcomes,
                          "duration_skip_delta": pool.get("duration_skips", 0) - old.get("duration_skips", 0)}
    recent = after.get("resourceDiagnostics", {}).get("requestTelemetry", {}).get("recent_stages", [])
    matched = {item.get("filename"): item for item in recent}
    for row in rows:
        stages = matched.get(row.get("upload_filename"))
        row["request_stages"] = stages
        if stages is None:
            errors.append(f"{row.get('sample_id')}: per-request stage evidence missing")
            continue
        phases = stages.get("phases", {})
        qwen_calls = phases.get("qwen_calls")
        if type(qwen_calls) not in (int, float) or qwen_calls <= 0:
            errors.append(f"{row.get('sample_id')}: qwen_calls unknown or empty")
        spk_calls, emo_calls, skips = (phases.get(key, 0) for key in ("spk_nn_calls", "emo_nn_calls", "emo_duration_skips"))
        if spk_calls != qwen_calls or emo_calls + skips != qwen_calls:
            errors.append(f"{row.get('sample_id')}: Qwen/CAM/emo+skip logical-call conservation failed")
        for prefix in ("spk", "emo"):
            calls = phases.get(f"{prefix}_nn_calls", 0)
            outcomes = sum(phases.get(f"{prefix}_nn_{key}", 0) for key in ("success", "empty", "failed", "cancelled"))
            if outcomes != calls:
                errors.append(f"{row.get('sample_id')}/{prefix}: outcomes do not conserve actual calls")
            if any(phases.get(f"{prefix}_nn_{key}", 0) for key in ("empty", "failed", "cancelled")):
                errors.append(f"{row.get('sample_id')}/{prefix}: degraded NN result")
        row["real_dual_nn_success"] = spk_calls > 0 and emo_calls > 0 and all(
            phases.get(f"{prefix}_nn_success", 0) == phases.get(f"{prefix}_nn_calls", 0) for prefix in ("spk", "emo"))
        row["emotion_duration_only"] = emo_calls == 0 and skips == qwen_calls
        if row.get("segments", 0) <= 0 or row.get("role_nonnull") != row.get("segments") or row.get("emotion_nonnull") != row.get("segments"):
            errors.append(f"{row.get('sample_id')}: real role/emotion fields not complete")
    for name, prefix in (("speech_campplus", "spk"), ("emotion2vec", "emo")):
        per_request = sum((row.get("request_stages") or {}).get("phases", {}).get(f"{prefix}_nn_calls", 0) for row in rows)
        if per_request != sum(evidence[name]["instance_call_deltas"].values()):
            errors.append(f"{name}: pool and request call evidence do not conserve")
    return {"passed": not errors, "errors": errors, "pools": evidence,
            "scope": "real NN/skip/field evidence, not human-labelled accuracy"}


def stage_evidence(before, after, rows, mode):
    recent = after.get("resourceDiagnostics", {}).get("requestTelemetry", {}).get("recent_stages", [])
    by_name = {item.get("filename"): item for item in recent}
    errors = []
    for row in rows:
        stages = by_name.get(row.get("upload_filename"))
        row["request_stages"] = stages
        if stages is None:
            errors.append(f"{row.get('sample_id')}: stage evidence missing for unique upload filename")
        elif mode == "A" and any(stages.get("phases", {}).get(key, 0)
                for key in ("spk_nn_calls", "emo_nn_calls", "spk_duration_skips", "emo_duration_skips")):
            errors.append(f"{row.get('sample_id')}: A request unexpectedly used enhancement")
    if mode == "A":
        for name in ("speech_campplus", "emotion2vec"):
            previous = before.get("localModels", {}).get(name, {}).get("calls", 0)
            current = after.get("localModels", {}).get(name, {}).get("calls", 0)
            if previous != current:
                errors.append(f"{name}: pure A batch changed actual NN calls")
    return {"passed": not errors, "errors": errors}


async def get_status(client):
    response = await client.get("/get_status", timeout=5)
    response.raise_for_status()
    return response.json()


async def batch(client, samples, mode, output, owner, *, hard_s=HARD_S, target_s=None, owner_count=6):
    output.mkdir(exist_ok=False)
    if not 0 < hard_s <= HARD_S:
        raise ValueError("不能放宽336秒")
    rows = [dict(sample_id=s["sample_id"], sha256=s.get("sha256"), status="not_submitted") for s in samples]
    safety = Safety(owner_count=owner_count)
    started = time.monotonic()
    progress = ExecutionProgress(started)
    trigger, snapshot = None, {}
    prefix = uuid.uuid4().hex

    async def request(index):
        row, sample = rows[index], samples[index]
        row.update(status="running", upload_filename=f"{prefix}-{index}.wav")
        try:
            with open(sample["path"], "rb") as stream:
                row["started"] = time.monotonic()
                response = await client.post("/v3/seacraft_asr", files={"audioFile": (row["upload_filename"], stream)}, data=fields_for(mode, index))
                row["response_received"] = time.monotonic()
                row["elapsed_s"] = row["response_received"] - row["started"]
            row["http_status"] = response.status_code
            response.raise_for_status()
            value = response.json()
            write_json(output / f"response-{sample['sample_id']}.json", value)
            row["code"] = value.get("code")
            warnings, state = value.get("segment_warnings"), value.get("processing_status")
            valid = (row["code"] in (None, 0) and isinstance(value.get("segments"), list)
                     and isinstance(warnings, list) and state in ("complete", "completed_with_warnings")
                     and bool(warnings) == (state == "completed_with_warnings"))
            row["status"] = "completed" if valid else "failed"
            if isinstance(warnings, list):
                row.update(warning_count=len(warnings), discarded=sum(isinstance(w, dict) and w.get("action") == "discard_unrecoverable" for w in warnings))
            segments = value.get("segments") if isinstance(value.get("segments"), list) else []
            row.update(processing_status=state, segments=len(segments),
                       role_nonnull=sum(s.get("role") is not None for s in segments),
                       emotion_nonnull=sum(s.get("emotion") is not None for s in segments))
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            raise
        except Exception as exc:
            row.update(status="failed", error=repr(exc))
        finally:
            if "elapsed_s" not in row:
                row["observed_elapsed_s"] = time.monotonic() - row.get("started", started)
            write_json(output / f"request-{sample['sample_id']}.json", row)

    tasks = [asyncio.create_task(request(i)) for i in range(len(samples))]
    status_task = None
    try:
        while not all(t.done() for t in tasks):
            owner.register()
            now = time.monotonic()
            if owner.process.poll() is not None:
                trigger = "owned_service_exited"
            elif now - started >= hard_s:
                trigger = "hard_336s"
            elif any(r["status"] == "failed" for r in rows):
                trigger = "request_failed"
            if trigger:
                break
            if status_task is not None and status_task.done():
                try:
                    snapshot = status_task.result()
                    progress.observe(snapshot, now, sum(t.done() for t in tasks))
                    trigger = safety.check(snapshot)
                except Exception as exc:
                    snapshot = {"snapshot_error": repr(exc)}
                with (output / "status.jsonl").open("a") as stream:
                    stream.write(json.dumps({"elapsed_s": now - started, "status": snapshot,
                        "step_progress_observation": progress.step_observation,
                        "window_observation": window_observation(snapshot, owner_count=owner_count)}, ensure_ascii=True) + "\n")
                status_task = None
                if trigger:
                    break
            if progress.expired(now):
                trigger = "no_progress_120s"
                break
            if status_task is None:
                status_task = asyncio.create_task(get_status(client))
            pending = [task for task in tasks if not task.done()]
            if not pending:
                break
            await asyncio.wait(pending, timeout=min(.5, max(.001, hard_s - (time.monotonic() - started))))
        if not trigger and any(r["status"] != "completed" for r in rows):
            trigger = "request_failed"
        if not trigger and any(r.get("response_received", started) - started > hard_s for r in rows):
            trigger = "hard_336s"
    finally:
        ended = time.monotonic()
        for task in tasks:
            if not task.done():
                task.cancel()
        if status_task:
            status_task.cancel()
        await await_completion(asyncio.gather(*tasks, *([status_task] if status_task else []), return_exceptions=True))
        summary = summarize(rows, ended - started, trigger, target_s=target_s)
        write_json(output / "summary.json", summary)
        write_json(output / "last-status.json", snapshot)
    return summary


async def admission_probes(client, clip, output, owner, *, owner_count=6):
    """仅真实8.01秒HTTP请求创造活动窗口；窗口未观测就报inconclusive，不fake hold。"""
    output.mkdir(exist_ok=False)
    prefix, rows, tasks = uuid.uuid4().hex, {}, []
    started, gate = time.monotonic(), Safety(owner_count=owner_count)
    observed = []
    async def submit(label, mode):
        row = rows[label] = {"mode": mode, "started": time.monotonic()}
        with open(clip, "rb") as stream:
            response = await client.post("/v3/seacraft_asr", files={"audioFile": (f"probe-{prefix}-{label}.wav", stream)}, data=fields_for(mode, 0))
        row.update(response_received=time.monotonic(), http_status=response.status_code)
        response.raise_for_status()
        value = response.json()
        row["code"] = value.get("code")
        row["business_complete"] = (row["code"] in (None, 0) and isinstance(value.get("segments"), list)
                                    and value.get("processing_status") in ("complete", "completed_with_warnings"))
        write_json(output / f"response-{label}.json", value)
        return row
    def launch(label, mode):
        task = asyncio.create_task(submit(label, mode))
        tasks.append(task)
        return task
    async def active(total, enhanced):
        deadline = min(started + HARD_S, time.monotonic() + 15)
        while time.monotonic() < deadline:
            owner.register()
            if owner.process.poll() is not None:
                raise RuntimeError("owned_service_exited")
            status = await get_status(client)
            failure = gate.check(status)
            if failure:
                raise RuntimeError(failure)
            observed.append({"time": time.monotonic(), "audio": status.get("audioConcurrency"), "B": status.get("spkEmoConcurrency")})
            if (status.get("audioConcurrency", {}).get("active_total"), status.get("spkEmoConcurrency", {}).get("active")) == (total, enhanced):
                return True
            if tasks and all(task.done() for task in tasks):
                return False
            await asyncio.sleep(.05)
        return False
    result = {"passed": False, "status": "inconclusive", "scope": "real HTTP, observed active windows; no artificial hold"}
    try:
        for i in range(4):
            launch(f"B{i}", "B")
        if not await active(4, 4):
            result["reason"] = "4B_activity_window_not_observed"
            return result
        fifth = await asyncio.wait_for(launch("fifthB", "B"), min(15, HARD_S - (time.monotonic() - started)))
        # N4/B4的总容量与B上限相同，总满4013优先，单独4017在此配置不可达。
        expected_fifth = 4013 if owner_count == 4 else 4017
        if fifth["code"] != expected_fifth:
            result["reason"] = f"fifthB_not_{expected_fifth}; capacity may have changed during HTTP admission"
            return result
        if not await active(4, 4):
            result["reason"] = "4B_window_released_before_A"
            return result
        for i in range(owner_count - 4):
            launch(f"A{i}", "A")
        if not await active(owner_count, 4):
            result["reason"] = f"4B_plus_{owner_count - 4}A_activity_window_not_observed"
            return result
        rejected = await asyncio.wait_for(asyncio.gather(launch("fullA", "A"), launch("fullB", "B")),
                                           min(15, HARD_S - (time.monotonic() - started)))
        if [row["code"] for row in rejected] != [4013, 4013]:
            result["reason"] = "total_full_rejection_not_observed; capacity may have changed during HTTP admission"
            return result
        while not all(task.done() for task in tasks):
            if time.monotonic() - started >= HARD_S:
                raise TimeoutError("hard_336s")
            owner.register()
            if owner.process.poll() is not None:
                raise RuntimeError("owned_service_exited")
            failure = gate.check(await get_status(client))
            if failure:
                raise RuntimeError(failure)
            pending = [task for task in tasks if not task.done()]
            if not pending:
                break
            await asyncio.wait(pending, timeout=.1)
        await asyncio.gather(*tasks)
        accepted_ok = all(rows[label].get("business_complete") for label in [*(f"B{i}" for i in range(4)), *(f"A{i}" for i in range(owner_count - 4))])
        result.update(passed=accepted_ok, status="passed" if accepted_ok else "failed", reason=None if accepted_ok else "accepted_request_failed",
                      owner_count=owner_count, accepted_A_count=owner_count - 4, fifth_B_expected_code=expected_fifth,
                      B_only_4017_reachable=owner_count > 4)
    except asyncio.CancelledError:
        result.update(status="cancelled", reason="controller_cancelled")
        raise
    except Exception as exc:
        result.update(status="failed", reason=repr(exc))
    finally:
        # 仍在途时取消客户端只代表终止观察；execute随后停止服务并等待设备安全收尾。
        for task in tasks:
            if not task.done():
                task.cancel()
        await await_completion(asyncio.gather(*tasks, return_exceptions=True))
        result.update(records=rows, observations=observed, elapsed_s=time.monotonic() - started)
        write_json(output / "summary.json", result)
    return result


def namespace_listening_ports(pid, *, proc_root=Path("/proc")):
    # psutil可能只看调用方netns；按目标进程fd inode核对其自己的tcp表，
    # 不进入/修改命名空间，不把同netns其它进程的监听误算为目标服务。
    process = Path(proc_root) / str(pid)
    inodes = set()
    for fd in (process / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[8:-1])
    ports = set()
    for name, family in (("tcp", socket.AF_INET), ("tcp6", socket.AF_INET6)):
        try:
            lines = (process / "net" / name).read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            values = line.split()
            if len(values) < 10 or values[3] != "0A" or values[9] not in inodes:
                continue
            address, port = values[1].split(":")
            raw = bytes.fromhex(address)
            raw = (raw[::-1] if family == socket.AF_INET else
                   b"".join(raw[i:i+4][::-1] for i in range(0, len(raw), 4)))
            ports.add((socket.inet_ntop(family, raw), int(port, 16)))
    return sorted(ports)


def shared_identity(pid=1601397):
    import psutil
    try:
        process = psutil.Process(pid)
        connections = getattr(process, "net_connections", None) or process.connections
        ports = sorted({(item.laddr.ip, item.laddr.port) for item in connections(kind="inet")
                        if item.status == psutil.CONN_LISTEN})
        scope = "psutil process socket ownership"
        if not ports:
            ports = namespace_listening_ports(pid)
            scope = "target /proc/PID/net tcp tables filtered by owned fd socket inodes"
        return {"available": True, "pid": pid, "birth": process.create_time(), "listening_ports": ports,
                "alive": process.is_running() and process.status() != psutil.STATUS_ZOMBIE,
                "listening_scope": scope}
    except (psutil.Error, OSError) as exc:
        return {"available": False, "pid": pid, "reason": repr(exc)}


async def background_snapshot():
    from core.resource_diagnostics import collect_snapshot
    try:
        return await await_completion(asyncio.to_thread(collect_snapshot, include_health=False, include_process_tree=False))
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}


def device7_hbm(snapshot):
    """仅识别物理7号卡的真实MB值；其它卡可用不能替代本卡证据。"""
    npu = snapshot.get("npu", {})
    rows = [row for row in npu.get("npus", []) if row.get("index") == 7]
    result = {"index": 7, "available": False, "status": "unknown"}
    if npu.get("available") is not True or len(rows) != 1 or rows[0].get("available") is not True:
        return dict(result, reason="physical NPU7 record missing, ambiguous or unavailable")
    row = rows[0]
    keys = ("hbm_used_mb", "hbm_total_mb", "hbm_free_mb")
    if not all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) for key in keys):
        return dict(result, reason="physical NPU7 requires finite numeric used/total/free MB")
    used, total, free = (row[key] for key in keys)
    if total <= 0 or not 0 <= used <= total or not 0 <= free <= total or not math.isclose(used + free, total, abs_tol=.001):
        return dict(result, reason="physical NPU7 HBM values are inconsistent")
    return {"index": 7, "available": True, "status": "observed", **{key: row[key] for key in keys},
            "chip": row.get("chip"), "bus_id": row.get("bus_id")}


def hbm_recovery_evidence(before, after, *, owned_pids=(), owned_normal_exit=None):
    old, new = device7_hbm(before), device7_hbm(after)
    observed = old["available"] and new["available"]
    same_device = observed and all(old.get(key) == new.get(key) for key in ("chip", "bus_id", "hbm_total_mb"))
    valid = bool(observed and same_device)
    used_delta = new["hbm_used_mb"] - old["hbm_used_mb"] if valid else None
    external = []
    def apps(snapshot):
        return {(row.get("pid"), row.get("chip")): row.get("memory_mb")
                for row in snapshot.get("npu", {}).get("compute_apps", [])
                if row.get("index") == 7 and row.get("pid") not in owned_pids}
    old_apps, new_apps = apps(before), apps(after)
    for identity in sorted(old_apps.keys() | new_apps.keys(), key=str):
        previous, current = old_apps.get(identity), new_apps.get(identity)
        known = all(type(value) in (int, float) and math.isfinite(value) for value in (previous, current))
        external.append({"pid": identity[0], "chip": identity[1], "before_mb": previous, "after_mb": current,
                         "delta_mb": current - previous if known else None})
    # 数值相同且全部owned正常退出时可以确认回归；非零变化留给真实共享负载归因，
    # 不把采样成功当恢复成功，也不发明一个“相近”的free阈值。
    recovered = True if valid and owned_normal_exit is True and used_delta == 0 else False if owned_normal_exit is False else None
    return {"passed": recovered, "observation_complete": valid, "hbm_observed": valid,
            "status": "recovered" if recovered is True else "failed" if recovered is False else "unknown",
            "before": old, "after": new,
            "used_delta_mb": used_delta,
            "free_delta_mb": new["hbm_free_mb"] - old["hbm_free_mb"] if valid else None,
            "external_process_changes": external, "owned_normal_exit": owned_normal_exit,
            "near_background": recovered,
            "scope": "physical NPU7 actual MB evidence plus owned exit; nonzero aggregate delta requires shared-load review, no replacement free threshold",
            "reason": None if recovered is True else "recovery not verified: inspect NPU7 numeric evidence, external changes and owned cleanup; unknown is not zero"}


def integrity_result(before, after, original_config, final_config, shared_before, shared_after):
    changes = source_differences(before, after)
    shared_ok = (shared_before.get("available") and shared_after.get("available") and shared_after.get("alive")
                 and shared_before.get("birth") == shared_after.get("birth")
                 and bool(shared_before.get("listening_ports"))
                 and shared_before.get("listening_ports") == shared_after.get("listening_ports"))
    result = {"source_changes": changes, "config_changed": original_config != final_config,
              "source_identity_complete": before.get("complete") is True and after.get("complete") is True,
              "shared_service_preserved": bool(shared_ok), "shared_before": shared_before, "shared_after": shared_after}
    result["passed"] = not changes and not result["config_changed"] and result["source_identity_complete"] and bool(shared_ok)
    return result


def phase_plan(instance_count, phases=None):
    if type(instance_count) is not int or instance_count not in (4, 5, 6):
        raise ValueError("only final N4 or explicitly selected historical N5/N6 matrix is supported")
    primary = f"A{instance_count}"
    resident = primary + "-resident"
    phases = list(phases) if phases is not None else ([primary, "B4", resident] if instance_count == 4 else [primary, "A4", "B4", resident])
    if (not phases or len(set(phases)) != len(phases)
            or any(phase not in (primary, "A4", "B4", resident) for phase in phases)
            or (resident in phases and ("B4" not in phases or phases.index("B4") > phases.index(resident)))):
        raise ValueError("阶段必须匹配实例数，且共驻复测必须在同服务B4之后")
    return phases


async def execute(args):
    import httpx
    instance_count = getattr(args, "instance_count", 6)
    phases = phase_plan(instance_count, args.phases)
    primary, resident = f"A{instance_count}", f"A{instance_count}-resident"
    out = Path(args.output).resolve()
    out.mkdir(exist_ok=False)
    config = Path(args.config).resolve()
    original_config = config.read_bytes()
    expected = deployment_contract(original_config, instance_count=instance_count)
    before = source_identity()
    controller_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out / "source-before.json", before)
    (out / "config.toml").write_bytes(original_config)
    samples = load_inputs(args.manifest)
    write_json(out / "inputs.json", samples)
    shared_before = shared_identity()
    background_before = await background_snapshot()
    write_json(out / "background-before.json", background_before)
    env = dict(os.environ, CONFIG_PATH=str(config), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               ASR_STATS_PATH=str(out / "private-stats.json"), PYTHONDONTWRITEBYTECODE="1")
    owner, reports = None, {}
    async def finalize():
        # 整个finally由独立owned任务看护，重复cancel不能在close后跳过证据/完整性检查。
        if owner is not None:
            try:
                reports["shutdown"] = await owner.close(deadline_s=210, observe_s=5)
            except BaseException as exc:
                reports["shutdown"] = {"passed": False, "error": repr(exc)}
        after = source_identity()
        write_json(out / "source-after.json", after)
        try:
            final_config = config.read_bytes()
        except OSError:
            final_config = None
        integrity = integrity_result(before, after, original_config, final_config, shared_before, shared_identity())
        integrity["controller_sha256"] = controller_sha
        integrity["controller_unchanged"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == controller_sha
        integrity["passed"] = integrity["passed"] and integrity["controller_unchanged"]
        reports["integrity"] = integrity
        background_after = await background_snapshot()
        write_json(out / "background-after.json", background_after)
        reports["resource_recovery"] = hbm_recovery_evidence(background_before, background_after,
            owned_pids=getattr(owner, "identities", {}), owned_normal_exit=reports.get("shutdown", {}).get("passed"))
        hbm_observed = reports["resource_recovery"]["hbm_observed"]
        formal = (instance_count == 6 and all(reports.get(name, {}).get("passed", False) for name in ("A6", "A4", "B4", "A6-resident", "admission_probes", "shutdown", "integrity", "resource_recovery")))
        requested_complete = all(reports.get(name, {}).get("passed", False) for name in ["warmup", *phases, "shutdown", "integrity"])
        final_n4_functional = (instance_count == 4 and all(reports.get(name, {}).get("passed", False)
            for name in ("warmup", "A4", "B4", "A4-resident", "admission_probes", "shutdown", "integrity")))
        write_json(out / "run.json", {"phases": reports, "integrity": integrity,
                   "requested_phases": phases, "hbm_free_stop_enabled": False,
                   "admission_probes_requested": not getattr(args, "skip_admission_probes", False),
                   "formal_matrix_complete": formal, "requested_phases_complete": requested_complete,
                   "instance_count": instance_count, "deployment_contract": expected,
                   "final_n4_functional_matrix_complete": final_n4_functional,
                   "test_scope": ("user-adopted final N4/CAM4/emo1 acceptance" if instance_count == 4 else "original N6 acceptance" if instance_count == 6 else "user-authorized N5 follow-up; does not satisfy original N6 acceptance"),
                   "target_s": TARGET_S, "hard_s": HARD_S,
                   "background_hbm_scope": "physical NPU7 before/after actual MB values; shared load may change, no replacement free threshold",
                   "background_hbm_observed": hbm_observed})
    try:
        if not shared_before.get("available") or not shared_before.get("alive") or not shared_before.get("listening_ports"):
            raise RuntimeError("共享服务PID/birth/原端口未可核验，不能启动压力")
        if not before.get("complete"):
            raise RuntimeError("源码身份快照不完整")
        # 配置仍为正式原件；仅相对运行产物落入独立cwd，避免日志保留策略删除历史。
        service_cwd = out / "service-cwd"
        service_cwd.mkdir()
        command = service_command(18083, 30)
        command[1] = str(ROOT / "main.py")
        owner = OwnedService(command, env, "127.0.0.1", 18083, out, cwd=service_cwd)
        write_json(out / "service-launch.json", {"command": command, "cwd": str(service_cwd),
                   "CONFIG_PATH": str(config), "runtime_configuration_changed": False})
        async with httpx.AsyncClient(base_url="http://127.0.0.1:18083", timeout=None, trust_env=False) as client:
            deadline = time.monotonic() + 420
            while True:
                owner.register()
                if owner.process.poll() is not None:
                    raise RuntimeError("正式服务启动失败，查看service.log")
                try:
                    status = await get_status(client)
                    if status.get("qwenModel", {}).get("capacity") == instance_count:
                        break
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{instance_count} owner启动未在420秒内就绪")
                await asyncio.sleep(1)
            write_json(out / "startup-status.json", status)
            check = service_validation(status, expected, idle=True)
            if not check["passed"]:
                raise RuntimeError("正式配置/初态未通过: " + repr(check))
            clip = str(Path(args.warmup_clip).resolve())
            warm = [{"sample_id": f"warm{i}", "path": clip} for i in range(instance_count)]
            reports["warmup"] = await batch(client, warm, "A", out / "warmup", owner, owner_count=instance_count)
            if not reports["warmup"]["complete"]:
                return reports
            status = await get_status(client)
            covered = covered_owners(status)
            write_json(out / "warmup-final-status.json", status)
            write_json(out / "owner-coverage.json", covered)
            if len(covered) != instance_count or not service_validation(status, expected, baseline=covered, idle=True)["passed"]:
                raise RuntimeError(f"暖身未证明{instance_count} owner真实执行及归零")
            for phase in phases:
                pre = await get_status(client)
                check = service_validation(pre, expected, baseline=covered, idle=True, resident=phase == resident)
                write_json(out / f"{phase}-pre-status.json", pre)
                if not check["passed"]:
                    reports[phase] = {"passed": False, "complete": False, "validation": check}
                    break
                if phase == "B4":
                    cold_before = pre
                    warmb = await batch(client, warm[:4], "B", out / "B-cold-warmup", owner, owner_count=instance_count)
                    pre = await get_status(client)
                    warmb["enhancement_validation"] = enhancement_validation(cold_before, pre, warmb["records"])
                    warmb["passed"] = warmb["complete"] and warmb["enhancement_validation"]["passed"]
                    reports["B-cold-warmup"] = warmb
                    write_json(out / "B-warm-status.json", pre)
                    write_json(out / "B-cold-warmup/summary.json", warmb)
                    if not warmb["passed"]:
                        break
                chosen = samples[:instance_count] if phase in (primary, resident) else samples[:4]
                report = await batch(client, chosen, "B" if phase == "B4" else "A", out / phase, owner,
                                     target_s=TARGET_S if phase in (primary, resident) else None, owner_count=instance_count)
                post = await get_status(client)
                report["validation"] = service_validation(post, expected, baseline=covered, idle=True,
                                                          resident=phase in ("B4", resident))
                if phase == "B4":
                    report["enhancement_validation"] = enhancement_validation(pre, post, report["records"])
                report["stage_validation"] = stage_evidence(pre, post, report["records"], "B" if phase == "B4" else "A")
                report["passed"] = (report["complete"] and report["validation"]["passed"] and report["stage_validation"]["passed"]
                                    and report.get("enhancement_validation", {"passed": True})["passed"])
                reports[phase] = report
                write_json(out / phase / "summary.json", report)
                write_json(out / f"{phase}-final-status.json", post)
                write_json(out / "reports.json", reports)
                if not report["passed"]:
                    break
                if phase == "B4" and not getattr(args, "skip_admission_probes", False):
                    reports["admission_probes"] = await admission_probes(client, clip, out / "admission-probes", owner, owner_count=instance_count)
                    if not reports["admission_probes"]["passed"]:
                        break
    except BaseException as exc:
        reports["execution_error"] = {"passed": False, "error": repr(exc)}
        raise
    finally:
        await await_completion(finalize())
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "openspec/changes/2026-11-23-qwen-multi-instance-serving/test-inputs.json"))
    parser.add_argument("--config", default=str(ROOT / "config.toml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-clip", default="/tmp/qc-warmup.wav")
    parser.add_argument("--instance-count", type=int, choices=(4, 5, 6), default=4, help="最终N4/CAM4/emo1验收；5/6仅明确选择的历史对照")
    parser.add_argument("--phases", nargs="+", choices=("A5", "A6", "A4", "B4", "A4-resident", "A5-resident", "A6-resident"), default=None)
    parser.add_argument("--skip-admission-probes", action="store_true", help="仅用于独立标定轮；不代表准入探针通过")
    args = parser.parse_args()
    result = asyncio.run(execute(args))
    return 0 if all(r.get("passed", r.get("complete", False)) for r in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
