"""单Qwen process + parent NN16流策略；一次仅一种policy，默认只计划。"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.benchmark_npu_stream_policy import StreamPolicy, MODES, stream_identity
from scripts.benchmark_qwen_process_candidate import (
    persist, selected_rows, qwen_quality, nn_quality, resolve_core_artifact, supervise,
)


class ProcessNNPolicy(StreamPolicy):
    """仅将原实验的install收窄到NN16；原TLS同步/捕获/guard/恢复逻辑不复制。"""
    def install(self):
        try:
            expected = {f"{name}[{i}]" for name in ("speech_campplus", "emotion2vec") for i in range(8)}
            if len(self.runners) != 16 or {tag for tag, _, _ in self.runners} != expected: raise ValueError("requires exactly parent CAM8/emo8, no parent Qwen runner")
            api = self.runners[0][1].api
            if any(runner.api is not api or runner.device != "npu:7" for _, runner, _ in self.runners):
                raise ValueError("NN16 must share physical7 API")
            api.set_device("npu:7")
            if self.mode != "default_global":
                with api.device("npu:7"):
                    default = api.default_stream("npu:7")
                    streams = [api.Stream(device="npu:7") for _ in range(8)]
                if len({stream_identity(s) for s in streams}) != 8 or stream_identity(default) in {stream_identity(s) for s in streams}:
                    raise ValueError("requires eight distinct nondefault stream handles")
                for index, stream in enumerate(streams):
                    for name in ("speech_campplus", "emotion2vec"): self.streams[f"{name}[{index}]"] = stream
            for tag, runner, method in self.runners: self.attach(runner, method, tag, api)
            return self
        except BaseException:
            self.restore(); raise

    def stamp(self, api, tag, event, **extra):
        super().stamp(api, tag, event, thread_cpu_ns=time.thread_time_ns(),
                      native_thread_id=threading.get_native_id(), **extra)

    def snapshot(self):
        result = super().snapshot()
        result.update(qwen_in_parent=False, policy_scope="parent NN16 only; CAM[i]/emo[i] share pool[i] when pooled; Qwen is a separate process and unhooked",
                      cpu_timing_scope="same-thread CPU counter deltas vs wall envelope; wall-minus-CPU includes device/OS/lock waits, not a kernel attribution")
        return result


def reference_for(report, identity, ids, current=None):
    if report.get("policy") != "default_global" or report.get("status") != "completed" or report.get("cleanup", {}).get("status") != "passed":
        raise ValueError("reference must be a normally closed completed default_global job")
    supervisor = report.get("supervisor", {})
    if supervisor.get("returncode") != 0 or not supervisor.get("child_exited") or supervisor.get("cleanup_status") != "passed" or supervisor.get("term_sent") or supervisor.get("forced_kill"):
        raise ValueError("reference supervisor did not complete normal owned cleanup")
    if report.get("input_identity") != identity or report.get("selected_ids") != ids:
        raise ValueError("reference input WAV/PCM identity differs")
    if current is not None:
        for key in ("config_sha256", "manifest_sha256", "source_sha256", "segments"):
            if report.get(key) != current.get(key): raise ValueError("reference frozen condition differs: " + key)
    reference = report.get("quality_reference")
    if not isinstance(reference, dict) or not reference.get("qwen") or not reference.get("nn"):
        raise ValueError("reference evidence missing")
    return copy.deepcopy(reference)


def input_identity(rows):
    from scripts.benchmark_npu_stream_policy import pcm_identity
    return [{"id": str(row["id"]), "input_wav_sha256": row["input_wav_sha256"],
             **pcm_identity(row["pcm"])} for row in rows]


async def run_policy(rows, output, report, bindings, reference=None):
    """四轮窄编排；CPU测试替身沿用既有paired bindings形态。"""
    from core.async_utils import await_completion
    ids = [str(row["id"]) for row in rows]
    if "selected_ids" in report and report["selected_ids"] != ids:
        raise ValueError("actual selected segment IDs differ from frozen plan")
    client = policy = None; pools = {}; warm = None
    report.update(status="running", rounds=[], cleanup={"status": "pending"})
    try:
        pools = await bindings.prepare_pools()
        report["nn_loaded_status"] = bindings.pool_status()
        bindings.fence(pools)
        policy = bindings.make_policy(pools)  # before any NN forward: cold ALIBI shape stays in assigned stream.
        policy.observe_guards()
        client = bindings.make_client()
        await client.discover_model()
        report["qwen_loaded_status"] = client.snapshot()
        for index in range(4):
            phase = f"round-{index}"; policy.phase = phase
            bindings.fence(pools)
            before_cache = bindings.cache_shapes(pools)
            resource_started = time.monotonic()
            before_resources = await asyncio.to_thread(bindings.sample, True)
            resource_preparation_s = time.monotonic() - resource_started
            result = await bindings.round_fn(client, rows, bindings.gate_factory(), Path(output)/f"{phase}.json",
                warmup=index == 0, watchdog_s=report["watchdog_s"], no_progress_s=120,
                sample_interval_s=1, sampler=bindings.sample, parser=bindings.parser,
                enhanced=bindings.enhanced, cleanup_timeout_s=30)
            values = policy.phase_results(phase)
            item = {"round": index, "warmup": index == 0, "status": result.get("status"),
                    "work_done_elapsed_s": result.get("elapsed_s"), "measurement_cleanup_s": result.get("measurement_cleanup_s"),
                    "members": result.get("members", []), "original_nn_cpu_results": values,
                    "actual_batches": result.get("batches", []), "enhanced": result.get("enhanced"),
                    "client_after": client.snapshot(), "cache_before": before_cache, "cache_after": bindings.cache_shapes(pools),
                    "before_parent_resources": before_resources, "resource_preparation_s": resource_preparation_s}
            report["rounds"].append(item); persist(Path(output)/"summary.json", report)
            if item["status"] != "completed": raise RuntimeError("round failed: " + str(item["status"]))
            # Existing gates validate exact member/count/shape/labels even before external baseline exists.
            qwen_quality(None, item["members"], ids); nn_quality(None, values, rows)
            if index == 0:
                warm = {"qwen": copy.deepcopy(item["members"]), "nn": copy.deepcopy(values)}
                tags = {r["model_instance"] for r in policy.snapshot()["events"] if r["phase"] == phase and r["event"] == "call_started"}
                expected = {f"{name}[{i}]" for name in ("speech_campplus", "emotion2vec") for i in range(8)}
                if tags != expected: raise RuntimeError("not every NN instance actually forwarded in warm round")
            if reference is None and index == 1:
                reference = {"qwen": copy.deepcopy(item["members"]), "nn": copy.deepcopy(values)}
                report["quality_reference"] = copy.deepcopy(reference)
                report["cold_vs_steady"] = {"qwen": qwen_quality(reference["qwen"], warm["qwen"], ids),
                                             "nn": nn_quality(reference["nn"], warm["nn"], rows)}
                if not all(v["passed"] for v in report["cold_vs_steady"].values()): raise ValueError("cold vs steady quality mismatch")
            if reference is not None:
                item["quality"] = {"qwen": qwen_quality(reference["qwen"], item["members"], ids),
                                   "nn": nn_quality(reference["nn"], values, rows)}
                if not all(v["passed"] for v in item["quality"].values()): raise ValueError("policy numerical/text quality gate failed")
            evidence = policy.snapshot()
            if evidence["dropped_events"] or evidence["dropped_results"] or evidence.get("cpu_guard_observation", {}).get("dropped", 0):
                raise RuntimeError("bounded evidence overflow")
            persist(Path(output)/"summary.json", report)
        report.update(status="completed", quality_reference=copy.deepcopy(reference))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        errors = []
        if policy is not None:
            try: persist(Path(output)/"nn-stream-evidence.json", policy.snapshot())
            except BaseException as exc: errors.append("evidence: " + str(exc))
            finally:
                try: policy.restore()
                except BaseException as exc: errors.append("restore: " + str(exc))
        async def drain():
            try:
                if client is not None:
                    await client.aclose()
                    report["closed_qwen"] = client.snapshot()
                    if any(report["closed_qwen"].get(key) for key in ("loaded", "busy", "waiting")): raise RuntimeError("Qwen not drained")
            except BaseException as exc: errors.append("Qwen close: " + str(exc))
            finally:
                try: await bindings.close_pools()
                except BaseException as exc: errors.append("NN close: " + str(exc))
        cleanup = asyncio.create_task(drain())
        cancelled = False
        try: await await_completion(cleanup)
        except asyncio.CancelledError:
            cancelled = True
            if cleanup.cancelled(): errors.append("owned cleanup itself cancelled")
        report["cleanup"] = {"status": "failed" if errors else "passed", "errors": errors, "ports_started": False,
                             "wrappers_restored": policy is not None and not policy.originals, "cancel_received_during_cleanup": cancelled,
                             "nn_after": bindings.pool_status()}
        if errors: report["status"] = "failed"
        persist(Path(output)/"summary.json", report)
        if errors: raise RuntimeError("owned cleanup failed: " + str(errors))
        if cancelled: raise asyncio.CancelledError
    return report


def require_source_execution():
    # 仅执行实验需要source；plan仍可散列保护.so，绝不为探查加载它。
    artifacts = {name: resolve_core_artifact(ROOT, name) for name in
                 ("qwen_asr_process", "qwen_asr_client", "qwen_asr_runtime", "qwen_batch_scheduler",
                  "models", "local_inference", "model_loading", "async_utils", "config")}
    if any(path.suffix != ".py" for path in artifacts.values()):
        raise ValueError("NN stream policy execution is source-only; protected runtime validation is separate")
    return {name: str(path) for name, path in artifacts.items()}


async def execute(args, report):
    report["source_execution_artifacts"] = require_source_execution()
    os.environ["CONFIG_PATH"] = str(args.config.resolve())
    from core.config import settings
    from core.qwen_asr_client import QwenAsrClient
    from core.qwen_asr_process import LocalQwenProcessRuntime
    from core.qwen_asr_queue import RequestConcurrency
    from core.models import load_v3_models_if_needed, get_model_pool, close_v3_model_pools, local_model_status
    from scripts.benchmark_qwen_batch_curve import read_manifest, run_round, resource_sample, enhanced_workload
    from utils.qwen_response import parse_qwen_text
    rows = selected_rows(read_manifest(args.manifest), args.segments)
    identity = input_identity(rows); report["input_identity"] = identity
    reference = reference_for(json.loads(args.reference.read_text()), identity, report["selected_ids"], report) if args.reference else None
    pools = {}
    async def prepare():
        await load_v3_models_if_needed(True, True)
        pools.update({name: get_model_pool(name) for name in ("speech_campplus", "emotion2vec")})
        if any(len(pool.models) != 8 or any(m is None or m.device != "npu:7" for m in pool.models) for pool in pools.values()):
            raise ValueError("requires healthy fresh physical7 NN8+8")
        return pools
    def make_client():
        return QwenAsrClient(local_instance_count=1, execution_mode="process", runtime_factory=LocalQwenProcessRuntime,
                             retry_count=0, max_inference_batch_size=args.segments)
    def make_policy(pools):
        entries = [(f"{name}[{i}]", m, "generate") for name, pool in pools.items() for i, m in enumerate(pool.models)]
        return ProcessNNPolicy(args.policy, entries).install()
    def fence(pools):
        for api in {id(m.api): m.api for pool in pools.values() for m in pool.models}.values():
            api.set_device("npu:7"); api.synchronize("npu:7")  # TLS external: load/close/round fence stays global.
    def caches(pools):
        return {f"{name}[{i}]": {str(key): {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
                                 for key, value in getattr(m.model, "alibi_biases", {}).items()}
                for name, pool in pools.items() for i, m in enumerate(pool.models)}
    async def close_pools():
        await close_v3_model_pools()
        if any(any(m is not None for m in pool.models) for pool in pools.values()): raise RuntimeError("NN model remains loaded")
    bindings = SimpleNamespace(prepare_pools=prepare, make_client=make_client, make_policy=make_policy, fence=fence,
        close_pools=close_pools, pool_status=lambda: copy.deepcopy(local_model_status), cache_shapes=caches,
        round_fn=run_round, gate_factory=lambda: RequestConcurrency(16, 128), sample=resource_sample,
        parser=parse_qwen_text, enhanced=enhanced_workload)
    await run_policy(rows, args.output, report, bindings, reference)


def plan(args):
    try: import tomllib
    except ImportError: import tomli as tomllib
    cfg = tomllib.loads(args.config.read_text()); q = cfg.get("qwen_asr", {}); pool = cfg.get("model_device_pool", {})
    if q.get("execution_mode", "thread") != "process" or q.get("local_device") != "npu:7" or q.get("local_instance_count", 1) != 1:
        raise ValueError("requires single process Qwen on physical7")
    if q.get("local_dtype", "bfloat16") != "bfloat16": raise ValueError("experiment requires unchanged Qwen BF16")
    if any(pool.get(key) != ["npu:7"] * 8 for key in ("speech_campplus_devices", "emotion2vec_devices")):
        raise ValueError("requires parent CAM8/emo8 on physical7")
    body = json.loads(args.manifest.read_text()); rows = body.get("segments", []) if isinstance(body, dict) else body
    selected = selected_rows([dict(row, duration_s=float(row["end_s"]) - float(row["start_s"])) for row in rows], args.segments)
    if args.policy != "default_global" and args.reference is None: raise ValueError("pooled policy requires completed default_global --reference")
    sources = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ("scripts/benchmark_qwen_process_candidate.py", "scripts/benchmark_npu_stream_policy.py", "scripts/benchmark_qwen_batch_curve.py"):
        path = ROOT/name; sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for module in ("qwen_asr_process", "qwen_asr_client", "qwen_asr_runtime", "qwen_batch_scheduler", "models", "local_inference", "model_loading", "async_utils", "config"):
        path = resolve_core_artifact(ROOT, module); sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = {}
    for row in selected:
        item = coverage.setdefault(str(row["request_id"]), {"start_min_s": float("inf"), "end_max_s": 0, "count": 0})
        item["start_min_s"] = min(item["start_min_s"], float(row["start_s"])); item["end_max_s"] = max(item["end_max_s"], float(row["end_s"])); item["count"] += 1
    return {"status": "planned", "policy": args.policy, "selected_ids": [str(row["id"]) for row in selected],
            "manifest_declared_scope": body.get("scope") if isinstance(body, dict) else None, "manifest_coverage": body.get("coverage_per_source") if isinstance(body, dict) else None, "selection_coverage": coverage,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "source_sha256": sources, "segments": args.segments, "warmup_rounds": 1, "steady_rounds": 3, "watchdog_s": args.watchdog_s,
            "scope": "offline full-time selected segment subset; not all corpus shapes/contents or complete-audio throughput", "qwen_transport": "single model process, unchanged HF/BF16/global sync",
            "nn_scope": "fresh parentNN16 per job; pooled8 means CAM[i]/emo[i] share stream[i]", "ports_started": False,
            "quality_scope": "returned CPU CAM192/emo7+labels atol=rtol1e-5, Qwen raw/parsed exact; no accuracy/CPU-reference inference claim",
            "reference_path": str(args.reference) if args.reference else None,
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "output"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--policy", choices=MODES, required=True); parser.add_argument("--reference", type=Path)
    parser.add_argument("--segments", type=int, choices=(16, 128), default=128)
    parser.add_argument("--watchdog-s", type=float, default=420); parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.watchdog_s) or not 5 <= args.watchdog_s <= 420: parser.error("watchdog must be5..420")
    for name in ("config", "manifest", "output", "reference"):
        if getattr(args, name) is not None: setattr(args, name, getattr(args, name).resolve())
    expected = plan(args)
    if not args.execute: print(json.dumps(expected, ensure_ascii=True, indent=2)); return 0
    if not args.worker: args.output.mkdir(parents=True, exist_ok=False); persist(args.output/"summary.json", expected)
    if args.worker:
        report = json.loads((args.output/"summary.json").read_text())
        for key in ("source_sha256", "config_sha256", "manifest_sha256", "selected_ids", "reference_sha256"):
            if report[key] != expected[key]: raise ValueError("frozen identity changed before worker: " + key)
        async def work():
            loop = asyncio.get_running_loop(); task = asyncio.current_task()
            for number in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(number, task.cancel)
            await execute(args, report)
        try:
            asyncio.run(work())
        except BaseException as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            after = plan(args)
            keys = ("source_sha256", "config_sha256", "manifest_sha256", "selected_ids", "reference_sha256")
            report["source_changed_during_job"] = any(report[key] != after[key] for key in keys)
            if report["source_changed_during_job"]: report["status"] = "source_changed"
            persist(args.output/"summary.json", report)
        return int(report.get("status") != "completed" or report.get("cleanup", {}).get("status") != "passed")
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config), "--manifest", str(args.manifest),
               "--output", str(args.output), "--policy", args.policy, "--segments", str(args.segments), "--watchdog-s", str(args.watchdog_s), "--execute", "--worker"]
    if args.reference: command += ["--reference", str(args.reference)]
    supervisor = supervise(command, args); report = json.loads((args.output/"summary.json").read_text()); report["supervisor"] = supervisor
    if supervisor["returncode"] or supervisor["term_sent"] or supervisor["forced_kill"] or supervisor["cleanup_status"] != "passed": report["status"] = "failed"
    persist(args.output/"summary.json", report)
    return int(report.get("status") != "completed" or report.get("cleanup", {}).get("status") != "passed")


if __name__ == "__main__": raise SystemExit(main())
