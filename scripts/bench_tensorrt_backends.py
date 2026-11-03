"""隔离测试服务，比较八份完整课堂的 PyTorch / TensorRT 端到端结果。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def create_validation_app():
    """仅供此脚本启动的临时服务；预热不改变正常服务的懒加载行为。"""
    import numpy as np
    from main import create_app, lifespan as service_lifespan
    from core.async_utils import run_in_thread
    from core.local_inference import generate_local
    from core.models import get_model_pool, load_v3_models_if_needed
    from core.request_telemetry import snapshot

    @asynccontextmanager
    async def lifespan(app):
        async with service_lifespan(app):
            await load_v3_models_if_needed(True, True)
            with wave.open(os.environ["LOCAL_BACKEND_WARMUP"], "rb") as handle:
                audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
            for name in ("speech_campplus", "emotion2vec"):
                pool = get_model_pool(name)
                for _ in range(pool.configured_instances):
                    async with pool.lease() as (model, device):
                        await run_in_thread(generate_local, model, device, input=audio, fs=16000,
                                            granularity="utterance", extract_embedding=False)
            try:
                yield
            finally:
                write_json(Path(os.environ["LOCAL_BACKEND_TELEMETRY"]), snapshot())

    app = create_app()
    app.router.lifespan_context = lifespan
    return app


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def trial(args, label, samples, blobs):
    import numpy as np
    import requests

    root = Path(__file__).resolve().parents[1]
    directory = args.output / label
    directory.mkdir(parents=True, exist_ok=False)
    text = (root / "config.toml").read_text()
    replacements = {"audio_concurrency": "8", "audio_spk_emo_concurrency": "8",
                    "segment_concurrency": "16", "request_concurrency": "128",
                    "open_v2_role": "true", "open_v2_emotion": "true",
                    "speech_campplus_devices": '["cuda:0", "cuda:0", "cuda:0"]',
                    "emotion2vec_devices": '["cuda:0"]',
                    "speech_campplus_use_tensorrt": "false" if label == "pytorch" else "true",
                    "emotion2vec_use_tensorrt": "false" if label == "pytorch" else "true",
                    "speech_campplus_tensorrt_precision": '"fp32"' if label == "fp32" else '"fp16"',
                    "emotion2vec_tensorrt_precision": '"fp32"' if label == "fp32" else '"fp16"',
                    "log_path": json.dumps(str(directory / "service.log"))}
    for key, value in replacements.items():
        text, count = re.subn(rf"^{key}\s*=.*$", f"{key} = {value}", text, flags=re.M)
        if count != 1:
            raise ValueError(f"配置字段 {key} 不唯一/缺失: {count}")
    config = directory / "config.toml"
    config.write_text(text)
    env = os.environ.copy()
    env["CONFIG_PATH"] = str(config)
    env["LOCAL_BACKEND_WARMUP"] = str(args.output / "warmup.wav")
    env["LOCAL_BACKEND_TELEMETRY"] = str(directory / "telemetry.json")
    url = f"http://127.0.0.1:{args.port}"
    log = open(directory / "console.log", "w")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn",
                               "scripts.bench_tensorrt_backends:create_validation_app", "--factory",
                               "--host", "127.0.0.1", "--port", str(args.port), "--no-access-log"],
                              cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    client = requests.Session()
    client.trust_env = False
    stop = threading.Event()
    observations = []
    monitor = None
    loading_started = time.perf_counter()
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"{label} 测试服务启动失败，见 {directory / 'console.log'}")
            try:
                ready = client.get(url + "/get_status", timeout=2)
                if ready.ok:
                    break
            except requests.RequestException:
                pass
            if time.perf_counter() - loading_started > 300:
                raise TimeoutError("测试服务预热超过 300 秒")
            time.sleep(0.5)
        before = ready.json()
        write_json(directory / "status-before.json", before)
        for name, count in (("speech_campplus", 3), ("emotion2vec", 1)):
            model = before["localModels"][name]
            if model["instances"] != count or model["failed_instances"]:
                raise RuntimeError(f"{name} 未按要求就绪: {model}")
            expected = "pytorch_cuda" if label == "pytorch" else "tensorrt"
            if any(d["effective_backend"] != expected for d in model["instance_details"]):
                raise RuntimeError(f"{name} 后端未生效")
        cold_s = time.perf_counter() - loading_started
        print(label, "ready; loading/warmup", round(cold_s, 3), "s", flush=True)

        def sample_status():
            session = requests.Session()
            session.trust_env = False
            while not stop.is_set():
                row = {"at": datetime.now(timezone.utc).isoformat()}
                try:
                    row["status"] = session.get(url + "/get_status", timeout=5).json()
                    row["gpu"] = subprocess.check_output([
                        "nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                except Exception as exc:
                    row["sampling_error"] = str(exc)
                observations.append(row)
                stop.wait(2)
            session.close()

        barrier = threading.Barrier(8)

        def request(index):
            request_id = str(uuid.uuid4())
            session = requests.Session()
            session.trust_env = False
            barrier.wait(timeout=30)
            started_at = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            result = None
            error = None
            status_code = None
            try:
                response = session.post(url + "/v3/seacraft_asr",
                    files={"audioFile": (f"course-{index + 1}-{request_id}.wav", blobs[index], "audio/wav")},
                    data={"showSpk": "true", "showEmotion": "true"}, timeout=args.timeout)
                elapsed = time.perf_counter() - start
                status_code = response.status_code
                result = response.json()
                success = response.ok and isinstance(result, dict) and bool(result.get("segments")) and result.get("code") in (None, 0)
                if not success:
                    error = str(result)[:1000]
            except Exception as exc:
                elapsed = time.perf_counter() - start
                success, error = False, repr(exc)
            ended_at = datetime.now(timezone.utc).isoformat()
            session.close()
            write_json(directory / f"audio-{index + 1}-response.json", result)
            segments = (result or {}).get("segments", []) if isinstance(result, dict) else []
            row = {**samples[index], "index": index + 1, "request_id": request_id,
                   "started_at": started_at, "ended_at": ended_at, "started_monotonic": start,
                   "ended_monotonic": start + elapsed, "elapsed_s": elapsed,
                   "rtf": elapsed / samples[index]["duration_s"], "http_status": status_code,
                   "success": success, "error": error, "output_segments": len(segments),
                   "role_null": sum(s.get("role") is None for s in segments),
                   "emotion_null": sum(s.get("emotion") is None for s in segments)}
            write_json(directory / f"audio-{index + 1}-timing.json", row)
            print(label, index + 1, "success", success, "seconds", round(elapsed, 3), flush=True)
            return row

        monitor = threading.Thread(target=sample_status, daemon=True)
        monitor.start()
        with ThreadPoolExecutor(max_workers=8) as executor:
            rows = [future.result() for future in as_completed([executor.submit(request, i) for i in range(8)])]
        rows.sort(key=lambda r: r["index"])
        after = client.get(url + "/get_status", timeout=10).json()
        write_json(directory / "status-after.json", after)
        wall = max(r["ended_monotonic"] for r in rows) - min(r["started_monotonic"] for r in rows)
        success = [row for row in rows if row["success"]]
        active = [obs.get("status", {}).get("spkEmoConcurrency", {}).get("active", 0) for obs in observations]
        summary = {"backend": label, "cold_load_and_warmup_s": cold_s, "samples": rows,
                   "successful": len(success), "expected": 8, "complete": len(success) == 8,
                   "percentile_method": "linear interpolation, h=(N-1)*p",
                   "N": len(success), "batch_wall_s": wall,
                   "throughput_audio_s_per_wall_s": sum(r["duration_s"] for r in success) / wall,
                   "p50_s": float(np.percentile([r["elapsed_s"] for r in success], 50)) if success else None,
                   "p95_s": float(np.percentile([r["elapsed_s"] for r in success], 95)) if success else None,
                   "peak_active_B_sampled": max(active, default=0),
                   "models_before": before["localModels"], "models_after": after["localModels"]}
        write_json(directory / "summary.json", summary)
        print(label, "SUMMARY", json.dumps({k: v for k, v in summary.items() if k not in ("samples", "models_before", "models_after")}), flush=True)
        return summary
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=12)
        write_json(directory / "observations.json", observations)
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=55)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)
        log.close()
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--backends", nargs="+", default=["pytorch", "fp32", "fp16"],
                        choices=["pytorch", "fp32", "fp16"])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.baseline.read_text())
    samples = manifest["samples"]
    if len(samples) != 8:
        raise ValueError("必须指定八份固定完整音频")
    from core.model_assets import sha256_file

    for sample in samples:
        if sha256_file(sample["path"]) != sample["sha256"]:
            raise ValueError(f"样本哈希变化: {sample['path']}")
    blobs = [Path(sample["path"]).read_bytes() for sample in samples]
    with wave.open(samples[0]["path"], "rb") as source:
        source.setpos(60 * 16000)
        with wave.open(str(args.output / "warmup.wav"), "wb") as output:
            output.setparams(source.getparams())
            output.writeframes(source.readframes(3 * 16000))
    results = []
    for label in args.backends:
        results.append(trial(args, label, samples, blobs))
        write_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
