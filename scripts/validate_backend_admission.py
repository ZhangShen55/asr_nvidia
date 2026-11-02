"""B=4 下的 1/2/4 路短课堂请求及八路准入回归，与完整音频性能测试分列。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_tensorrt_backends import write_json


def main():
    import numpy as np
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18085)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.baseline.read_text())
    blobs, samples = [], []
    for index, source in enumerate(manifest["samples"], 1):
        response = json.loads((args.full_run / "pytorch" / f"audio-{index}-response.json").read_text())
        eligible = [s for s in response["segments"] if 2 <= float(s["ed"]) - float(s["bg"]) <= 8
                    and len(s["segment_text"].strip()) >= 4]
        segment = eligible[len(eligible) // 2]
        start = float(segment["bg"])
        duration = float(segment["ed"]) - start
        buffer = io.BytesIO()
        with wave.open(source["path"], "rb") as audio:
            audio.setpos(round(start * 16000))
            with wave.open(buffer, "wb") as output:
                output.setparams(audio.getparams())
                output.writeframes(audio.readframes(round(duration * 16000)))
        blobs.append(buffer.getvalue())
        samples.append({"source": source["path"], "start_s": start, "duration_s": duration})
    write_json(args.output / "samples.json", samples)
    all_results = []
    for backend in ("pytorch", "fp32", "fp16"):
        directory = args.output / backend
        directory.mkdir(exist_ok=False)
        config = (args.full_run / backend / "config.toml").read_text()
        config = config.replace("audio_spk_emo_concurrency = 8", "audio_spk_emo_concurrency = 4")
        config = "\n".join(f"log_path = {json.dumps(str(directory / 'service.log'))}" if line.startswith("log_path =") else line
                           for line in config.splitlines()) + "\n"
        path = directory / "config.toml"
        path.write_text(config)
        env = os.environ.copy()
        env.update(CONFIG_PATH=str(path), LOCAL_BACKEND_WARMUP=str((args.full_run / "warmup.wav").resolve()),
                   LOCAL_BACKEND_TELEMETRY=str(directory / "telemetry.json"))
        log = open(directory / "console.log", "w")
        server = subprocess.Popen([sys.executable, "-m", "uvicorn",
            "scripts.bench_tensorrt_backends:create_validation_app", "--factory", "--host", "127.0.0.1",
            "--port", str(args.port), "--no-access-log"], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        session = requests.Session()
        session.trust_env = False
        url = f"http://127.0.0.1:{args.port}"
        try:
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"启动失败: {directory}")
                try:
                    if session.get(url + "/get_status", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("预热超时")
                time.sleep(0.5)
            for count in (1, 2, 4, 8):
                barrier = threading.Barrier(count)
                observations = []
                stop = threading.Event()

                def observe():
                    while not stop.is_set():
                        try:
                            observations.append(requests.get(url + "/get_status", timeout=5).json())
                        except requests.RequestException:
                            pass
                        stop.wait(0.05)

                def request(index):
                    barrier.wait(30)
                    started = time.perf_counter()
                    response = requests.post(url + "/v2/seacraft_asr", files={"audioFile":
                        (f"short-{backend}-{count}-{index}.wav", blobs[index], "audio/wav")},
                        data={"showSpk": "true", "showEmotion": "true"}, timeout=120)
                    elapsed = time.perf_counter() - started
                    payload = response.json()
                    return {"index": index, "elapsed_s": elapsed, "code": payload.get("code"),
                            "success": bool(payload.get("segments")), "response": payload}

                monitor = threading.Thread(target=observe, daemon=True)
                monitor.start()
                try:
                    with ThreadPoolExecutor(max_workers=count) as executor:
                        rows = list(executor.map(request, range(count)))
                finally:
                    stop.set()
                    monitor.join(timeout=6)
                final = session.get(url + "/get_status", timeout=5).json()
                assert final["audioConcurrency"]["active_total"] == final["spkEmoConcurrency"]["active"] == 0
                assert all(m["busy"] == m["waiting"] == 0 for m in final["localModels"].values())
                success = [r for r in rows if r["success"]]
                rejected = [r for r in rows if r["code"] == 4017]
                if count <= 4:
                    assert len(success) == count, rows
                else:
                    assert len(success) == 4 and len(rejected) == 4, rows
                result = {"backend": backend, "B_cap": 4, "submitted": count,
                          "successful": len(success), "rejected_4017": len(rejected), "rows": rows,
                          "p50_success_s": float(np.percentile([r["elapsed_s"] for r in success], 50)),
                          "p95_success_s": float(np.percentile([r["elapsed_s"] for r in success], 95)),
                          "peak_active_B_sampled": max([d["spkEmoConcurrency"]["active"] for d in observations] + [0]),
                          "final_status": final, "observations": observations,
                          "scope": "short classroom segment requests; not full-audio end-to-end benchmark"}
                write_json(directory / f"concurrency-{count}.json", result)
                all_results.append(result)
                print(backend, count, "success", len(success), "4017", len(rejected), flush=True)
        finally:
            session.close()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(55)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(5)
            log.close()
    write_json(args.output / "summary.json", all_results)


if __name__ == "__main__":
    main()
