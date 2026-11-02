"""用全新临时服务进程验证独立回滚及故障资产，绝不修改正式模型目录。"""
import argparse
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fault_assets(source, target, fault):
    target.mkdir()
    for entry in source.iterdir():
        if entry.name != "tensorrt":
            (target / entry.name).symlink_to(entry.resolve())
    directory = target / "tensorrt" / "fp16"
    directory.mkdir(parents=True)
    (target / "tensorrt" / "sources").symlink_to((source / "tensorrt" / "sources").resolve())
    manifest = json.loads((source / "tensorrt" / "fp16" / "manifest.json").read_text())
    if fault == "incompatible":
        manifest["compute_capability"] = [0, 0]
    write_json(directory / "manifest.json", manifest)
    if fault == "corrupt":
        (directory / "engine.plan").write_bytes(b"invalid-engine")
    elif fault != "missing":
        (directory / "engine.plan").symlink_to((source / "tensorrt" / "fp16" / "engine.plan").resolve())


def trial(args, label, cam_trt, emo_trt, faulty=None):
    import requests

    root = Path(__file__).resolve().parents[1]
    directory = args.evidence / "rollback" / label
    directory.mkdir(parents=True, exist_ok=False)
    changes = {
        "speech_campplus_use_tensorrt": str(cam_trt).lower(),
        "emotion2vec_use_tensorrt": str(emo_trt).lower(),
        "speech_campplus_tensorrt_precision": '"fp16"',
        "emotion2vec_tensorrt_precision": '"fp16"',
        "speech_campplus_devices": '["cuda:0", "cuda:0", "cuda:0"]',
        "emotion2vec_devices": '["cuda:0"]',
        "open_v2_role": "true", "open_v2_emotion": "true",
        "log_path": json.dumps(str(directory / "service.log")),
    }
    if faulty:
        name, fault = faulty
        key, original = ({"speech_campplus": ("spk_model_dir", "speech_campplus_sv_zh_en_16k-common_advanced"),
                          "emotion2vec": ("emotion_model_dir", "emotion2vec_plus_large")}[name])
        target = directory / "fault-model"
        fault_assets(root / "model" / original, target, fault)
        changes[key] = json.dumps(str(target))
    text = (root / "config.toml").read_text()
    for key, value in changes.items():
        text, count = re.subn(rf"^{key}\s*=.*$", lambda _: f"{key} = {value}", text, flags=re.M)
        assert count == 1, (key, count)
    config = directory / "config.toml"
    config.write_text(text)
    env = {**os.environ, "CONFIG_PATH": str(config)}
    url = f"http://127.0.0.1:{args.port}"
    client = requests.Session()
    client.trust_env = False
    # 性能预热可以使用静音；HTTP 验收必须使用已经通过 VAD 的真实课堂语音。
    sample = json.loads((args.evidence / "admission" / "samples.json").read_text())[0]
    buffer = io.BytesIO()
    with wave.open(sample["source"], "rb") as audio:
        audio.setpos(round(sample["start_s"] * 16000))
        with wave.open(buffer, "wb") as output:
            output.setparams(audio.getparams())
            output.writeframes(audio.readframes(round(sample["duration_s"] * 16000)))
    blob = buffer.getvalue()
    with (directory / "console.log").open("w") as log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                                   "--port", str(args.port), "--no-access-log"], cwd=root, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            start = time.perf_counter()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"临时服务启动失败: {directory}")
                try:
                    status = client.get(url + "/get_status", timeout=2)
                    if status.ok:
                        break
                except requests.RequestException:
                    pass
                if time.perf_counter() - start > 180:
                    raise TimeoutError(label)
                time.sleep(0.5)
            write_json(directory / "status-before.json", status.json())

            def request(speaker, emotion, tag):
                response = client.post(url + "/v2/seacraft_asr",
                    files={"audioFile": (f"rollback-{label}-{tag}.wav", blob, "audio/wav")},
                    data={"showSpk": str(speaker).lower(), "showEmotion": str(emotion).lower()}, timeout=180)
                value = response.json()
                write_json(directory / f"{tag}-response.json", value)
                return value

            first = request(True, True, "both")
            if faulty:
                name, _ = faulty
                assert first.get("code") == (4015 if name == "speech_campplus" else 4016), first
                healthy = request(name != "speech_campplus", name != "emotion2vec", "other-model")
                assert healthy.get("segments") and not healthy.get("code"), healthy
            else:
                assert first.get("segments") and not first.get("code"), first
            after = client.get(url + "/get_status", timeout=5).json()
            write_json(directory / "status-after.json", after)
            for name, enabled in (("speech_campplus", cam_trt), ("emotion2vec", emo_trt)):
                pool = after["localModels"][name]
                if faulty and name == faulty[0]:
                    assert pool["instances"] == 0 and pool["failed_instances"] == len(pool["devices"]), pool
                    assert all(d["effective_backend"] is None and d["failure"] for d in pool["instance_details"])
                else:
                    expected = "tensorrt" if enabled else "pytorch_cuda"
                    assert pool["instances"] > 0 and not pool["failed_instances"], pool
                    assert all(d["effective_backend"] == expected for d in pool["instance_details"]), pool
                assert pool["busy"] == pool["waiting"] == 0, pool
            assert after["audioConcurrency"]["active_total"] == after["spkEmoConcurrency"]["active"] == 0
            result = {"case": label, "new_service_pid": server.pid, "cam_tensorrt": cam_trt,
                      "emo_tensorrt": emo_trt, "fault": faulty, "pass": True,
                      "first_response_code": first.get("code"), "models": after["localModels"]}
            write_json(directory / "result.json", result)
            print(label, "PASS", flush=True)
            return result
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=55)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=5)
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18086)
    args = parser.parse_args()
    args.evidence = args.evidence.resolve()
    cases = [("both-trt", True, True, None), ("rollback-cam", False, True, None),
             ("rollback-emo", True, False, None), ("both-torch", False, False, None)]
    cases += [(f"{name}-{fault}", True, True, (name, fault))
              for name in ("speech_campplus", "emotion2vec")
              for fault in ("missing", "corrupt", "incompatible")]
    results = []
    for case in cases:
        results.append(trial(args, *case))
        write_json(args.evidence / "rollback" / "summary.json", results)


if __name__ == "__main__":
    main()
