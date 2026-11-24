"""源码镜像交付契约；真实设备/完整音频验收由独立部署报告记录。"""
from pathlib import Path
import json
import os
import re
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"


def test_source_delivery_isolated_environment_and_minimal_assets():
    text = (DOCKER / "Dockerfile").read_text()
    assert "cann:8.2.rc1-910b-ubuntu22.04-py3.10" in text
    assert "conda-unpack" in text and "/opt/asr/bin/python" in text
    assert "2.5.1.post1" in text
    assert "ASR_CONTAINER_RUNTIME=cann82-source" in text
    assert "PYTHONNOUSERSITE=1" in text
    copied = [line for line in text.splitlines() if line.startswith(("COPY ", "ADD "))]
    assert not any(re.search(r"(?:^|\s)(?:\.\.?/?)\s", line) for line in copied)
    assert not any(re.search(r"(?:tests?|docs|scripts|openspec|config\.toml|\.so)(?:/|\s)", line) for line in copied)
    for model in ("speech_fsmn_vad_zh-cn-16k-common-pytorch", "speech_campplus_sv_zh_en_16k-common_advanced", "emotion2vec_plus_large"):
        assert any("--from=model_assets" in line and f"/app/model/{model}" in line for line in copied)
    assert "build_cython" not in text and "setup_cython" not in text


def test_compose_requires_readonly_mounts_and_single_service():
    services = yaml.safe_load((DOCKER / "docker-compose.yml").read_text())["services"]
    assert len(services) == 1
    service = next(iter(services.values()))
    assert service["runtime"] == "ascend" and service["privileged"] is True
    assert service["ipc"] == "host" and service["init"] is True
    assert service["restart"] == "no" and service["stop_grace_period"] == "240s"
    assert service["environment"]["CONFIG_PATH"] == "/config.toml"
    assert service["environment"]["ASR_CONTAINER_RUNTIME"] == "cann82-source"
    for target in ("/config.toml", "/app/qwen-model"):
        mount = next(v for v in service["volumes"] if isinstance(v, dict) and v["target"] == target)
        assert mount["type"] == "bind" and mount["read_only"] is True
        assert mount["bind"]["create_host_path"] is False
    assert service["healthcheck"]["start_period"] == "360s"
    assert "asr_offline_910b:v3_" in service["image"]


def test_four_file_delivery_and_explicit_white_list():
    assert {p.name for p in DOCKER.iterdir()} == {"Dockerfile", "docker-compose.yml", "config.toml", "README.md"}
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    significant = [line.strip() for line in ignore if line.strip() and not line.startswith("#")]
    assert significant[0] == "**"
    for line in significant[1:]:
        if line.startswith("!"):
            assert not any(s in line for s in ("tests", "test/", "docs", "scripts", "openspec", "config.toml"))
    readme = (DOCKER / "README.md").read_text()
    for contract in ("CONFIG_PATH", "npu:7", "npu:0", "/v3/seacraft_asr", "asr_offline_910b:v3_"):
        assert contract in readme


def test_source_protection_tool_still_refuses_unmarked_worktree(tmp_path):
    source = tmp_path / "core" / "valuable.py"
    source.parent.mkdir(); source.write_text("original")
    result = subprocess.run(["bash", str(ROOT / "scripts/build_cython.sh"), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and ".asr-build-stage" in result.stderr
    assert source.read_text() == "original"


def _isolated_actual_exec_entrypoint(tmp_path, *, immediate_exit=None):
    """只替换绝对CANN/解释器路径，实际Dockerfile shell/exec及CMD保持。"""
    text = (DOCKER / "Dockerfile").read_text()
    entry = json.loads(next(line.removeprefix("ENTRYPOINT ") for line in text.splitlines() if line.startswith("ENTRYPOINT ")))
    args = json.loads(next(line.removeprefix("CMD ") for line in text.splitlines() if line.startswith("CMD ")))
    ready = tmp_path / "ready.json"
    fake = tmp_path / "runtime.py"
    fake.write_text('''import json,os,signal,sys,time
from pathlib import Path
assert os.environ['PYTHONPATH'].startswith('/fake/cann')
assert sys.argv[sys.argv.index('--workers')+1]=='1'
if os.environ.get('IMMEDIATE_EXIT'): raise SystemExit(int(os.environ['IMMEDIATE_EXIT']))
def stop(*_): raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(os.environ['READY_FILE']).write_text(json.dumps({'pid':os.getpid(),'args':sys.argv}))
while True:time.sleep(.02)
''')
    cann = tmp_path / "set_env.sh"
    cann.write_text('export PYTHONPATH="/fake/cann${PYTHONPATH:+:$PYTHONPATH}"\n')
    entry[2] = entry[2].replace("/usr/local/Ascend/ascend-toolkit/set_env.sh", str(cann))
    entry[2] = entry[2].replace("/opt/asr/bin/python", sys.executable).replace("/app/main.py", str(fake))
    env = dict(os.environ, READY_FILE=str(ready), IMMEDIATE_EXIT="" if immediate_exit is None else str(immediate_exit))
    process = subprocess.Popen(entry + args, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if immediate_exit is None:
            until = time.monotonic()+5
            while not ready.exists():
                assert process.poll() is None and time.monotonic() < until
                time.sleep(.01)
            assert json.loads(ready.read_text())["pid"] == process.pid
            process.send_signal(signal.SIGTERM)
        _, err = process.communicate(timeout=10)
        return process.returncode, err
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)


def test_exec_entrypoint_preserves_pid_and_normal_term_exit(tmp_path):
    code, stderr = _isolated_actual_exec_entrypoint(tmp_path)
    assert code == 0, stderr


def test_exec_entrypoint_preserves_runtime_failure(tmp_path):
    code, _ = _isolated_actual_exec_entrypoint(tmp_path, immediate_exit=7)
    assert code == 7
