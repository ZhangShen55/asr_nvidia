from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"


def test_npu_delivery_isolated_environment_and_raw_model_assets():
    text = (DOCKER / "Dockerfile").read_text()
    assert "ascend_cann8.1.rc1" in text
    assert "COPY --from=builder /src/model /app/model" in text
    assert "COPY --from=builder /src/config.toml /config.toml" in text
    assert "conda-unpack" in text and "/opt/asr/bin/python" in text
    assert "ASR_DEPLOYMENT_PROFILE=npu-910b" in text
    assert "PYTHONNOUSERSITE=1" in text
    assert "download.pytorch.org/whl/cu" not in text
    assert "tensorrt" not in text.lower()


def test_compose_declares_ascend_binding_and_explicit_shutdown():
    text = (DOCKER / "docker-compose.yaml").read_text()
    for field in ("runtime: ascend", "privileged: true", "ipc: host", "ASCEND_VISIBLE_DEVICES", 'restart: "no"', "init: true"):
        assert field in text
    assert "${ASR_CONFIG:-./config.toml}:/config.toml:ro" in text
    assert ":/app/qwen-model:ro" in text
    assert "NVIDIA_VISIBLE_DEVICES" not in text
    assert "driver: nvidia" not in text


def test_build_contract_and_historical_assets_do_not_require_engines():
    script = (DOCKER / "build.sh").read_text()
    readme = (DOCKER / "README.md").read_text()
    assert "asr_offline:v3_" in script
    assert "engine.plan" not in script and "GPU_PROFILE" not in script
    assert "ASR_ENV_ARCHIVE" in script and "conda_pack" in script
    assert "缺少必需模型文件" in script
    assert "/v3/seacraft_asr" in readme and 'docker compose -p "$ASR_COMPOSE_PROJECT" down' in readme
    assert "CONFIG_PATH" in readme and "npu:7" in readme and "npu:0" in readme
    for name in ("Dockerfile.rtx3090", "Dockerfile.rtx4070tisuper", "Dockerfile.rtx4090"):
        assert not (DOCKER / name).exists()


def test_entrypoint_sources_cann_and_uses_owned_process_groups():
    script = (DOCKER / "start.sh").read_text()
    assert 'source "$ASR_CANN_ENV"' in script and "set +u" in script
    assert '"$ASR_PYTHON" main.py --host' in script
    assert "-m uvicorn" not in script
    assert "setsid" in script and "ASR_SHUTDOWN_TIMEOUT_S" in script
    assert 'trap \'cleanup "$?"\' EXIT' in script
    assert 'kill -TERM "$pid"' in script
    assert 'kill -TERM -- "-$pid"' in script
    assert "exit_status=1" in script and "ASR_SHUTDOWN_TERM_TIMEOUT_S" in script
    assert "CUDA_MODULE_LOADING" not in script
    assert "--privileged" not in script


def test_source_protection_refuses_unmarked_worktree(tmp_path):
    source = tmp_path / "core" / "valuable.py"
    source.parent.mkdir();source.write_text("original")
    result = subprocess.run(["bash", str(ROOT / "scripts/build_cython.sh"), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert ".asr-build-stage" in result.stderr
    assert source.read_text() == "original"


def test_legacy_root_docker_entries_are_removed():
    for name in ("Dockerfile", "start.sh", "config.docker.toml", "nginx.conf", ".dockerignore"):
        assert not (ROOT / name).exists()


def _exercise_owned_cleanup(tmp_path, *, ignore_initial_term):
    """Execute the actual trap functions against an owned process, without ports/NPU/nginx."""
    import os
    import signal
    import sys
    import time
    if os.name != "posix":
        import pytest
        pytest.skip("POSIX process groups required")
    trace = tmp_path / "trace"
    ready = tmp_path / "ready"
    program = tmp_path / "owned_process.py"
    program.write_text('''import os,signal,time
from pathlib import Path
trace=Path(os.environ["TRACE"])
def record(name):
    with trace.open("a") as out: out.write(name+"\\n")
def interrupt(*_):
    record("INT")
    raise SystemExit(2)
term_count=0
def terminate(*_):
    global term_count
    term_count += 1
    record("TERM")
    if os.environ["IGNORE_INITIAL_TERM"] == "1" and term_count == 1: return
    raise SystemExit(0)
signal.signal(signal.SIGINT,interrupt)
signal.signal(signal.SIGTERM,terminate)
Path(os.environ["READY"]).touch()
while True: time.sleep(.05)
''')
    env = dict(os.environ, TRACE=str(trace), READY=str(ready), IGNORE_INITIAL_TERM=str(int(ignore_initial_term)))
    child = subprocess.Popen([sys.executable, str(program)], env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            if child.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("owned process failed to become ready")
            time.sleep(.01)
        script = (DOCKER / "start.sh").read_text()
        functions = script[script.index("owned_group_alive() {"):script.index("\ntrap 'cleanup")]
        shell = (f"set -euo pipefail\nchildren=({child.pid})\nnginx_pid=''\n"
                 f"ASR_SHUTDOWN_TIMEOUT_S={1 if ignore_initial_term else 3}\nASR_SHUTDOWN_TERM_TIMEOUT_S=2\n"

                 + functions + "\ntrue\ncleanup 0\n")
        result = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, errors="replace", timeout=10)
        child.wait(timeout=3)
        return result, trace.read_text().splitlines()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=3)


def test_unknown_not_owned_child_wait_127_is_not_reported_as_success(tmp_path):
    # The child is owned by pytest, not the extracted shell. Its wait status is truly unknown there.
    result, signals = _exercise_owned_cleanup(tmp_path, ignore_initial_term=False)
    assert result.returncode != 0 and signals == ["TERM"]
    assert "exit=127" in result.stderr


def _exercise_full_entrypoint_term(tmp_path, *, ignore_first_term, main_exit="0", early_exit="", external_term=True, race_point=None, repeat_term=False):
    """Run the whole wrapper: only its /etc output path and external programs are isolated."""
    import json
    import os
    import shlex
    import shutil
    import signal
    import sys
    import time
    if os.name != "posix":
        import pytest
        pytest.skip("POSIX process groups required")
    app = tmp_path / "app"
    binaries = tmp_path / "bin"
    app.mkdir(); binaries.mkdir()
    trace = tmp_path / "events.jsonl"
    ready_dir = tmp_path / "ready"; ready_dir.mkdir()
    fake = tmp_path / "external_programs.py"
    fake.write_text(r'''import json,os,signal,sys,time
from pathlib import Path
def event(role, kind):
    with Path(os.environ['WRAPPER_TRACE']).open('a') as out:
        out.write(json.dumps({'role':role,'event':kind,'pid':os.getpid(),'pgid':os.getpgrp()})+'\n')
if sys.argv[1]=='--setsid':
    os.setsid()
    os.execvp(sys.argv[2],sys.argv[2:])
if sys.argv[1]=='-':
    print(1)
    raise SystemExit(0)
if sys.argv[1].startswith('scripts/check_npu_environment.py'):
    assert os.environ['PYTHONPATH'].startswith('/fake/cann')
    raise SystemExit(0)
role='nginx' if sys.argv[1]=='--nginx' else 'asr'
if role=='nginx' and '-t' in sys.argv: raise SystemExit(0)
if role=='asr':
    assert sys.argv[1]=='main.py' and sys.argv[sys.argv.index('--workers')+1]=='1'
terms=0
def stop(signum,_frame):
    global terms
    kind='TERM' if signum==signal.SIGTERM else 'QUIT'
    event(role,kind)
    if role=='asr' and signum==signal.SIGTERM:
        terms+=1
        if os.environ['IGNORE_FIRST_TERM']=='1' and terms==1:return
    if role=='asr' and os.environ['MAIN_EXIT']=='signal':
        signal.signal(signal.SIGTERM,signal.SIG_DFL)
        os.kill(os.getpid(),signal.SIGTERM)
    if role=='asr':time.sleep(float(os.environ.get('MAIN_TERM_DELAY','0')))
    raise SystemExit(int(os.environ['MAIN_EXIT']) if role=='asr' else 0)
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGQUIT,stop)
event(role,'READY')
(Path(os.environ['WRAPPER_READY_DIR'])/role).touch()
if role=='asr' and os.environ['EARLY_EXIT']:
    time.sleep(.05)
    raise SystemExit(int(os.environ['EARLY_EXIT']))
while True:time.sleep(.05)
''')
    def executable(name, arguments):
        path = binaries / name
        path.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(fake)) + " " + arguments + ' "$@"\n')
        path.chmod(0o755)
        return path
    python = executable("runtime-python", "")
    executable("nginx", "--nginx")
    if not shutil.which("setsid"):
        executable("setsid", "--setsid")
    cann = tmp_path / "cann-set-env.sh"
    cann.write_text('export PYTHONPATH="/fake/cann${PYTHONPATH:+:$PYTHONPATH}"\n')
    config = tmp_path / "config.toml"; config.write_text("instance_count = 1\n")
    entrypoint = tmp_path / "start.sh"
    source = (DOCKER / "start.sh").read_text()
    assert source.count("/etc/nginx/asr-conf.d") == 2
    # This redirects filesystem output only; traps/startup/owned process control are exact production code.
    copied_source = source.replace("/etc/nginx/asr-conf.d", shlex.quote(str(tmp_path / "nginx-conf")))
    if race_point:
        prefixes = {"before_sleep": "sleep 0.2", "before_exit": 'exit "${ASR_STOP_STATUS:-0}"', "pid_registration": 'children+=("$!")', "pid_registration_int_then_term": 'children+=("$!")'}
        prefix = prefixes[race_point]
        # DEBUG is test-only scheduling. Deliver a real TERM at exact boundaries;
        # normal production startup/supervision/cleanup code is otherwise unchanged.
        pause = 'while [[ ! -f "$WRAPPER_READY_DIR/asr" || ! -f "$WRAPPER_READY_DIR/nginx" ]]; do sleep .01; done; '
        action = 'kill -INT "$BASHPID"; kill -TERM "$BASHPID"' if race_point == "pid_registration_int_then_term" else 'kill -TERM "$BASHPID"'
        debug_body = 'if [[ "$BASH_COMMAND" == ' + shlex.quote(prefix) + '* ]]; then trap - DEBUG; ' + pause + action + '; fi'
        hook = "trap " + shlex.quote(debug_body) + " DEBUG\n"
        copied_source = copied_source.replace("setsid nginx -g", hook + "setsid nginx -g",1)
    entrypoint.write_text(copied_source)
    env = dict(os.environ, PATH=str(binaries)+os.pathsep+os.environ.get("PATH", ""),
               ASR_APP_ROOT=str(app), ASR_PYTHON=str(python), ASR_CANN_ENV=str(cann),
               ASR_RUNTIME_DIR=str(tmp_path / "runtime"), CONFIG_PATH=str(config),
               ASR_SHUTDOWN_TIMEOUT_S="1" if ignore_first_term else "3", ASR_SHUTDOWN_TERM_TIMEOUT_S="2",
               WRAPPER_TRACE=str(trace), IGNORE_FIRST_TERM=str(int(ignore_first_term)),
               MAIN_EXIT=str(main_exit), EARLY_EXIT=str(early_exit), WRAPPER_READY_DIR=str(ready_dir),
               MAIN_TERM_DELAY=".15" if repeat_term else "0")
    process = subprocess.Popen(["bash", str(entrypoint)], env=env, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    def events():
        if not trace.exists(): return []
        rows = []
        for line in trace.read_text().splitlines():
            try: rows.append(json.loads(line))
            except ValueError: pass  # another owned process may be appending the final bytes
        return rows
    try:
        deadline=time.monotonic()+5
        while {row['role'] for row in events() if row['event']=='READY'} != {'asr','nginx'}:
            if process.poll() is not None or time.monotonic()>deadline:
                raise AssertionError("full wrapper failed before both fake external programs became ready")
            time.sleep(.01)
        if external_term:
            process.send_signal(signal.SIGTERM)
            if repeat_term:
                repeat_deadline=time.monotonic()+3
                while not any(row["role"]=="asr" and row["event"]=="TERM" for row in events()):
                    assert process.poll() is None and time.monotonic()<repeat_deadline
                    time.sleep(.005)
                assert process.poll() is None
                process.send_signal(signal.SIGTERM)
        stdout,stderr=process.communicate(timeout=8)
        rows=events()
        ready=[row for row in rows if row['event']=='READY']
        for row in ready:
            command=subprocess.run(["ps","-p",str(row["pid"]),"-o","command="],capture_output=True,text=True).stdout
            assert str(fake) not in command
        return process.returncode,rows,stdout,stderr
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
        # Children have their own setsid groups. Clean only recorded PIDs whose command still names this unique fixture.
        for row in events():
            if row['event']!='READY':continue
            command=subprocess.run(["ps","-p",str(row["pid"]),"-o","command="],capture_output=True,text=True).stdout
            try:
                if str(fake) in command and os.getpgid(row["pid"])==row["pid"]:
                    os.killpg(row["pid"],signal.SIGKILL)
            except ProcessLookupError:pass
        process.communicate(timeout=3)


def test_full_entrypoint_external_term_returns_zero_after_owned_cleanup(tmp_path):
    code,events,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False)
    assert code==0,stderr
    assert [row['event'] for row in events if row['role']=='asr'] == ['READY','TERM']
    assert [row['event'] for row in events if row['role']=='nginx'] == ['READY','QUIT']


def test_full_entrypoint_external_term_cannot_hide_timeout_failure(tmp_path):
    code,events,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=True)
    assert code!=0 and "收尾失败" in stderr
    assert [row['event'] for row in events if row['role']=='asr'] == ['READY','TERM','TERM']



def test_full_entrypoint_term_preserves_child_nonzero_exit(tmp_path):
    code,_,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,main_exit="7")
    assert code!=0 and "exit=7" in stderr


def test_full_entrypoint_term_preserves_child_signal_exit(tmp_path):
    code,_,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,main_exit="signal")
    assert code!=0 and "exit=143" in stderr


def test_full_entrypoint_early_zero_exit_is_not_confused_with_unknown_127(tmp_path):
    code,_,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,early_exit="0",external_term=False)
    assert code==0,stderr
    assert "exit=127" not in stderr


def test_full_entrypoint_early_nonzero_exit_is_preserved(tmp_path):
    code,_,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,early_exit="7",external_term=False)
    assert code!=0 and "exit=7" in stderr



def test_full_entrypoint_term_between_flag_check_and_sleep_does_not_lose_wakeup(tmp_path):
    code,_,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,external_term=False,race_point="before_sleep")
    assert code==0,stderr


def test_full_entrypoint_term_after_detecting_child_failure_does_not_hide_it(tmp_path):
    code,_,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,early_exit="7",external_term=False,race_point="before_exit")
    assert code!=0 and "exit=7" in stderr
    assert "exit=127" not in stderr


def test_full_entrypoint_startup_term_does_not_abandon_pid_registration(tmp_path):
    code,events,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,external_term=False,race_point="pid_registration")
    assert code==0,stderr
    assert [row["event"] for row in events if row["role"]=="asr"]==["READY","TERM"]



def test_full_entrypoint_startup_int_is_not_washed_by_following_term(tmp_path):
    code,_,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,external_term=False,race_point="pid_registration_int_then_term")
    assert code==130,stderr


def test_full_entrypoint_repeated_term_during_cleanup_does_not_abandon_children(tmp_path):
    code,events,_,stderr=_exercise_full_entrypoint_term(tmp_path,ignore_first_term=False,repeat_term=True)
    assert code==0,stderr
    assert [row["event"] for row in events if row["role"]=="asr"]==["READY","TERM"]
