import asyncio
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import threading
import time

import pytest

spec = importlib.util.spec_from_file_location("v3_request_client", Path(__file__).resolve().parents[1] / "test/v3_request.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


@pytest.fixture
def server():
    records = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            msg = BytesParser(policy=policy.default).parsebytes(("Content-Type: " + self.headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
            fields = {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                value = part.get_payload(decode=True)
                fields.setdefault(name, []).append(value)
            records.append((self.path, fields))
            data = fields["audioFile"][0]
            if data == b"slow":
                time.sleep(.2)
            payload = {"code": 4017} if data == b"fail" else {"code": 0, "segments": [{"text": "PRIVATE_TEST_TEXT"}]}
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def do_GET(self):
            assert self.path == "/get_status"
            raw = b'{"test_status":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", records
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def execute(tmp_path, url, values, *extra):
    files = []
    for i, value in enumerate(values):
        p = tmp_path / f"{i}.wav"
        p.write_bytes(value)
        files.append(p)
    args = client.parser().parse_args(["--url", url, "--audio", *map(str, files), *extra])
    base, paths, fields = client.plan(args)
    output = tmp_path / "output"
    output.mkdir()
    return asyncio.run(client.run(args, paths, fields, base, output)), output


def test_real_http_multipart_a_defaults_and_no_raw_response(tmp_path, server):
    url, records = server
    result, out = execute(tmp_path, url + "/v3/seacraft_asr", [b"audio"], "--hot-word", "a", "--hot-word", "b")
    assert result["complete"] and result["successful"] == 1
    assert records[0][0] == "/v3/seacraft_asr"
    f = records[0][1]
    assert f["audioFile"] == [b"audio"] and f["showSpk"] == [b"false"] and f["showEmotion"] == [b"false"]
    assert f["showRoleIdentify"] == [b"false"] and f["hotWords"] == [b"a", b"b"]
    assert not list(out.glob("*.body"))
    assert "PRIVATE_TEST_TEXT" not in (out / "summary.json").read_text()


def test_b_override_multifile_correspondence_and_business_failure(tmp_path, server):
    url, records = server
    result, _ = execute(tmp_path, url, [b"one", b"fail", b"three"], "--mode", "B", "--no-show-emotion", "--concurrency", "2")
    assert not result["complete"] and result["successful"] == 2
    assert [r["status"] for r in result["records"]] == ["completed", "failed", "completed"]
    assert result["records"][1]["code"] == 4017
    assert sorted(r[1]["audioFile"][0] for r in records) == [b"fail", b"one", b"three"]
    assert all(r[1]["showSpk"] == [b"true"] and r[1]["showEmotion"] == [b"false"] for r in records)


def test_timeout_stops_refill_captures_status_and_does_not_kill_service(tmp_path, server):
    url, records = server
    result, out = execute(tmp_path, url, [b"slow", b"never-sent"], "--timeout", ".05")
    assert not result["complete"] and result["diagnostic_trigger"]
    assert result["records"][0]["status"] == "timeout"
    assert result["records"][1]["status"] == "not_submitted"
    assert len(records) == 1
    assert json.loads((out / "diagnostic-status.json").read_text()) == {"test_status": True}
    assert result["service_ownership"] == "external_nonowned"


def test_raw_response_opt_in(tmp_path, server):
    result, out = execute(tmp_path, server[0], [b"audio"], "--save-responses")
    assert result["complete"]
    assert "PRIVATE_TEST_TEXT" in (out / "response-0000.body").read_text()


@pytest.mark.parametrize("value", ["421", "nan", "inf", "0", "-1"])
def test_invalid_timeout_prevents_network(tmp_path, value):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"x")
    args = client.parser().parse_args(["--audio", str(audio), "--timeout", value])
    with pytest.raises(ValueError):
        client.plan(args)


def test_directory_limit_dry_plan_and_no_128_full_requests(tmp_path):
    (tmp_path / "1.wav").write_bytes(b"x")
    (tmp_path / "2.mp3").write_bytes(b"x")
    args = client.parser().parse_args(["--audio-dir", str(tmp_path), "--limit", "1", "--dry-run"])
    assert len(client.plan(args)[1]) == 1
    with pytest.raises(SystemExit):
        client.parser().parse_args(["--audio-dir", str(tmp_path), "--concurrency", "128"])


def test_surrogateescape_source_bytes_and_safe_directory_default(tmp_path, server):
    raw = os.fsencode(tmp_path) + b"/\xd2\xf4\xc6\xb5.wav"
    with open(raw, "wb") as f:
        f.write(b"gbk-source")
    (tmp_path / "second.wav").write_bytes(b"second")
    args = client.parser().parse_args(["--audio-dir", str(tmp_path), "--url", server[0]])
    base, files, fields = client.plan(args)
    assert len(files) == 1
    source = Path(os.fsdecode(raw))
    output = tmp_path / "out"
    output.mkdir()
    result = asyncio.run(client.run(args, [source], fields, base, output))
    assert result["complete"]
    assert server[1][0][1]["audioFile"] == [b"gbk-source"]
