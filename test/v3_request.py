"""像Postman一样发送v3 multipart请求；不拥有或启动被测服务。"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import time
from urllib.parse import urlsplit


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8083", help="服务根URL或完整/v3/seacraft_asr URL")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", nargs="+", type=Path)
    source.add_argument("--audio-dir", type=Path, help="递归选择目录内wav/mp3文件")
    p.add_argument("--limit", type=int, default=1, help="目录模式最多选几条；默认1，0为全部（慎用）")
    p.add_argument("--mode", choices=["A", "B"], default="A", help="A纯转写；B说话人+情绪")
    p.add_argument("--concurrency", choices=[1, 2, 4, 8], type=int, default=1, help="完整音频同时请求数，不是VAD段并发")
    p.add_argument("--show-spk", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--show-emotion", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--language", default="auto")
    p.add_argument("--hot-word", action="append", default=[])
    p.add_argument("--word-timestamps", action="store_true")
    p.add_argument("--show-speed", action="store_true")
    p.add_argument("--timeout", type=float, default=420, help="每请求总墙钟期限，最大420秒")
    p.add_argument("--output", type=Path, help="新建结果目录，必须位于项目logs下")
    p.add_argument("--save-responses", action="store_true", help="将原始转写响应另存ignored logs；控制台不打印正文")
    p.add_argument("--dry-run", "--list-only", action="store_true", help="仅列出请求计划，不访问服务")
    return p


def plan(args):
    url = urlsplit(args.url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("URL须为不含凭据/query/fragment的http(s)地址")
    path = url.path.rstrip("/")
    if path not in ("", "/v3/seacraft_asr"):
        raise ValueError("URL路径须为空或/v3/seacraft_asr")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 420:
        raise ValueError("timeout须在(0,420]秒内")
    if args.limit < 0:
        raise ValueError("limit不可为负数")
    files = args.audio
    if args.audio_dir is not None:
        if not args.audio_dir.is_dir():
            raise ValueError("audio-dir目录不存在")
        files = sorted((p for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".wav", ".mp3")), key=os.fsencode)
        if args.limit:
            files = files[:args.limit]
    if not files or any(not p.is_file() for p in files):
        raise ValueError("音频列表为空或包含不存在的文件")
    fields = {"language": args.language, "showSpk": str(args.mode == "B" if args.show_spk is None else args.show_spk).lower(),
              "showEmotion": str(args.mode == "B" if args.show_emotion is None else args.show_emotion).lower(),
              "showRoleIdentify": "false", "wordTimestamps": str(args.word_timestamps).lower(),
              "showSpeed": str(args.show_speed).lower(), "openPanel": "false"}
    if args.hot_word:
        fields["hotWords"] = args.hot_word
    return url._replace(path="", query="", fragment="").geturl().rstrip("/"), files, fields


async def run(args, files, fields, base, output):
    import httpx
    records = [{"index": i, "path": str(p), "status": "not_submitted"} for i, p in enumerate(files)]
    started = time.monotonic()
    trigger = None
    pending = iter(records)
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(args.timeout), follow_redirects=False) as client:
        async def worker():
            nonlocal trigger
            for row in pending:
                if trigger:
                    break
                row.update(status="running", started_utc=datetime.now(timezone.utc).isoformat())
                begin = time.monotonic()
                try:
                    # ASCII上传名保留唯一序号；磁盘路径可含GBK surrogateescape，不重命名源音频。
                    path = files[row["index"]]
                    with path.open("rb") as audio:
                        response = await asyncio.wait_for(client.post(base + "/v3/seacraft_asr", data=fields,
                            files={"audioFile": (f"audio-{row['index']:04d}{path.suffix.lower()}", audio, mimetypes.guess_type(path.name)[0] or "application/octet-stream")}), args.timeout)
                    row["http_status"] = response.status_code
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                    if args.save_responses:
                        (output / f"response-{row['index']:04d}.body").write_bytes(response.content)
                    row["code"] = payload.get("code") if isinstance(payload, dict) else None
                    segments = payload.get("segments") if isinstance(payload, dict) else None
                    row["segments"] = len(segments) if isinstance(segments, list) else None
                    row["status"] = "completed" if response.is_success and isinstance(payload, dict) and row["code"] in (None, 0) and isinstance(segments, list) else "failed"
                except asyncio.CancelledError:
                    row["status"] = "cancelled"
                    raise
                except (asyncio.TimeoutError, httpx.TimeoutException):
                    row["status"] = "timeout"
                    # 立即停追加；取消其他客户端请求不代表服务端设备任务已结束。
                    trigger = {"reason": "request_wall_timeout", "index": row["index"], "limit_s": args.timeout}
                except Exception as exc:
                    row.update(status="failed", error_type=type(exc).__name__)
                finally:
                    row["elapsed_s"] = time.monotonic() - begin
                    if row["elapsed_s"] > args.timeout and row["status"] != "cancelled" and trigger is None:
                        trigger = {"reason": "request_wall_timeout", "index": row["index"], "limit_s": args.timeout}
                    write_json(output / f"request-{row['index']:04d}.json", row)
                if trigger:
                    break

        tasks = [asyncio.create_task(worker()) for _ in range(min(args.concurrency, len(files)))]
        try:
            while any(not t.done() for t in tasks):
                await asyncio.wait([t for t in tasks if not t.done()], timeout=.05, return_when=asyncio.FIRST_COMPLETED)
                if trigger:
                    break
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if trigger:
                try:
                    status = await asyncio.wait_for(client.get(base + "/get_status", timeout=3), 3)
                    write_json(output / "diagnostic-status.json", status.json())
                except Exception as exc:
                    write_json(output / "diagnostic-status.json", {"error_type": type(exc).__name__})
            summary = {"complete": trigger is None and all(r["status"] == "completed" for r in records), "successful": sum(r["status"] == "completed" for r in records),
                       "expected": len(records), "concurrency": args.concurrency, "fields": fields, "batch_elapsed_s": time.monotonic() - started,
                       "diagnostic_trigger": trigger, "records": records, "service_ownership": "external_nonowned",
                       "service_shutdown": "not_requested; client does not own service; cancelled HTTP does not prove device cleanup"}
            write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        base, files, fields = plan(args)
        if args.dry_run:
            print(json.dumps({"url": base + "/v3/seacraft_asr", "files": len(files), "concurrency": args.concurrency, "fields": fields,
                              "samples": [{"sample_id": f"audio-{i:04d}", "path_bytes_b64": base64.b64encode(os.fsencode(f)).decode(), "bytes": f.stat().st_size} for i, f in enumerate(files)]}, ensure_ascii=True))
            return 0
        logs = Path(__file__).resolve().parents[1] / "logs"
        output = (args.output or logs / ("v3-request-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))).resolve()
        if not output.is_relative_to(logs.resolve()):
            raise ValueError("output须位于项目logs目录下，避免原始转写误入库")
        output.mkdir(parents=True, exist_ok=False)
        result = asyncio.run(run(args, files, fields, base, output))
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    except KeyboardInterrupt:
        print("请求客户端已中断；现有服务未关闭，请由服务启动者核对设备任务和收尾。")
        return 130
    print(json.dumps({k: result[k] for k in ("complete", "successful", "expected", "concurrency", "batch_elapsed_s", "diagnostic_trigger")}, ensure_ascii=True))
    for row in result["records"]:
        print(json.dumps({k: v for k, v in row.items() if k != "path"}, ensure_ascii=True))
    print("结果目录: " + str(output))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
