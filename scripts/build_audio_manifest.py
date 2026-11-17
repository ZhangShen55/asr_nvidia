#!/usr/bin/env python3
"""只读音频清单：原始路径字节与显示名分离，不推断语种/说话人标签。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import wave

EXTENSIONS = {b".wav", b".mp3", b".flac", b".m4a", b".ogg"}


def display_path(path: bytes):
    """逐组件解码，兼容同一树内混合UTF-8/GBK；该值仅供显示。"""
    parts, encodings = [], set()
    for component in path.split(os.sep.encode()):
        if not component:
            parts.append("")
            continue
        for encoding in ("utf-8", "gb18030"):
            try:
                parts.append(component.decode(encoding))
                encodings.add(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            parts.append("[bytes:" + component.hex() + "]")
            encodings.add("hex")
    return os.sep.join(parts), sorted(encodings)


def sha256_file(path: bytes, chunk_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_ffprobe():
    candidate = Path(sys.executable).parent / "ffprobe"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("ffprobe")


def probe_mp3(path: bytes, executable):
    if not executable:
        raise RuntimeError("ffprobe不可用；仅完成文件哈希，未验证容器元数据")
    result = subprocess.run(
        [os.fsencode(executable), b"-v", b"error", b"-select_streams", b"a:0",
         b"-show_entries", b"stream=codec_name,sample_rate,channels:format=duration",
         b"-of", b"json", path], capture_output=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[:2000])
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("ffprobe未发现音频流")
    stream = streams[0]
    duration = data.get("format", {}).get("duration")
    return {"evidence": "container_probe", "codec": stream.get("codec_name"),
            "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            "channels": stream.get("channels"),
            "duration_seconds": float(duration) if duration is not None else None,
            "full_decode_checked": False, "tool": str(executable)}


def build_manifest(root, *, ffprobe=None):
    root_bytes = os.path.abspath(os.fsencode(root))
    if not os.path.isdir(root_bytes):
        raise ValueError("音频扫描根目录不存在")
    executable = find_ffprobe() if ffprobe is None else ffprobe
    samples, skipped, errors = [], [], []
    def walk_error(exc):
        errors.append({"stage": "directory_scan", "path_hex": os.fsencode(exc.filename or root_bytes).hex(),
                       "type": type(exc).__name__, "message": str(exc)})
    for directory, dirs, files in os.walk(root_bytes, followlinks=False, onerror=walk_error):
        dirs.sort()
        for filename in sorted(files):
            extension = os.path.splitext(filename)[1].lower()
            if extension not in EXTENSIONS:
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root_bytes)
            shown, encodings = display_path(relative)
            try:
                regular = stat.S_ISREG(os.stat(path, follow_symlinks=False).st_mode)
            except OSError as exc:
                errors.append({"stage": "file_stat", "path_hex": path.hex(), "type": type(exc).__name__, "message": str(exc)})
                continue
            if not regular:
                skipped.append({"relative_path_hex": relative.hex(), "display_path": shown,
                                "reason": "不是普通文件；不跟随符号链接读取根目录外资产"})
                continue
            sample_id = "audio-" + hashlib.sha256(relative).hexdigest()[:20]
            item = {"sample_id": sample_id, "path_hex": path.hex(),
                    "relative_path_hex": relative.hex(), "display_path": shown,
                    "display_encodings": encodings, "extension": extension.decode("ascii"),
                    "labels": None, "errors": []}
            stage = "file_hash"
            try:
                before = os.stat(path)
                item["bytes"] = before.st_size
                item["sha256"] = sha256_file(path)
                stage = "media_metadata"
                if extension == b".wav":
                    with wave.open(os.fsdecode(path), "rb") as audio:
                        item["media"] = {"evidence": "wav_header", "sample_rate": audio.getframerate(),
                                         "channels": audio.getnchannels(), "sample_width_bytes": audio.getsampwidth(),
                                         "frames": audio.getnframes(),
                                         "duration_seconds": audio.getnframes() / audio.getframerate(),
                                         "compression": audio.getcomptype(), "full_decode_checked": False}
                else:
                    item["media"] = probe_mp3(path, executable)
                stage = "source_stability"
                after = os.stat(path)
                if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
                    raise RuntimeError("扫描期间源文件发生变化，不能视为稳定快照")
            except Exception as exc:
                error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
                item["errors"].append(error)
                errors.append({"sample_id": sample_id, **error})
            samples.append(item)
    ids = [item["sample_id"] for item in samples]
    if len(ids) != len(set(ids)):
        raise RuntimeError("sample_id碰撞，拒绝产生不可追溯清单")
    duplicates = defaultdict(list)
    for item in samples:
        if "sha256" in item:
            duplicates[item["sha256"]].append(item["sample_id"])
    root_display, _ = display_path(root_bytes)
    counts = Counter(item["extension"] for item in samples)
    wav_duration = sum(item.get("media", {}).get("duration_seconds", 0)
                       for item in samples if item["extension"] == ".wav")
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "root_path_hex": root_bytes.hex(), "root_display_path": root_display,
            "evidence": "file_hash_and_header_or_container_probe",
            "full_decode_checked": False, "labels_inferred_from_filename": False,
            "summary": {"sample_count": len(samples), "extension_counts": dict(counts),
                        "total_bytes": sum(item.get("bytes", 0) for item in samples),
                        "wav_header_duration_seconds": wav_duration,
                        "error_count": len(errors), "skipped_count": len(skipped)},
            "duplicates": [{"sha256": digest, "sample_ids": ids}
                           for digest, ids in sorted(duplicates.items()) if len(ids) > 1],
            "duplicate_scope": "exact_file_hash_only; different encodings may contain the same recording",
            "samples": samples, "errors": errors, "skipped": skipped}


def write_manifest(manifest, output):
    destination = os.path.abspath(os.fsencode(output))
    root = bytes.fromhex(manifest["root_path_hex"])
    if os.path.commonpath([os.path.realpath(root), os.path.realpath(destination)]) == os.path.realpath(root):
        raise ValueError("输出不得放在原始音频根目录内，拒绝覆盖输入资产")
    target = Path(os.fsdecode(destination))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".audio-manifest-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="test", help="可用父目录test递归发现非UTF-8文件名")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.root)
    write_manifest(manifest, args.output)
    print(json.dumps({"output": args.output, **manifest["summary"],
                      "duplicate_groups": len(manifest["duplicates"]),
                      "full_decode_checked": False}, ensure_ascii=False))
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
