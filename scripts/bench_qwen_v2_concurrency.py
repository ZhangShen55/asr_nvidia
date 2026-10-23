"""v2 接口的多档并发压测：采集吞吐、端到端时延、GPU 占用、引擎排队与效果指标。

用法：
    python scripts/bench_qwen_v2_concurrency.py \
        --base-url http://127.0.0.1:8083 \
        --concurrency 4,8,16,32 \
        --out-dir /tmp/bench_qwen

每个并发档位从音频池中取前 N 个**完整**音频文件（不截断），一轮压测。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
_DURATION_CACHE: Dict[str, float] = {}
_VOLUME_CACHE: Dict[str, float] = {}


def docker_cmd(args: List[str]) -> Optional[str]:
    """执行 docker 命令；普通用户无权限时回退到 sudo -n。"""
    for prefix in (["docker"], ["sudo", "-n", "docker"]):
        try:
            return subprocess.check_output(
                prefix + args, text=True, timeout=20, stderr=subprocess.DEVNULL
            )
        except Exception:
            continue
    return None


def _duration_min(name: str) -> float:
    """音频时长（分钟），带缓存。"""
    if name in _DURATION_CACHE:
        return _DURATION_CACHE[name]
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(REPO_ROOT / "test" / "audio_file" / name),
            ],
            text=True,
            timeout=30,
        ).strip()
        value = float(out)
    except Exception:
        value = 0.0
    _DURATION_CACHE[name] = value
    return value


def _mean_volume_db(name: str) -> float:
    """音频平均音量（dB），用于剔除静音录音。"""
    if name in _VOLUME_CACHE:
        return _VOLUME_CACHE[name]
    value = -999.0
    try:
        out = subprocess.check_output(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(REPO_ROOT / "test" / "audio_file" / name),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "/dev/null",
            ],
            text=True,
            timeout=120,
            stderr=subprocess.STDOUT,
        )
        for line in out.splitlines():
            if "mean_volume:" in line:
                value = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                break
    except Exception:
        pass
    _VOLUME_CACHE[name] = value
    return value


# ---------------------------------------------------------------- 采样工具


def sample_gpu() -> List[Dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    rows = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "util": int(parts[1]),
                    "mem_mb": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def container_cpu_seconds(container: str) -> Optional[float]:
    """容器内所有进程累计占用的 CPU 秒数。"""
    out = docker_cmd(["inspect", "-f", "{{.State.Pid}}", container])
    if not out:
        return None
    pid = out.strip()
    if not pid or pid == "0":
        return None

    pids = [int(pid)]
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,ppid"], text=True, timeout=15
        ).strip().splitlines()[1:]
        children: Dict[int, List[int]] = {}
        for line in out:
            parts = line.split()
            if len(parts) == 2:
                children.setdefault(int(parts[1]), []).append(int(parts[0]))
        stack = list(pids)
        seen = set(pids)
        while stack:
            cur = stack.pop()
            for child in children.get(cur, []):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        pids = sorted(seen)
    except Exception:
        pass

    ticks = os.sysconf("SC_CLK_TCK")
    total = 0.0
    for p in pids:
        try:
            with open(f"/proc/{p}/stat", "rb") as handle:
                data = handle.read().decode("utf-8", "ignore")
            fields = data[data.rindex(")") + 2:].split()
            total += (int(fields[11]) + int(fields[12])) / ticks
        except Exception:
            continue
    return total


def parse_engine_metrics(metrics_text: str) -> Dict[str, float]:
    """从 vLLM /metrics 文本里抽取排队与 KV cache 指标。"""
    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_usage",
        "vllm:num_preemptions_total": "preemptions",
        "vllm:request_queue_time_seconds_sum": "queue_time_sum",
        "vllm:request_queue_time_seconds_count": "queue_time_count",
        "vllm:e2e_request_latency_seconds_sum": "e2e_sum",
        "vllm:e2e_request_latency_seconds_count": "e2e_count",
        "vllm:request_success_total{engine=\"0\",finished_reason=\"length\"": "truncated",
    }
    out: Dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        for prefix, key in wanted.items():
            if line.startswith(prefix):
                parts = line.rsplit(" ", 1)
                try:
                    out[key] = float(parts[-1])
                except ValueError:
                    pass
    return out


STATS_RE = re.compile(
    r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*([\d.]+)\s*tokens/s,\s*"
    r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs"
)
KV_RE = re.compile(r"GPU KV cache usage:\s*([\d.]+)%")


def scrape_engine_stats(container: str) -> Dict[str, float]:
    """从容器日志里抓 vLLM 的周期性统计行（Running/Waiting/KV/吞吐）。"""
    out = docker_cmd(["logs", "--since", "20s", container])
    if not out:
        return {}
    stats: Dict[str, float] = {}
    for line in out.splitlines():
        match = STATS_RE.search(line)
        if match:
            stats["prompt_tps"] = float(match.group(1))
            stats["gen_tps"] = float(match.group(2))
            stats["running"] = float(match.group(3))
            stats["waiting"] = float(match.group(4))
        kv = KV_RE.search(line)
        if kv:
            stats["kv_usage"] = float(kv.group(1))
    return stats


def fetch_engine_metrics(url: str) -> Dict[str, float]:
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return {}
        return parse_engine_metrics(resp.text)
    except Exception:
        return {}


# ---------------------------------------------------------------- 效果指标

PUNCTUATION = set("，。？！；：、,.?!;:…—“”‘’\"'()（）《》【】[]{}-—～~·")


def longest_repeat_loop(text: str) -> int:
    """段内最长重复片段长度（同一子串连续重复 >= 3 次）。"""
    n = len(text)
    for size in range(3, min(n // 2, 120) + 1):
        for start in range(0, n - size * 2 + 1):
            chunk = text[start:start + size]
            if not chunk.strip():
                continue
            count = 1
            pos = start + size
            while text[pos:pos + size] == chunk:
                count += 1
                pos += size
            if count >= 3:
                return size
    return 0


def analyze_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """统计单个响应的效果指标。"""
    segments = payload.get("segments") or []
    pure_punct = 0
    leading_punct = 0
    loop_segments = 0
    longest_loop = 0
    consecutive_dup = 0
    empty_segments = 0
    prefix_residue = 0
    special_token_residue = 0
    lengths: List[int] = []
    prev_text = None

    for seg in segments:
        text = str(seg.get("segment_text", "") or "")
        stripped = text.strip()
        if not stripped:
            empty_segments += 1
            continue
        lengths.append(len(stripped))
        if all(ch in PUNCTUATION for ch in stripped):
            pure_punct += 1
        if stripped[0] in PUNCTUATION:
            leading_punct += 1
        if "asr_text" in stripped or "<|" in stripped:
            prefix_residue += 1
        if re.search(r"<(/?)(asr_text|audio|transcript)\b", stripped):
            special_token_residue += 1
        loop = longest_repeat_loop(stripped)
        if loop >= 6:
            loop_segments += 1
            longest_loop = max(longest_loop, loop)
        if prev_text is not None and stripped == prev_text:
            consecutive_dup += 1
        prev_text = stripped

    text_all = str(payload.get("text") or "")
    return {
        "segments": len(segments),
        "avg_segment_chars": round(statistics.mean(lengths), 2) if lengths else 0.0,
        "pure_punctuation_segments": pure_punct,
        "leading_punctuation_segments": leading_punct,
        "empty_segments": empty_segments,
        "repeat_loop_segments": loop_segments,
        "longest_repeat_loop": longest_loop,
        "consecutive_duplicate_segments": consecutive_dup,
        "prefix_residue_segments": prefix_residue,
        "special_token_residue_segments": special_token_residue,
        "total_chars": len(text_all),
    }


# ---------------------------------------------------------------- 请求


def post_audio(
    base_url: str,
    path: str,
    show_spk: bool,
    show_emotion: bool,
    timeout: int = 7200,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v2/seacraft_asr"
    data = {}
    if show_spk:
        data["showSpk"] = "true"
    if show_emotion:
        data["showEmotion"] = "true"
    started = time.perf_counter()
    status = 0
    payload: Dict[str, Any] = {}
    error = None
    try:
        with open(path, "rb") as audio:
            response = requests.post(
                url,
                files={
                    "audioFile": (
                        os.path.basename(path),
                        audio,
                        "application/octet-stream",
                    )
                },
                data=data,
                timeout=timeout,
            )
        status = response.status_code
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    return {
        "file": os.path.basename(path),
        "elapsed_s": round(elapsed, 2),
        "http": status,
        "error": error,
        "code": payload.get("code"),
        "msg": payload.get("msg"),
        "language": payload.get("language"),
        "language_ratio": payload.get("language_ratio"),
        "payload": payload,
    }


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 3)


def run_level(
    base_url: str,
    files: List[str],
    concurrency: int,
    show_spk: bool,
    show_emotion: bool,
    out_dir: Path,
    label: str,
    metrics_url: str,
    container: str,
) -> Dict[str, Any]:
    gpu_samples: List[List[Dict[str, Any]]] = []
    engine_samples: List[Dict[str, float]] = []
    cpu_samples: List[float] = []
    stop = threading.Event()

    def _gpu_sampler():
        while not stop.is_set():
            gpu_samples.append(sample_gpu())
            time.sleep(0.2)

    def _engine_sampler():
        while not stop.is_set():
            m = fetch_engine_metrics(metrics_url)
            if m:
                stats = scrape_engine_stats(container)
                stats.update(m)
                engine_samples.append(stats)
            time.sleep(1.0)

    def _cpu_sampler():
        prev_t = time.time()
        prev_v = container_cpu_seconds(container)
        while not stop.is_set():
            time.sleep(2.0)
            cur_t = time.time()
            cur_v = container_cpu_seconds(container)
            if prev_v is not None and cur_v is not None and cur_t > prev_t:
                cpu_samples.append((cur_v - prev_v) / (cur_t - prev_t))
            prev_t, prev_v = cur_t, cur_v

    threads = [
        threading.Thread(target=_gpu_sampler, daemon=True),
        threading.Thread(target=_engine_sampler, daemon=True),
        threading.Thread(target=_cpu_sampler, daemon=True),
    ]
    for thread in threads:
        thread.start()

    level_started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(post_audio, base_url, f, show_spk, show_emotion)
            for f in files
        ]
        for index, future in enumerate(futures):
            item = future.result()
            results.append(item)
            print(
                f"  [{label} c={concurrency}] {index + 1}/{len(files)} "
                f"{item['file'][:34]} {item['elapsed_s']}s "
                f"code={item['code']} err={item['error']}",
                flush=True,
            )
    wall_s = time.perf_counter() - level_started
    stop.set()
    for thread in threads:
        thread.join(timeout=3)

    resp_dir = out_dir / f"responses_{label}_c{concurrency}"
    resp_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(results):
        name = f"{index:03d}_{item['file']}.json"
        with open(resp_dir / name, "w", encoding="utf-8") as handle:
            json.dump(item["payload"], handle, ensure_ascii=False, indent=1)

    latencies = [item["elapsed_s"] for item in results]
    ok = [i for i in results if i["code"] in (None, 0) and not i["error"]]
    failed = [i for i in results if i not in ok]
    total_segments = sum(len(i["payload"].get("segments") or []) for i in ok)
    total_chars = sum(len(str(i["payload"].get("text") or "")) for i in ok)
    audio_seconds = sum(_duration_min(i["file"]) for i in ok)

    quality = [analyze_payload(i["payload"]) for i in ok]
    agg_quality = {
        "pure_punctuation_segments": sum(q["pure_punctuation_segments"] for q in quality),
        "leading_punctuation_segments": sum(q["leading_punctuation_segments"] for q in quality),
        "empty_segments": sum(q["empty_segments"] for q in quality),
        "repeat_loop_segments": sum(q["repeat_loop_segments"] for q in quality),
        "longest_repeat_loop": max((q["longest_repeat_loop"] for q in quality), default=0),
        "consecutive_duplicate_segments": sum(
            q["consecutive_duplicate_segments"] for q in quality
        ),
        "prefix_residue_segments": sum(q["prefix_residue_segments"] for q in quality),
        "special_token_residue_segments": sum(
            q["special_token_residue_segments"] for q in quality
        ),
    }

    gpu_util: Dict[str, List[int]] = {}
    gpu_mem: Dict[str, List[int]] = {}
    for snapshot in gpu_samples:
        for row in snapshot:
            gpu_util.setdefault(str(row["index"]), []).append(row["util"])
            gpu_mem.setdefault(str(row["index"]), []).append(row["mem_mb"])

    engine_agg = {}
    for key in [
        "running",
        "waiting",
        "kv_usage",
        "prompt_tps",
        "gen_tps",
        "preemptions",
        "truncated",
    ]:
        values = [s[key] for s in engine_samples if key in s]
        engine_agg[key] = {
            "max": round(max(values), 3) if values else None,
            "mean": round(statistics.mean(values), 3) if values else None,
        }
    # 引擎排队时延（累计值相除得到均值）
    queue_sums = [s.get("queue_time_sum") for s in engine_samples if "queue_time_sum" in s]
    queue_counts = [s.get("queue_time_count") for s in engine_samples if "queue_time_count" in s]
    if queue_sums and queue_counts and queue_counts[-1]:
        engine_agg["avg_queue_time_s"] = round(queue_sums[-1] / queue_counts[-1], 4)
        engine_agg["requests_total"] = queue_counts[-1]
    e2e_sums = [s.get("e2e_sum") for s in engine_samples if "e2e_sum" in s]
    e2e_counts = [s.get("e2e_count") for s in engine_samples if "e2e_count" in s]
    if e2e_sums and e2e_counts and e2e_counts[-1]:
        engine_agg["avg_e2e_latency_s"] = round(e2e_sums[-1] / e2e_counts[-1], 4)

    return {
        "label": label,
        "concurrency": concurrency,
        "show_spk": show_spk,
        "show_emotion": show_emotion,
        "files": len(files),
        "ok": len(ok),
        "failed": len(failed),
        "wall_s": round(wall_s, 2),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "latency_max_s": round(max(latencies), 2) if latencies else None,
        "segments_total": total_segments,
        "segments_per_s": round(total_segments / wall_s, 2) if wall_s else 0.0,
        "chars_total": total_chars,
        "chars_per_s": round(total_chars / wall_s, 1) if wall_s else 0.0,
        "audio_minutes_per_s": round(audio_seconds / wall_s / 60, 3) if wall_s else 0.0,
        "gpu_util_avg": {k: round(statistics.mean(v), 1) for k, v in gpu_util.items()},
        "gpu_util_max": {k: max(v) for k, v in gpu_util.items()},
        "gpu_mem_max_mb": {k: max(v) for k, v in gpu_mem.items()},
        "qwen_container_cpu_cores": {
            "avg": round(statistics.mean(cpu_samples), 2) if cpu_samples else None,
            "max": round(max(cpu_samples), 2) if cpu_samples else None,
        },
        "engine": engine_agg,
        "quality": agg_quality,
        "failures": [
            {"file": f["file"], "code": f["code"], "msg": f["msg"], "error": f["error"]}
            for f in failed
        ],
        "per_file": [
            {
                "file": item["file"],
                "elapsed_s": item["elapsed_s"],
                "segments": len(item["payload"].get("segments") or []),
                "language": item["language"],
                "language_ratio": item["language_ratio"],
                "quality": analyze_payload(item["payload"]),
            }
            for item in results
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8083")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8806/metrics")
    parser.add_argument("--container", default="qwen3-asr-0.6b-8806")
    parser.add_argument("--audio-dir", default=str(REPO_ROOT / "test" / "audio_file"))
    parser.add_argument("--concurrency", default="4,8,16,32")
    parser.add_argument("--mode", choices=["no-role", "role", "both"], default="no-role")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(args.audio_dir)
    pool = []
    skipped = []
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file():
            continue
        if _duration_min(path.name) < 600:
            continue
        # 剔除静音录音：VAD 切不出段，测出来只是浪费并发槽
        mean_db = _mean_volume_db(path.name)
        if mean_db < -50:
            skipped.append((path.name, mean_db))
            continue
        pool.append(str(path))
    print(f"音频池可用文件数：{len(pool)}")
    for name, db in skipped:
        print(f"  跳过静音文件：{name}（平均音量 {db} dB）")

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    show_spk = args.mode in ("role", "both")
    show_emotion = args.mode == "both"

    all_results = []
    for level in levels:
        for round_index in range(args.rounds):
            files = pool[:level]
            label = f"{args.mode}_r{round_index + 1}"
            print(f"=== 并发 {level}（{args.mode}）第 {round_index + 1} 轮 ===", flush=True)
            result = run_level(
                args.base_url,
                files,
                level,
                show_spk,
                show_emotion,
                out_dir,
                label,
                args.metrics_url,
                args.container,
            )
            result["round"] = round_index + 1
            all_results.append(result)
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "per_file"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            with open(
                out_dir / f"result_{label}_c{level}.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)
    print(f"全部结果已写入 {out_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
