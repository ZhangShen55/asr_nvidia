"""运行时资源诊断与安全闸门。

本模块只负责读取系统和服务状态，不改变 ASR 请求路径。所有无法读取的指标
都使用 ``available=False`` 标记，避免把采集失败误报成零值。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GIB = 1024 * BYTES_PER_MIB
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "seacraft_asr"
_METRIC_RE = re.compile(r"^(?P<key>[^:]+):\s*(?P<value>[-+]?\d+)\s*(?P<unit>\w+)?$")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    except (FileNotFoundError, OSError):
        return {}
    return values


def _read_whitespace_file(path: Path) -> dict[str, str]:
    """读取 ``key value`` 形式的内核统计文件，例如 ``/proc/vmstat``。"""
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0]] = parts[1]
    except (FileNotFoundError, OSError):
        return {}
    return values


def _parse_kib(value: str | None) -> int | None:
    if value is None:
        return None
    parts = value.split()
    try:
        number = int(parts[0])
    except (IndexError, ValueError):
        return None
    unit = parts[1].lower() if len(parts) > 1 else "kb"
    if unit in {"kb", "kib"}:
        return number * 1024
    if unit in {"mb", "mib"}:
        return number * BYTES_PER_MIB
    if unit in {"gb", "gib"}:
        return number * BYTES_PER_GIB
    return number


def host_memory_snapshot() -> dict[str, Any]:
    values = _read_key_value_file(Path("/proc/meminfo"))
    total = _parse_kib(values.get("MemTotal"))
    available = _parse_kib(values.get("MemAvailable"))
    free = _parse_kib(values.get("MemFree"))
    swap_total = _parse_kib(values.get("SwapTotal"))
    swap_free = _parse_kib(values.get("SwapFree"))
    if total is None or available is None:
        return {"available": False, "reason": "读取 /proc/meminfo 失败"}
    swap_used = None
    if swap_total is not None and swap_free is not None:
        swap_used = max(0, swap_total - swap_free)
    return {
        "available": True,
        "total_bytes": total,
        "available_bytes": available,
        "free_bytes": free,
        "available_ratio": available / total if total else None,
        "swap_total_bytes": swap_total,
        "swap_free_bytes": swap_free,
        "swap_used_bytes": swap_used,
    }


def vmstat_snapshot() -> dict[str, Any]:
    values = _read_whitespace_file(Path("/proc/vmstat"))
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError):
        page_size = 4096
    if "pswpin" not in values or "pswpout" not in values:
        return {"available": False, "reason": "读取 /proc/vmstat 失败"}
    try:
        swap_in_pages = int(values["pswpin"])
        swap_out_pages = int(values["pswpout"])
    except ValueError:
        return {"available": False, "reason": "解析 /proc/vmstat 失败"}
    return {
        "available": True,
        "page_size_bytes": page_size,
        "swap_in_pages": swap_in_pages,
        "swap_out_pages": swap_out_pages,
        "swap_in_bytes": swap_in_pages * page_size,
        "swap_out_bytes": swap_out_pages * page_size,
    }


def _process_status(pid: int) -> dict[str, Any]:
    values = _read_key_value_file(Path(f"/proc/{pid}/status"))
    if not values:
        return {"available": False, "pid": pid, "reason": "进程不存在或无权限"}
    def number(key: str) -> int | None:
        return _parse_kib(values.get(key))

    try:
        threads = int(values.get("Threads", "0"))
    except ValueError:
        threads = None
    return {
        "available": True,
        "pid": pid,
        "state": values.get("State"),
        "name": values.get("Name"),
        "ppid": int(values.get("PPid", "0")),
        "rss_bytes": number("VmRSS"),
        "swap_bytes": number("VmSwap"),
        "threads": threads,
        "alive": True,
    }


def _process_memory(pid: int) -> dict[str, Any]:
    values = _read_key_value_file(Path(f"/proc/{pid}/smaps_rollup"))
    if not values:
        return {"available": False, "reason": "smaps_rollup 不可用"}
    result: dict[str, Any] = {"available": True}
    for key in ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Swap"):
        result[f"{key.lower()}_bytes"] = _parse_kib(values.get(key))
    return result


def _fd_count(pid: int) -> int | None:
    try:
        return len(list(Path(f"/proc/{pid}/fd").iterdir()))
    except OSError:
        return None


def process_snapshot(pid: int) -> dict[str, Any]:
    result = _process_status(pid)
    if not result.get("available"):
        return result
    result["memory"] = _process_memory(pid)
    result["fd_count"] = _fd_count(pid)
    return result


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def process_tree(root_pid: int | None = None) -> list[dict[str, Any]]:
    """返回目标进程及其后代的资源快照。"""
    proc_dir = Path("/proc")
    rows: dict[int, dict[str, Any]] = {}
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            status = _process_status(pid)
        except (OSError, ValueError):
            continue
        if status.get("available"):
            status["cmdline"] = _cmdline(pid)
            rows[pid] = status
    if root_pid is None:
        return [process_snapshot(pid) | {"cmdline": row.get("cmdline", "")} for pid, row in rows.items()]
    selected: set[int] = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if pid not in selected and row.get("ppid") in selected:
                selected.add(pid)
                changed = True
    return [
        process_snapshot(pid) | {"cmdline": rows[pid].get("cmdline", "")}
        for pid in sorted(selected)
        if pid in rows
    ]


def find_processes(patterns: Iterable[str]) -> list[dict[str, Any]]:
    patterns = tuple(patterns)
    matches = []
    # 进程发现只需读取 status/cmdline；不要为每个系统进程读取 smaps_rollup，
    # 否则 /get_status 会把一次轻量查询变成数百次内核文件扫描。
    for row in _process_listing():
        text = f"{row.get('name', '')} {row.get('cmdline', '')}"
        if any(pattern in text for pattern in patterns):
            matches.append(row)
    return matches


def _process_listing() -> list[dict[str, Any]]:
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            row = _process_status(pid)
        except (OSError, ValueError):
            continue
        if row.get("available"):
            row["cmdline"] = _cmdline(pid)
            rows.append(row)
    return rows


def _parse_csv_line(line: str, fields: tuple[str, ...]) -> dict[str, Any]:
    parts = [part.strip() for part in line.split(",")]
    row: dict[str, Any] = {}
    for field, value in zip(fields, parts):
        if value in {"N/A", "[Not Supported]", ""}:
            row[field] = None
        elif field in {"index", "memory_total_mb", "memory_used_mb", "memory_free_mb", "utilization_gpu"}:
            try:
                row[field] = int(float(value))
            except ValueError:
                row[field] = None
        else:
            row[field] = value
    return row


def gpu_snapshot() -> dict[str, Any]:
    fields = ("index", "name", "memory_total_mb", "memory_used_mb", "memory_free_mb", "utilization_gpu")
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"nvidia-smi 不可用: {exc}"}
    if result.returncode != 0:
        return {"available": False, "reason": result.stderr.strip() or f"exit={result.returncode}"}
    gpus = [_parse_csv_line(line, fields) for line in result.stdout.splitlines() if line.strip()]
    if not gpus:
        return {"available": False, "reason": "nvidia-smi 未返回 GPU"}
    apps_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    apps: list[dict[str, Any]] = []
    try:
        apps_result = subprocess.run(apps_command, capture_output=True, text=True, timeout=2, check=False)
        if apps_result.returncode == 0:
            for line in apps_result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 4:
                    try:
                        pid = int(parts[1])
                    except ValueError:
                        pid = None
                    try:
                        memory_mb = int(float(parts[3]))
                    except ValueError:
                        memory_mb = None
                    apps.append({"gpu_uuid": parts[0], "pid": pid, "process_name": parts[2], "memory_mb": memory_mb})
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"available": True, "gpus": gpus, "compute_apps": apps}


def cache_snapshot(cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    root = Path(cache_dir)
    if not root.exists():
        return {"available": True, "path": str(root), "files": 0, "bytes": 0}
    if not root.is_dir():
        return {"available": False, "path": str(root), "reason": "缓存路径不是目录"}
    files = 0
    total = 0
    for path in root.iterdir():
        try:
            if path.is_file() and path.name.startswith("seacraft-asr-"):
                files += 1
                total += path.stat().st_size
        except OSError:
            continue
    usage = shutil.disk_usage(root)
    return {
        "available": True,
        "path": str(root),
        "files": files,
        "bytes": total,
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "disk_free_ratio": usage.free / usage.total if usage.total else None,
    }


def endpoint_health(url: str, timeout_s: float = 2.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read(4096)
            return {
                "available": True,
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "body_bytes": len(body),
            }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "available": False,
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "reason": str(exc),
        }


def service_processes() -> dict[str, Any]:
    backend = find_processes(("asr-offline-model", "uvicorn"))
    vad = find_processes(("asr-offline-vad",))
    qwen = find_processes(("qwen-asr-serve", "vllm"))
    return {"backend": backend, "vad": vad, "qwen": qwen}


def collect_snapshot(
    *,
    backend_pid: int | None = None,
    backend_url: str = "http://127.0.0.1:8085/get_status",
    qwen_url: str = "http://127.0.0.1:8806/v1/models",
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    service_status: dict[str, Any] | None = None,
    include_health: bool = True,
    include_process_tree: bool = True,
) -> dict[str, Any]:
    if backend_pid is None:
        candidates = find_processes(("asr-offline-model",))
        backend_pid = candidates[0]["pid"] if candidates else None
    backend = process_snapshot(backend_pid) if backend_pid else {"available": False, "reason": "未找到后端 PID"}
    tree = process_tree(backend_pid) if backend_pid and include_process_tree else []
    result = {
        "timestamp": utc_timestamp(),
        "host": host_memory_snapshot(),
        "vmstat": vmstat_snapshot(),
        "backend": backend,
        "process_tree": tree,
        "services": service_processes(),
        "gpu": gpu_snapshot(),
        "cache": cache_snapshot(cache_dir),
        "status": service_status,
    }
    if include_health:
        result["health"] = {
            "backend": endpoint_health(backend_url),
            "qwen": endpoint_health(qwen_url),
        }
    else:
        result["health"] = {"available": False, "reason": "状态接口采样跳过自调用健康检查"}
    return result


def status_snapshot(
    *,
    backend_pid: int | None = None,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """生成供 ``/get_status`` 使用的非阻塞资源快照。

    这里不访问 8085 自身，避免状态接口为了报告健康状态而递归调用自己；
    Qwen 健康检查和完整端点检查由独立诊断脚本负责。
    """
    return collect_snapshot(
        backend_pid=backend_pid,
        cache_dir=cache_dir,
        include_health=False,
        include_process_tree=False,
    )


@dataclass
class SafetyPolicy:
    """以测试基线为参照的安全阈值。"""

    min_available_bytes: int = 8 * BYTES_PER_GIB
    min_available_ratio: float = 0.20
    max_swap_delta_bytes: int = 256 * BYTES_PER_MIB
    max_gpu_used_ratio: float = 0.90
    warn_gpu_used_ratio: float = 0.80
    max_disk_used_ratio: float = 0.90
    consecutive_swap_samples: int = 3
    consecutive_health_failures: int = 2


@dataclass
class SafetyBaseline:
    memory: dict[str, Any]
    vmstat: dict[str, Any]
    gpu: dict[str, Any]
    cache: dict[str, Any]

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "SafetyBaseline":
        return cls(
            memory=snapshot.get("host", {}),
            vmstat=snapshot.get("vmstat", {}),
            gpu=snapshot.get("gpu", {}),
            cache=snapshot.get("cache", {}),
        )


@dataclass
class SafetyDecision:
    level: str = "green"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def should_stop(self) -> bool:
        return self.level == "red"


def evaluate_safety(
    snapshot: dict[str, Any],
    baseline: SafetyBaseline,
    policy: SafetyPolicy | None = None,
    *,
    swap_samples: int = 0,
    health_failures: int = 0,
    expected_vad_workers: int | None = None,
) -> SafetyDecision:
    policy = policy or SafetyPolicy()
    decision = SafetyDecision()
    host = snapshot.get("host", {})
    base_host = baseline.memory
    if host.get("available"):
        available = host.get("available_bytes")
        total = host.get("total_bytes")
        if isinstance(available, int) and isinstance(total, int):
            hard_floor = max(policy.min_available_bytes, int(total * policy.min_available_ratio))
            if available < hard_floor:
                decision.reasons.append(f"可用内存低于安全水位: {available}/{hard_floor} bytes")
            elif available < int(total * 0.30):
                decision.warnings.append("可用内存低于 30%")
        swap_used = host.get("swap_used_bytes")
        base_swap = base_host.get("swap_used_bytes")
        if isinstance(swap_used, int) and isinstance(base_swap, int):
            if swap_used - base_swap > policy.max_swap_delta_bytes:
                decision.reasons.append("测试期间 Swap 增量超过安全阈值")
    vm = snapshot.get("vmstat", {})
    base_vm = baseline.vmstat
    if vm.get("available") and base_vm.get("available") and swap_samples >= policy.consecutive_swap_samples:
        if vm.get("swap_in_pages", 0) > base_vm.get("swap_in_pages", 0) or vm.get("swap_out_pages", 0) > base_vm.get("swap_out_pages", 0):
            decision.reasons.append("连续采样检测到 Swap I/O")
    gpu = snapshot.get("gpu", {})
    if gpu.get("available"):
        for row in gpu.get("gpus", []):
            total = row.get("memory_total_mb")
            used = row.get("memory_used_mb")
            if not isinstance(total, int) or not isinstance(used, int) or total <= 0:
                continue
            ratio = used / total
            if ratio >= policy.max_gpu_used_ratio:
                decision.reasons.append(f"GPU {row.get('index')} 显存使用率达到 {ratio:.1%}")
            elif ratio >= policy.warn_gpu_used_ratio:
                decision.warnings.append(f"GPU {row.get('index')} 显存使用率达到 {ratio:.1%}")
    cache = snapshot.get("cache", {})
    if cache.get("available") and cache.get("disk_total_bytes"):
        used_ratio = 1 - cache.get("disk_free_ratio", 1)
        if used_ratio >= policy.max_disk_used_ratio:
            decision.reasons.append("服务缓存所在磁盘空间不足")
    health = snapshot.get("health", {})
    if health_failures >= policy.consecutive_health_failures:
        decision.reasons.append("后端健康检查连续失败")
    if expected_vad_workers is not None:
        vad_count = len(snapshot.get("services", {}).get("vad", []))
        if vad_count < expected_vad_workers:
            decision.reasons.append(f"VAD worker 数量异常: {vad_count}/{expected_vad_workers}")
    if decision.reasons:
        decision.level = "red"
    elif decision.warnings:
        decision.level = "yellow"
    return decision


def write_json_line(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


__all__ = [
    "BYTES_PER_GIB",
    "BYTES_PER_MIB",
    "DEFAULT_CACHE_DIR",
    "SafetyBaseline",
    "SafetyDecision",
    "SafetyPolicy",
    "cache_snapshot",
    "collect_snapshot",
    "endpoint_health",
    "evaluate_safety",
    "find_processes",
    "gpu_snapshot",
    "host_memory_snapshot",
    "process_snapshot",
    "process_tree",
    "service_processes",
    "status_snapshot",
    "utc_timestamp",
    "vmstat_snapshot",
    "write_json_line",
]
