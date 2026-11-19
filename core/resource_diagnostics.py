"""运行时资源诊断与安全闸门。

本模块只负责读取系统和服务状态，不改变 ASR 请求路径。所有无法读取的指标
都使用 ``available=False`` 标记，避免把采集失败误报成零值。
"""

from __future__ import annotations

import asyncio
import copy
import math
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


def parse_npu_smi(output: str) -> dict[str, Any]:
    """解析 Ascend ``npu-smi info``，HBM 使用量取 MB 原始值而非百分比估算。

    设备表由两行组成，第三列同时包含 AICore、普通内存和 HBM；进程表另
    列设备、Chip、PID 和 HBM。遇到未知格式保留不可用状态，绝不填充零。
    """
    npus: list[dict[str, Any]] = []
    apps: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for line in output.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4:
            identity = re.fullmatch(r"(\d+)\s+(\d+)", cells[0])
            if identity and cells[1].isdigit():
                apps.append({
                    "index": int(identity[1]), "chip": int(identity[2]),
                    "pid": int(cells[1]), "process_name": cells[2],
                    "memory_mb": int(cells[3]) if cells[3].isdigit() else None,
                })
            continue
        if len(cells) != 3:
            continue
        identity = re.fullmatch(r"(\d+)\s+(\S+)", cells[0])
        if identity and not identity[2].isdigit():
            pending = {"index": int(identity[1]), "name": identity[2], "health": cells[1]}
            continue
        if pending is None or not cells[0].isdigit() or not re.fullmatch(r"[0-9A-Fa-f:.]+", cells[1]):
            continue
        metrics = cells[2].split()
        utilization = int(metrics[0]) if metrics and metrics[0].isdigit() else None
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", cells[2])
        used, total = (int(value) for value in pairs[-1]) if len(pairs) >= 2 else (None, None)
        pending.update({
            "chip": int(cells[0]), "bus_id": cells[1],
            "available": used is not None and total is not None and total > 0,
            "hbm_used_mb": used, "hbm_total_mb": total,
            "hbm_free_mb": max(0, total - used) if total is not None and used is not None else None,
            "utilization_aicore": utilization,
        })
        npus.append(pending)
        pending = None
    available = any(row["available"] for row in npus)
    result: dict[str, Any] = {
        "available": available, "npus": npus, "compute_apps": apps,
        "memory_scope": "npu-smi device HBM aggregate; includes other processes and runtime allocations",
    }
    if not available:
        result["reason"] = "npu-smi 未返回可解析的 HBM 使用量"
    return result


def _process_start_identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        ticks = fields[19]
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        return {"pid": int(pid), "start_ticks": ticks, "sid": int(fields[3]),
                "create_time": boot + int(ticks) / os.sysconf("SC_CLK_TCK")}
    except (OSError, TypeError, ValueError, IndexError, StopIteration):
        return {"pid": pid, "available": False, "reason": "process identity unavailable at resource collection"}


def _npu_process_sample(row: dict[str, Any]) -> dict[str, Any]:
    """采样线程只读OS；核对读取RSS/PSS前后身份，不访问模型或发RPC。"""
    pid = row.get("pid")
    before = _process_start_identity(pid)
    resources = process_snapshot(pid) if type(pid) is int else {"available": False, "reason": "invalid PID"}
    after = _process_start_identity(pid)
    same_birth = (before.get("start_ticks") is not None
                  and before.get("start_ticks") == after.get("start_ticks")
                  and before.get("create_time") == after.get("create_time"))
    if not same_birth:
        after = {"pid": pid, "available": False, "reason": "process disappeared or identity changed during OS sampling"}
        resources = {"available": False, "pid": pid, "reason": after["reason"]}
    return dict(row, process_identity=after, process_resource=resources,
                process_resource_scope="OS RSS/PSS at this resource collection; independent of cached worker allocator/last RPC peaks")


def npu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=2, check=False)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "npus": [], "compute_apps": [], "reason": f"npu-smi 不可用: {exc}"}
    if result.returncode != 0:
        return {"available": False, "npus": [], "compute_apps": [], "reason": result.stderr.strip() or f"exit={result.returncode}"}
    return parse_npu_smi(result.stdout)


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


def service_processes(backend_pid: int | None) -> dict[str, Any]:
    """仅归属明确后端 PID 及其后代，避免把共享卡上的 uvicorn/vLLM 当成本服务。"""
    if backend_pid is None:
        return {"backend": [], "vad": []}
    rows = _process_listing()
    selected = {backend_pid}
    changed = True
    while changed:
        changed = False
        for row in rows:
            if row["pid"] not in selected and row.get("ppid") in selected:
                selected.add(row["pid"])
                changed = True
    return {
        "backend": [row for row in rows if row["pid"] == backend_pid],
        "vad": [row for row in rows if row["pid"] in selected and row["pid"] != backend_pid
                and "asr-offline-vad" in f"{row.get('name', '')} {row.get('cmdline', '')}"],
    }


def collect_snapshot(
    *,
    backend_pid: int | None = None,
    backend_url: str = "http://127.0.0.1:8085/openapi.json",
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    service_status: dict[str, Any] | None = None,
    include_health: bool = True,
    include_process_tree: bool = True,
) -> dict[str, Any]:
    # 独立工具必须提供 PID；没有 PID 时不会猜测共享服务器上的 uvicorn。
    backend = process_snapshot(backend_pid) if backend_pid else {"available": False, "reason": "未提供后端 PID"}
    tree = process_tree(backend_pid) if backend_pid and include_process_tree else []
    services = service_processes(backend_pid)
    owned_pids = {row["pid"] for rows in services.values() for row in rows}
    owned_pids.update(row["pid"] for row in tree)
    npu = npu_snapshot()
    apps = [_npu_process_sample(row) for row in npu.get("compute_apps", [])]
    npu["compute_apps"] = apps
    npu["service_compute_apps"] = [row for row in apps if row.get("pid") in owned_pids]
    npu["other_compute_apps"] = [row for row in apps if row.get("pid") not in owned_pids]
    result = {
        "timestamp": utc_timestamp(),
        "host": host_memory_snapshot(), "vmstat": vmstat_snapshot(),
        "backend": backend, "process_tree": tree, "services": services,
        "qwen_execution": "unknown_without_model_status", "npu": npu,
        "cache": cache_snapshot(cache_dir), "status": service_status,
    }
    if include_health:
        result["health"] = {"backend": endpoint_health(backend_url)}
    else:
        result["health"] = {"available": False, "reason": "状态接口采样跳过自调用健康检查"}
    return result


def status_snapshot(
    *,
    backend_pid: int | None = None,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """供 /get_status 使用；默认归属当前进程，跳过递归 HTTP 和完整进程内存扫描。"""
    return collect_snapshot(
        backend_pid=os.getpid() if backend_pid is None else backend_pid,
        cache_dir=cache_dir, include_health=False, include_process_tree=False,
    )


def unknown_resource_snapshot(reason="资源快照尚未完成"):
    return {"available": False, "reason": reason}


class ResourceSnapshotSampler:
    """app/loop独占的按需TTL缓存；GET不等待collector，不创建永久轮询任务。"""
    def __init__(self, *, interval_s=1.0, collector=None, clock=time.monotonic):
        if type(interval_s) not in (int, float) or not math.isfinite(interval_s) or not .1 <= interval_s <= 60:
            raise ValueError("resource_sample_interval_s必须为0.1～60秒有限数")
        self.interval_s = float(interval_s)
        self._collector = collector or status_snapshot
        self._clock = clock
        self._loop = None
        self._task = None
        self._close_task = None
        self._closing = False
        self._closed = False
        self._cached = None
        self._completed_at = None
        self._next_refresh_at = 0.0
        self._in_progress = False
        self._last_error = None
        self._last_started = None
        self._last_finished = None
        self._last_elapsed = None
        self._attempts = 0

    def _bind_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("资源采样器不能跨app事件循环复用")
        return loop

    def snapshot(self):
        loop = self._bind_loop()
        now = self._clock()
        if not self._closing and not self._in_progress and now >= self._next_refresh_at:
            # 在create_task之前先占singleflight，多个GET不能给同一采样器叠加线程。
            self._in_progress = True
            self._attempts += 1
            self._last_started = now
            self._last_finished = self._last_elapsed = None
            self._task = loop.create_task(self._refresh())
        value = copy.deepcopy(self._cached) if self._cached is not None else unknown_resource_snapshot()
        value["sampling"] = {
            "mode": "on_demand_ttl_singleflight", "interval_s": self.interval_s,
            "has_snapshot": self._cached is not None,
            "age_s": max(0.0, now - self._completed_at) if self._completed_at is not None else None,
            "in_progress": self._in_progress, "last_error": self._last_error,
            "attempts": self._attempts, "closing": self._closing, "closed": self._closed,
            "last_started_monotonic_s": self._last_started,
            "last_finished_monotonic_s": self._last_finished, "last_elapsed_s": self._last_elapsed,
            "last_attempt_timing_scope": "refresh submission through collector/cache completion; includes thread dispatch, not device kernel time",
            "freshness_scope": "last completed resource collection; live model/admission counts assembled separately",
        }
        return value

    async def _refresh(self):
        from core.async_utils import await_completion
        completed = False
        try:
            # 线程无法被协程取消；即便误取消refresh也要等底层真实结束，不能遗留采样。
            value = await await_completion(asyncio.to_thread(self._collector))
            if not isinstance(value, dict):
                raise TypeError("资源collector必须返回dict")
            self._cached = copy.deepcopy(value)
            completed = True
            self._last_error = None
        except asyncio.CancelledError:
            self._last_error = "CancelledError: collector ownership cancelled after actual completion"
            raise
        except Exception as exc:
            # 保留上一完整快照及其age；错误不伪装为最新资源零值。
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            finished = self._clock()
            self._last_finished = finished
            self._last_elapsed = max(0.0, finished - self._last_started)
            if completed:
                self._completed_at = finished
            self._in_progress = False
            self._next_refresh_at = finished + self.interval_s

    async def _close(self):
        from core.async_utils import await_completion
        try:
            if self._task is not None:
                await await_completion(self._task)
        finally:
            self._closed = True

    async def aclose(self):
        from core.async_utils import await_completion
        self._bind_loop()
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await await_completion(self._close_task)


@dataclass
class SafetyPolicy:
    """以测试基线为参照的安全阈值。"""

    min_available_bytes: int = 8 * BYTES_PER_GIB
    min_available_ratio: float = 0.20
    max_swap_delta_bytes: int = 256 * BYTES_PER_MIB
    npu_indices: tuple[int, ...] = (7,)
    max_npu_used_ratio: float = 0.90
    warn_npu_used_ratio: float = 0.80
    max_disk_used_ratio: float = 0.90
    consecutive_swap_samples: int = 3
    consecutive_health_failures: int = 2


@dataclass
class SafetyBaseline:
    memory: dict[str, Any]
    vmstat: dict[str, Any]
    npu: dict[str, Any]
    cache: dict[str, Any]

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "SafetyBaseline":
        return cls(
            memory=snapshot.get("host", {}),
            vmstat=snapshot.get("vmstat", {}),
            npu=snapshot.get("npu", {}),
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
    npu = snapshot.get("npu", {})
    selected = {row.get("index"): row for row in npu.get("npus", []) if row.get("index") in policy.npu_indices}
    for index in policy.npu_indices:
        row = selected.get(index, {})
        total = row.get("hbm_total_mb")
        used = row.get("hbm_used_mb")
        if not npu.get("available") or not isinstance(total, int) or not isinstance(used, int) or total <= 0:
            decision.warnings.append(f"NPU {index} HBM 指标不可用，无法判定安全水位")
            continue
        ratio = used / total
        if ratio >= policy.max_npu_used_ratio:
            decision.reasons.append(f"NPU {index} HBM 使用率达到 {ratio:.1%}")
        elif ratio >= policy.warn_npu_used_ratio:
            decision.warnings.append(f"NPU {index} HBM 使用率达到 {ratio:.1%}")
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
    "npu_snapshot",
    "parse_npu_smi",
    "host_memory_snapshot",
    "process_snapshot",
    "process_tree",
    "service_processes",
    "status_snapshot",
    "ResourceSnapshotSampler",
    "unknown_resource_snapshot",
    "utc_timestamp",
    "vmstat_snapshot",
    "write_json_line",
]
