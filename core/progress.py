"""可选的每音频段处理进度显示。"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:  # tqdm 是可选的，日志重定向环境不应因进度依赖阻断服务。
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None

_position_lock = threading.Lock()
_next_position = 0


class AudioProgress:
    def __init__(self, total: int, description: str, stream=None):
        self.total = max(0, int(total))
        self.description = description
        self.completed = 0
        self.failed = 0
        self._lock = threading.RLock()
        self._bar = None
        stream = stream or sys.stderr
        enabled = bool(_tqdm is not None and hasattr(stream, "isatty") and stream.isatty())
        if enabled:
            global _next_position
            with _position_lock:
                position = _next_position
                _next_position += 1
            self._bar = _tqdm(
                total=self.total,
                desc=description,
                unit="段",
                position=position,
                leave=True,
                dynamic_ncols=True,
                file=stream,
            )

    @property
    def enabled(self) -> bool:
        return self._bar is not None

    def update(self, failed: bool = False) -> None:
        with self._lock:
            self.completed = min(self.total, self.completed + 1)
            if failed:
                self.failed += 1
            if self._bar is not None:
                self._bar.update(1)
                self._bar.set_postfix_str(f"失败={self.failed}", refresh=False)

    def close(self) -> None:
        with self._lock:
            if self._bar is not None:
                self._bar.close()
                self._bar = None
            else:
                logger.info(
                    "[V2 进度] %s 完成=%d/%d 失败=%d",
                    self.description,
                    self.completed,
                    self.total,
                    self.failed,
                )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "completed": self.completed,
                "total": self.total,
                "failed": self.failed,
                "enabled": self.enabled,
            }


def create_audio_progress(total: int, description: str) -> AudioProgress:
    return AudioProgress(total=total, description=description)


__all__ = ["AudioProgress", "create_audio_progress"]
