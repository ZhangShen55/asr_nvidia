"""服务日志初始化：按天 + 按大小双轮转，并按保留天数清理历史文件。"""

import logging
import os
import sys
from datetime import date, datetime, timedelta

from core.config import settings


class _HotwordFilter(logging.Filter):
    def filter(self, record):
        return ('Attempting to parse hotwords' not in record.getMessage() and
                'Hotword list:' not in record.getMessage() and 'rtf_avg:' not in record.getMessage())


class RotatingSizeAndDayFileHandler(logging.FileHandler):
    """同时按大小与按天轮转的文件 handler。

    按大小：写入前若「当前文件大小 + 本条记录大小」超过 max_size_mb，
    把当前文件切出为 <日志名>.1，原 .1 变成 .2 依次顺延，
    超过保留个数的最旧文件直接删除。

    按天：跨天后的首次写入前，把当天的日志切出为 <日志名>.YYYY-MM-DD。

    按保留期：启动时与每次轮转后扫描日志目录，删除最后修改时间
    超过 max_days 的历史日志文件。

    轮转后必须重新打开文件句柄：Linux 下 rename 不会改变已打开的 fd，
    否则后续写入仍会落到已被改名的旧文件里。
    """

    def __init__(
        self,
        log_path: str,
        max_size_mb: int = 100,
        max_days: int = 7,
        max_files: int = 10,
    ):
        super().__init__(log_path, encoding="utf-8")
        self.base_path = os.path.abspath(log_path)
        self.directory = os.path.dirname(self.base_path) or "."
        self.max_size_bytes = max(1, int(max_size_mb)) * 1024 * 1024
        self.max_days = max(1, int(max_days))
        self.max_files = max(1, int(max_files))
        self._current_date = date.today()
        self._cleanup_expired()

    def _rotate_by_size(self):
        """按大小切出 .1 / .2 等文件，序号递增，最旧的被删除。"""
        oldest = f"{self.base_path}.{self.max_files}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for index in range(self.max_files - 1, 0, -1):
            source = f"{self.base_path}.{index}"
            if os.path.exists(source):
                os.replace(source, f"{self.base_path}.{index + 1}")
        if os.path.exists(self.base_path):
            os.replace(self.base_path, f"{self.base_path}.1")

    def _rotate_by_day(self):
        """按天切出 <日志名>.YYYY-MM-DD。"""
        target = f"{self.base_path}.{self._current_date.isoformat()}"
        if os.path.exists(target):
            index = 1
            while os.path.exists(f"{target}.{index}"):
                index += 1
            target = f"{target}.{index}"
        if os.path.exists(self.base_path):
            os.replace(self.base_path, target)

    def _cleanup_expired(self):
        """删除超过保留天数的历史日志文件。"""
        cutoff = datetime.now() - timedelta(days=self.max_days)
        prefix = os.path.basename(self.base_path)
        try:
            names = os.listdir(self.directory)
        except OSError:
            return
        for name in names:
            if not name.startswith(prefix):
                continue
            path = os.path.join(self.directory, name)
            if path == self.base_path or not os.path.isfile(path):
                continue
            try:
                if datetime.fromtimestamp(os.path.getmtime(path)) < cutoff:
                    os.remove(path)
            except OSError:
                continue

    def _reopen(self):
        """轮转后重新打开当前日志文件。"""
        if self.stream:
            self.stream.close()
            self.stream = None

    def _current_size(self):
        try:
            return os.path.getsize(self.base_path)
        except OSError:
            return 0

    def _maybe_rotate(self, record):
        today = date.today()
        rotated = False

        if today != self._current_date:
            if self._current_size() > 0:
                self._rotate_by_day()
                rotated = True
            self._current_date = today

        try:
            record_size = len(self.format(record).encode("utf-8")) + 1
        except Exception:  # noqa: BLE001 - 格式化异常时按 0 处理
            record_size = 0
        if self._current_size() + record_size > self.max_size_bytes:
            self._rotate_by_size()
            rotated = True

        if rotated:
            self._reopen()
            self._cleanup_expired()

    def emit(self, record):
        try:
            self._maybe_rotate(record)
            super().emit(record)
        except Exception:  # noqa: BLE001 - 日志失败不应影响业务
            self.handleError(record)


def setup_logging(log_path: str):
    # 1. 清空旧 handler
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    # 2. 创建两个真正的 handler
    file_hdl = RotatingSizeAndDayFileHandler(
        log_path,
        max_size_mb=settings.log_max_size_mb,
        max_days=settings.log_max_days,
    )
    console_hdl = logging.StreamHandler(sys.stdout)

    # 3. 将过滤器挂到 handler上
    file_hdl.addFilter(_HotwordFilter())
    console_hdl.addFilter(_HotwordFilter())

    # 4. 配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_hdl, console_hdl]
    )

    # 日志降噪
    logging.getLogger("ai-voice-analysis-service").setLevel(logging.WARNING)
    logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)
