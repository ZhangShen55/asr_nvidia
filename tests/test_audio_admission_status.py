import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core import status
from core.config import settings
from core.models import ModelUnavailableError, validate_v2_capabilities
from core.qwen_asr_queue import AudioSlotManager


class AdmissionStatusTests(unittest.IsolatedAsyncioTestCase):
    def payload(self, manager):
        stats = dict(offline=2, online=3, total_failed_tasks=1, queued_task_ids=[], processing_task_ids=[])
        with patch.object(status, "get_slot_manager", return_value=manager), \
                patch.object(manager, "snapshot", wraps=manager.snapshot) as snapshot, \
                patch.object(status, "status_snapshot", return_value={}), \
                patch.object(status, "get_vad_pool", return_value=SimpleNamespace(stats=lambda: {"busy": 0})), \
                patch("core.models.ModelPool.ensure_loaded", new_callable=AsyncMock) as load:
            result = status.build_status_payload(stats, datetime.utcnow())
            snapshot.assert_called_once_with()
            load.assert_not_awaited()
        for key in ("localModels", "vadPool", "resourceDiagnostics", "queuedTaskList", "processingTaskList"):
            self.assertIn(key, result)
        self.assertEqual(result["totalHaveDoneProcessTasks"], 5)
        return result

    async def test_before_first_request_and_disabled_rejection_preserve_zero_counts(self):
        manager = AudioSlotManager(2, spk_emo_capacity=4)
        with patch.dict(settings._cfg, {"features": {"open_v2_role": False, "open_v2_emotion": False}}):
            before = self.payload(manager)
            with self.assertRaises(ModelUnavailableError) as error:
                validate_v2_capabilities(True, True)
            self.assertEqual(error.exception.code, 4018)
            after = self.payload(manager)
        self.assertEqual(before["audioConcurrency"], {"configured_total": 2, "active_total": 0, "waiting_total": 0})
        self.assertEqual(after["audioConcurrency"], before["audioConcurrency"])
        self.assertEqual(after["spkEmoConcurrency"], {"configured": 4, "active": 0, "waiting": 0})

    async def test_mixed_activity_and_waiting_share_one_snapshot(self):
        manager = AudioSlotManager(8, wait_timeout_s=5, spk_emo_capacity=4)
        leases = [await manager.acquire(category) for category in ["A"] * 5 + ["B"] * 3]
        waiting = [asyncio.create_task(manager.acquire(category)) for category in ["A", "B", "B"]]
        try:
            await asyncio.sleep(0)
            result = self.payload(manager)
            self.assertEqual(result["audioConcurrency"], {"configured_total": 8, "active_total": 8, "waiting_total": 3})
            self.assertEqual(result["spkEmoConcurrency"], {"configured": 4, "active": 3, "waiting": 2})
            # 返回对象不能改变发布状态。
            result["audioConcurrency"]["active_total"] = 0
            self.assertEqual(manager.snapshot()["audioConcurrency"]["active_total"], 8)
        finally:
            for task in waiting:
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)
            for lease in leases:
                await manager.release(lease)
        self.assertEqual((manager.active, manager.active_b, manager.waiting), (0, 0, 0))
