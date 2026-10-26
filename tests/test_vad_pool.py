"""VAD worker 进程池的单元测试：创建回收、扩缩容、超时与失败语义。"""

from __future__ import annotations

import asyncio
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from core.config import Settings
from core.config import settings as app_settings
from core.vad import _parse_vad_result
from core.vad_pool import (
    VadPool,
    VadPoolError,
    VadQueueTimeoutError,
)

FAKE_BUILDER = "tests.fake_vad:build_fake_model"
FAKE_RUNNER = "tests.fake_vad:run_vad_task"


@contextmanager
def with_vad_device(value):
    """临时改写 vad_device（配置属性没有 setter，直接改底层字典）。"""
    section = app_settings._cfg.setdefault("qwen_asr", {})
    missing = object()
    old = section.get("vad_device", missing)
    if value is None:
        section.pop("vad_device", None)
    else:
        section["vad_device"] = value
    try:
        yield
    finally:
        if old is missing:
            section.pop("vad_device", None)
        else:
            section["vad_device"] = old


def make_pool(**overrides) -> VadPool:
    params = dict(
        pool_size=1,
        pool_max=4,
        watermark=1,
        collapse_timeout_s=30.0,
        queue_timeout_s=5.0,
        device="cpu",
        model_dir="./model/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000, "max_end_silence_time": 800},
        builder_spec=FAKE_BUILDER,
        runner_spec=FAKE_RUNNER,
        start_timeout_s=30.0,
    )
    params.update(overrides)
    return VadPool(**params)


class VadPoolTests(unittest.TestCase):
    def test_detect_returns_segments_and_parses_like_before(self):
        async def scenario():
            pool = make_pool(pool_size=2)
            try:
                await pool.start()
                self.assertEqual(pool.stats()["workers"], 2)
                raw = await pool.detect("ok")
                segments = _parse_vad_result(raw)
                self.assertEqual(len(segments), 2)
                self.assertEqual(segments[0].start_s, 0.0)
                self.assertEqual(segments[0].end_s, 1.0)
                self.assertEqual(segments[1].start_s, 2.0)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_energy_and_snr_kwargs_reach_fake_worker(self):
        pool = make_pool(
            vad_kwargs={
                "max_single_segment_time": 30000,
                "max_end_silence_time": 800,
                "speech_noise_thres": 0.95,
                "decibel_thres": -42.0,
                "snr_thres": 8.0,
            }
        )
        self.assertEqual(pool._vad_kwargs["decibel_thres"], -42.0)
        self.assertEqual(pool._vad_kwargs["snr_thres"], 8.0)

    def test_worker_process_name_is_distinct(self):
        async def scenario():
            pool = make_pool()
            try:
                raw = await pool.detect("procname")
                self.assertIn("asr-offline-vad", raw[0]["key"])
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_parallel_requests_do_not_serialize(self):
        async def scenario():
            pool = make_pool(pool_size=4, queue_timeout_s=30.0)
            try:
                started = time.perf_counter()
                await asyncio.gather(*[pool.detect("sleep:0.5") for _ in range(4)])
                elapsed = time.perf_counter() - started
                # 4 路并行：墙钟应远低于串行的 2 秒
                self.assertLess(elapsed, 1.5)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_waiting_queue_triggers_scale_up_then_collapse(self):
        async def scenario():
            pool = make_pool(
                pool_size=1,
                pool_max=3,
                watermark=1,
                collapse_timeout_s=1.0,
                queue_timeout_s=30.0,
            )
            try:
                await asyncio.gather(*[pool.detect("sleep:0.6") for _ in range(3)])
                # 三个请求同时压上来：1 个常驻 + 2 个扩容
                self.assertEqual(pool.stats()["workers"], 3)

                deadline = time.monotonic() + 8.0
                while pool.stats()["workers"] > 1 and time.monotonic() < deadline:
                    await asyncio.sleep(0.2)
                self.assertEqual(pool.stats()["workers"], 1)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_queue_timeout_returns_4014_and_others_survive(self):
        async def scenario():
            # pool_max=1：不允许扩容，才能构造出真正的排队
            pool = make_pool(pool_size=1, pool_max=1, watermark=0, queue_timeout_s=0.3)
            try:
                first = asyncio.create_task(pool.detect("sleep:1.0"))
                await asyncio.sleep(0.1)  # 确保 first 已占住唯一的 worker
                with self.assertRaises(VadQueueTimeoutError) as ctx:
                    await pool.detect("ok")
                self.assertEqual(ctx.exception.code, 4014)
                self.assertIn("排队超时", ctx.exception.message)

                # 占住 worker 的那个请求同样按调用超时失败，不会无限挂起
                with self.assertRaises(VadQueueTimeoutError) as ctx:
                    await first
                self.assertIn("切分超时", ctx.exception.message)

                # 超时请求不影响后续请求
                await asyncio.sleep(1.0)
                self.assertEqual(len(_parse_vad_result(await pool.detect("ok"))), 2)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_task_error_fails_request_and_worker_survives(self):
        async def scenario():
            pool = make_pool()
            try:
                with self.assertRaises(VadPoolError) as ctx:
                    await pool.detect("error")
                self.assertIn("模拟 VAD 任务失败", str(ctx.exception))
                # 单次任务失败不杀 worker：后续请求照常成功
                self.assertEqual(len(_parse_vad_result(await pool.detect("ok"))), 2)
                self.assertEqual(pool.stats()["workers"], 1)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_worker_crash_fails_request_and_is_replaced(self):
        async def scenario():
            pool = make_pool()
            try:
                with self.assertRaises(VadPoolError) as ctx:
                    await pool.detect("crash")
                self.assertIn("异常退出", str(ctx.exception))

                deadline = time.monotonic() + 10.0
                while pool.stats()["workers"] < 1 and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                raw = await pool.detect("ok")
                self.assertEqual(len(_parse_vad_result(raw)), 2)
                self.assertEqual(pool.stats()["workers"], 1)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_stats_expose_pool_bounds(self):
        async def scenario():
            pool = make_pool(pool_size=2, pool_max=5)
            try:
                await pool.start()
                stats = pool.stats()
                self.assertEqual(stats["device"], "cpu")
                self.assertEqual(stats["pool_size"], 2)
                self.assertEqual(stats["pool_max"], 5)
                self.assertEqual(stats["workers"], 2)
            finally:
                await pool.stop()

        asyncio.run(scenario())


class VadDeviceTests(unittest.TestCase):
    def test_device_resolution_paths(self):
        from core.models import resolve_vad_device

        with with_vad_device("cpu"):
            self.assertEqual(resolve_vad_device(), "cpu")
        with with_vad_device("cuda:1"):
            with patch("core.models.torch.cuda.is_available", return_value=True), \
                    patch("core.models.torch.cuda.device_count", return_value=2):
                self.assertEqual(resolve_vad_device(), "cuda:1")
        with with_vad_device("cuda:9"):
            with patch("core.models.torch.cuda.is_available", return_value=True), \
                    patch("core.models.torch.cuda.device_count", return_value=2):
                with self.assertRaises(RuntimeError):
                    resolve_vad_device()

    def test_cpu_mode_needs_no_cuda(self):
        from core.models import resolve_vad_device

        with with_vad_device("cpu"), \
                patch("core.models.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_vad_device(), "cpu")


class VadConfigDefaultsTests(unittest.TestCase):
    def test_defaults_match_spec(self):
        cfg = Settings(config_path="./config.toml")
        self.assertEqual(cfg.vad_device, "cpu")
        self.assertEqual(cfg.qwen_vad_pool_size, 8)
        self.assertEqual(cfg.qwen_vad_pool_max, 16)
        self.assertEqual(cfg.qwen_vad_queue_watermark, 4)
        self.assertEqual(cfg.qwen_vad_collapse_timeout_s, 120.0)
        self.assertEqual(cfg.qwen_vad_queue_timeout_s, 120.0)
        self.assertEqual(cfg.qwen_vad_speech_noise_thres, 0.6)

    def test_defaults_when_keys_are_absent(self):
        cfg = Settings.__new__(Settings)
        cfg._cfg = {}
        cfg.config_path = ""
        self.assertEqual(cfg.vad_device, "cpu")
        self.assertEqual(cfg.qwen_vad_pool_size, 8)
        self.assertEqual(cfg.qwen_vad_pool_max, 16)
        self.assertEqual(cfg.qwen_vad_queue_watermark, 4)
        self.assertEqual(cfg.qwen_vad_collapse_timeout_s, 120.0)
        self.assertEqual(cfg.qwen_vad_queue_timeout_s, 120.0)
        self.assertEqual(cfg.qwen_vad_speech_noise_thres, 0.6)

    def test_threshold_range_is_validated(self):
        cfg = Settings.__new__(Settings)
        cfg._cfg = {"qwen_asr": {"vad_speech_noise_thres": 0.3}}
        self.assertEqual(cfg.qwen_vad_speech_noise_thres, 0.3)
        cfg._cfg["qwen_asr"]["vad_speech_noise_thres"] = 1.1
        with self.assertRaises(ValueError):
            _ = cfg.qwen_vad_speech_noise_thres

    def test_pool_default_forwards_threshold(self):
        pool = make_pool(vad_kwargs=None)
        self.assertEqual(pool._vad_kwargs["speech_noise_thres"], 0.6)


if __name__ == "__main__":
    unittest.main()
