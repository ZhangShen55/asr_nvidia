import asyncio
import unittest

from core.vad_pool import VadPool, VadPoolError


FAKE_BUILDER = "tests.fake_vad:build_fake_model"
FAKE_RUNNER = "tests.fake_vad:run_vad_task"


def make_pool(devices):
    return VadPool(
        devices=devices,
        model_dir="./model/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        builder_spec=FAKE_BUILDER,
        runner_spec=FAKE_RUNNER,
        start_timeout_s=10,
    )


class StaticVadPoolTests(unittest.TestCase):
    def test_device_array_is_fixed_and_fifo_waits(self):
        async def scenario():
            pool = make_pool(["cpu", "cpu"])
            try:
                await pool.start()
                self.assertEqual(pool.stats()["configured_instances"], 2)
                self.assertEqual(pool.stats()["successful_instances"], 2)
                tasks = [asyncio.create_task(pool.detect(f"sleep:0.2")) for _ in range(3)]
                await asyncio.sleep(0.05)
                self.assertEqual(pool.stats()["workers"], 2)
                self.assertGreaterEqual(pool.stats()["waiting"], 1)
                await asyncio.gather(*tasks)
                self.assertEqual(pool.stats()["waiting"], 0)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_cancelled_waiting_request_is_removed(self):
        async def scenario():
            pool = make_pool(["cpu"])
            try:
                await pool.start()
                first = asyncio.create_task(pool.detect("sleep:0.5"))
                await asyncio.sleep(0.1)
                second = asyncio.create_task(pool.detect("ok"))
                await asyncio.sleep(0.05)
                self.assertEqual(pool.stats()["waiting"], 1)
                second.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await second
                self.assertEqual(pool.stats()["waiting"], 0)
                await first
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_empty_device_array_is_rejected(self):
        async def scenario():
            pool = make_pool([])
            with self.assertRaises(VadPoolError):
                await pool.start()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
