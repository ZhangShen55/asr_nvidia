"""CPU-only: request/status mocks only, no listener, worker, model or NPU."""
import asyncio
from contextlib import redirect_stdout
import hashlib
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

from scripts import validate_http_audio_admission as tool


def state(total=0, enhanced=0):
    return {"audioConcurrency": {"active_total": total}, "spkEmoConcurrency": {"active": enhanced},
            "resourceDiagnostics": {"segmentPipeline": {"active": 999}}}


class MockCapacityService:
    def __init__(self):
        self.total = self.enhanced = 0
        self.releases = {"B4": asyncio.Event(), "total8": asyncio.Event()}
        self.probes = {"B4": 0, "total8": 0}
        self.peaks = {"B4": 0, "total8": 0}

    async def status(self):
        return state(self.total, self.enhanced)

    async def request(self, label, kind):
        stage = label.split("-", 1)[0]
        if "-base-" in label:
            self.total += 1
            self.enhanced += int(kind == "B")
            self.peaks[stage] = max(self.peaks[stage], self.total)
            await self.releases[stage].wait()
            self.total -= 1
            self.enhanced -= int(kind == "B")
            return {"http_status": 200, "payload": {"code": 0, "segments": []}}
        code = 4013 if self.total == 8 else 4017 if kind == "B" and self.enhanced == 4 else 0
        self.probes[stage] += 1
        if self.probes[stage] == (1 if stage == "B4" else 2):
            self.releases[stage].set()
        return {"http_status": 200, "payload": {"code": code, "raw_marker": label}}


class AdmissionAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_count_barriers_and_total_limit_priority(self):
        fake = MockCapacityService()
        record = {}
        result = await tool.conduct_checks(fake.request, fake.status, record, observation_s=.1, poll_s=.001)
        self.assertEqual(result, "passed")
        self.assertEqual(fake.peaks, {"B4": 4, "total8": 8})
        self.assertEqual([p["expected_code"] for s in record["stages"] for p in s["probes"]], [4017, 4013, 4013])
        self.assertTrue(all(s["capacity_observation"]["observed"] and s["drained"]["observed"] for s in record["stages"]))
        self.assertTrue(all(p["raw_request"]["payload"]["raw_marker"] for s in record["stages"] for p in s["probes"]))
        self.assertEqual(len(record["requests"]), 15)

    async def test_work_finishes_too_fast_is_not_observed_not_passed(self):
        async def immediate(*_): return {"http_status": 200, "payload": {"code": 0, "segments": []}}
        async def idle(): return state()
        record = {}
        self.assertEqual(await tool.conduct_checks(immediate, idle, record, observation_s=.02, poll_s=.001), "not_observed")
        self.assertEqual(len(record["requests"]), 12)
        self.assertTrue(all(not s["probes"] for s in record["stages"]))

    async def test_close_keeps_ownership_through_repeated_cancellation(self):
        release = asyncio.Event()
        calls = []
        class Owner:
            identities = {}
            async def close(self, **kwargs):
                calls.append(kwargs)
                await release.wait()
                return {"passed": True}
        task = asyncio.create_task(tool.close_owned(Owner(), 240))
        await asyncio.sleep(0)
        task.cancel(); await asyncio.sleep(0); task.cancel(); await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        self.assertTrue((await task)["passed"])
        self.assertEqual(calls, [{"deadline_s": 240, "observe_s": 5}])

    async def test_failure_still_closes_owned_service_without_http(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.toml"; config.write_text("profile")
            args = SimpleNamespace(output=root/"result", url="http://127.0.0.1:18084", config=config,
                wav=root/"unused.wav", deadline_s=1, shutdown_timeout_s=240, observation_s=.01, poll_s=.001)
            calls = []
            class Owner:
                identities = {}
                process = SimpleNamespace(poll=lambda: None)
                def __init__(self, *_): pass
                def register(self): pass
                async def close(self, **kwargs): calls.append(kwargs); return {"passed": True}
            status_timeouts = []
            class Client:
                def __init__(self, **_): pass
                async def __aenter__(self): return self
                async def __aexit__(self, *_): pass
                async def get(self, *_args, **_kwargs):
                    status_timeouts.append(_kwargs.get("timeout"))
                    return SimpleNamespace(raise_for_status=lambda: None, json=state)
            async def bad_warmup(*_): raise ValueError("injected preheat failure")
            identity = {"complete": True, "files": {}, "root": "mock", "python_executable": "mock", "python_version": "mock"}
            info = {"config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
            with patch.object(tool, "OwnedService", Owner), patch("httpx.AsyncClient", Client), \
                 patch.object(tool, "warmup", bad_warmup), patch.object(tool, "source_identity", return_value=identity), \
                 patch.object(tool, "preserved_identity", return_value={"pid": 1601397, "present": True}):
                result = await tool.execute(args, info)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["shutdown"]["passed"])
            self.assertFalse(result["complete"])
            self.assertEqual(calls, [{"deadline_s": 240, "observe_s": 5}])
            self.assertEqual(status_timeouts, [20.0])


class AdmissionPlanTests(unittest.TestCase):
    def test_status_uses_actual_top_level_fields_and_probe_race_not_passed(self):
        self.assertEqual(tool.capacity(state(8, 4)), (8, 4))
        self.assertEqual(tool.classify_probe({"http_status": 200, "payload": {"code": 0}}, 4013), "not_observed")
        self.assertEqual(tool.classify_probe({"http_status": 200, "payload": {"code": 4017}}, 4013), "not_observed")
        self.assertEqual(tool.classify_probe({"http_status": 200, "payload": {"code": 4011}}, 4013), "failed")

    def test_default_plan_has_no_service_or_output_and_waiting_profiles_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = root/"config.toml"; wav = root/"real.wav"; output = root/"not-created"
            config.write_text("[qwen_asr]\naudio_concurrency=8\naudio_spk_emo_concurrency=4\nqueue_wait_timeout_s=0\nexecution_mode=\"process\"\n")
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "")); audio.writeframes(bytes(16000*2))
            argv = ["--config", str(config), "--wav", str(wav), "--output", str(output)]
            with patch.object(tool, "OwnedService", side_effect=AssertionError("plan started service")), redirect_stdout(StringIO()):
                self.assertEqual(tool.main(argv), 0)
            self.assertFalse(output.exists())
            config.write_text("[qwen_asr]\naudio_concurrency=8\naudio_spk_emo_concurrency=4\nqueue_wait_timeout_s=1\n")
            with self.assertRaisesRegex(ValueError, "queue_wait_timeout_s"):
                tool.plan(tool.arguments(argv))


class StatusTimeoutAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_snapshot_timeout_keeps_all_admitted_results_and_codes(self):
        import httpx
        fake = MockCapacityService()
        failed = set()
        async def status():
            for stage, count in (("B4", 1), ("total8", 2)):
                if fake.probes[stage] == count and stage not in failed:
                    failed.add(stage)
                    raise httpx.ReadTimeout("diagnostic-only snapshot timeout")
            return await fake.status()
        record = {}
        result = await tool.conduct_checks(fake.request, status, record, observation_s=.1, poll_s=.001)
        self.assertEqual(result, "passed")
        self.assertEqual(failed, {"B4", "total8"})
        self.assertEqual(len(record["requests"]), 15)
        self.assertTrue(all(row["status"] == "completed" for row in record["requests"]))
        self.assertTrue(all(stage["base_requests_successful"] and stage["drained"]["observed"] for stage in record["stages"]))
        self.assertEqual([probe["raw_request"]["payload"]["code"] for stage in record["stages"] for probe in stage["probes"]],
                         [4017, 4013, 4013])
        for stage in record["stages"]:
            self.assertIsNone(stage["after_probes_status"])
            self.assertEqual(stage["after_probes_status_observation"]["status"], "unknown")
            self.assertEqual(stage["after_probes_status_observation"]["error_type"], "ReadTimeout")
            self.assertFalse(stage["after_probes_status_observation"]["gate"])

    async def test_key_target_or_drain_read_failure_cannot_pass(self):
        import httpx
        for phase in ("target", "drain"):
            with self.subTest(phase=phase):
                fake = MockCapacityService()
                record = {}
                after_probe_seen = False
                async def status():
                    nonlocal after_probe_seen
                    if phase == "target" and record.get("stages"):
                        raise httpx.ReadTimeout("critical target status timeout")
                    if phase == "drain" and fake.probes["B4"]:
                        if after_probe_seen:
                            raise httpx.ReadTimeout("critical drain status timeout")
                        after_probe_seen = True
                    return await fake.status()
                with self.assertRaises(httpx.ReadTimeout):
                    await tool.conduct_checks(fake.request, status, record, observation_s=.1, poll_s=.001)
                self.assertFalse(any(stage["status"] == "passed" for stage in record["stages"]))


class StatusTimeoutCliTests(unittest.TestCase):
    def test_positive_finite_bounded_status_timeout_and_unchanged_deadline(self):
        argv = ["--config", "fixture.toml", "--wav", "fixture.wav", "--output", "fixture-output"]
        self.assertEqual(tool.arguments(argv).status_timeout_s, 20)
        self.assertEqual(tool.arguments(argv).deadline_s, 420)
        self.assertEqual(tool.arguments(argv + ["--status-timeout-s", "60"]).status_timeout_s, 60)
        for value in ("0", "-1", "nan", "inf", "60.01"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                tool.arguments(argv + ["--status-timeout-s", value])
        with self.assertRaises(SystemExit):
            tool.arguments(argv + ["--deadline-s", "421"])
