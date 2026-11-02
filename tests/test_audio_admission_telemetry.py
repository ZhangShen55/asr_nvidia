import unittest

from core import request_telemetry as telemetry
from core.qwen_asr_queue import AudioSlotManager


class AdmissionTelemetryTests(unittest.TestCase):
    def setUp(self):
        telemetry.reset_for_tests()
        self.addCleanup(telemetry.reset_for_tests)

    def record(self, **kwargs):
        return telemetry.record_request(request_id="test", task_id=None, filename="a.wav",
                                        started_at=0, ended_at=1, **kwargs)

    def test_failure_reasons_and_classification_are_counted_separately(self):
        cases = [(4013, "total_full", "rejected"), (4013, "waiting_full", "rejected"),
                 (4017, "spk_emo_full", "rejected"), (4018, "both_disabled", "rejected"),
                 (4010, "wait_timeout", "timeout"), (None, "cancelled", "cancelled")]
        capacity = AudioSlotManager(8, spk_emo_capacity=4).snapshot()
        for index, (code, reason, outcome) in enumerate(cases):
            self.record(code=code, status="cancelled" if code is None else "failed",
                        phases={"audio_wait_ms": index, "qwen_ms": 0},
                        admission={"category": "B", "outcome": outcome, "reason": reason, "snapshot": capacity})
        self.record(admission={"category": "A", "outcome": "admitted", "reason": None, "snapshot": capacity})
        result = telemetry.snapshot()
        self.assertEqual(result["admission"]["window"], "retained_events")
        self.assertEqual(result["admission"]["by_class"], {"A": 1, "B": 6})
        for _, reason, _ in cases:
            self.assertEqual(result["admission"]["rejections_by_reason"][reason], 1)
        self.assertEqual(result["phases_ms"]["audio_wait_ms"]["max_ms"], 5)
        self.assertEqual(result["recent"][2]["admission"]["snapshot"], capacity)
        self.assertEqual(result["failed"], 6)

    def test_copies_metadata_and_filters_unbounded_keys(self):
        admission = {"category": "B", "outcome": "admitted", "reason": "arbitrary-request-id",
                     "snapshot": AudioSlotManager(8).snapshot(), "transcript": "private"}
        event = self.record(admission=admission)
        admission["snapshot"]["audioConcurrency"]["active_total"] = 999
        event["admission"]["snapshot"]["audioConcurrency"]["active_total"] = 1000
        result = telemetry.snapshot()
        saved = result["recent"][0]["admission"]
        self.assertEqual(saved["snapshot"]["audioConcurrency"]["active_total"], 0)
        self.assertIsNone(saved["reason"])
        self.assertNotIn("transcript", saved)
        self.assertNotIn("arbitrary-request-id", result["admission"]["rejections_by_reason"])
        saved["category"] = "A"
        self.assertEqual(telemetry.snapshot()["recent"][0]["admission"]["category"], "B")

    def test_retained_window_eviction_and_legacy_events(self):
        self.record(code=4017, status="failed", admission={"category": "B", "outcome": "rejected", "reason": "spk_emo_full"})
        for _ in range(telemetry._MAX_EVENTS):
            self.record()
        result = telemetry.snapshot()
        self.assertEqual(result["events"], 4096)
        self.assertEqual(len(result["recent"]), 20)
        self.assertEqual(result["admission"]["by_class"], {"A": 0, "B": 0})
        self.assertEqual(result["admission"]["rejections_by_reason"]["spk_emo_full"], 0)
        self.assertNotIn("recent", telemetry.snapshot(include_events=False))
