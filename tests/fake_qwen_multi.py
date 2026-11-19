"""Importable CPU-only model-worker double for multi-process ownership tests."""
import io
import json
import os
import struct
import time
import wave
from pathlib import Path

from tests.test_local_qwen_process import CpuRuntime as BaseCpuRuntime


class CpuRuntime(BaseCpuRuntime):
    def event(self, event, **values):
        row = {
            "event": event, "pid": os.getpid(), "call": self.calls,
            "instance_index": self.config["instance_index"],
            "monotonic_s": time.monotonic(), **values,
        }
        with Path(self.config["events"]).open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def ensure_loaded(self):
        cold = not self.loaded
        if cold:
            self.event("load_started")
        super().ensure_loaded()
        if cold:
            self.event("load_done")

    def transcribe_batch(self, audios, languages=None):
        from core.qwen_asr_runtime import LocalUnsupportedLanguage

        self.ensure_loaded()
        self.calls += 1
        codes = []
        for audio in audios:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                codes.append(struct.unpack("<h", wav.readframes(1))[0])
        self.event("started", members=codes)
        if self.config.get("die_on_transcribe"):
            os._exit(17)
        gate = self.config.get("gate")
        while gate and not Path(gate).exists():
            time.sleep(.005)
        time.sleep(self.config.get("delay_s", 0))
        languages = languages or [None] * len(audios)
        valid = [index for index, language in enumerate(languages) if language != "bad"]
        self.last_batch_timings = {
            "batch_size": len(valid), "requested_batch_size": len(audios),
            "valid_member_indices": valid, "generate_started": True,
            "generate_completed": True, "generate_wall_ms": 1,
        }
        self.event("read_complete", members=codes)
        return [
            LocalUnsupportedLanguage("unsupported language") if language == "bad"
            else "language Chinese<asr_text>fake-" + str(code)
            for code, language in zip(codes, languages)
        ]

    def close(self):
        self.event("close_started")
        time.sleep(self.config.get("close_delay_s", 0))
        super().close()
        self.event("close_done")
