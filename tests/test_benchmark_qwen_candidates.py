"""CPU-only evidence for candidate batching, reporting and ownership; no NPU."""
import base64
from contextlib import nullcontext
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest
import torch

from scripts.benchmark_qwen_candidates import (
    CandidateRunner, Sample, length_batches, load_samples, parsed_output,
    run_benchmark, validate_device, validate_masks,
)


@pytest.fixture
def local_model_metadata(tmp_path):
    """CLI reporting tests use small owned metadata, never deployment model assets."""
    directory = tmp_path / "cpu-model-metadata"
    directory.mkdir()
    (directory / "config.json").write_text('{"cpu_fixture": true}')
    return directory


def wav(frames, value=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(value.to_bytes(2, "little", signed=True) * frames)
    return buffer.getvalue()


def sample(identifier, seconds, language=None):
    frames = round(seconds * 16000)
    return Sample(identifier, "unused.wav", wav(frames), frames, language, "source-hash", 0, frames)


def test_readonly_manifest_clips_preserve_pcm_and_language(tmp_path):
    source = tmp_path / "source.wav"
    original = wav(16000, -127)
    source.write_bytes(original)
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps({"samples": [
        {"id": "one", "path": source.name, "start_s": .1, "end_s": .4, "language": "auto"},
        {"id": "two", "path_bytes_b64": base64.b64encode(bytes(source)).decode(), "language": "en"},
    ]}))
    clips = load_samples(manifest)
    assert source.read_bytes() == original
    assert clips[0].frames == 4800
    assert clips[0].start_frame == 1600 and clips[0].end_frame == 6400
    with wave.open(io.BytesIO(clips[0].wav_bytes)) as reader:
        assert reader.readframes(4800) == (-127).to_bytes(2, "little", signed=True) * 4800
    assert [clip.language for clip in clips] == [None, "en"]
    assert clips[0].source_sha256 == clips[1].source_sha256
    assert len(clips[0].metadata()["clip_wav_sha256"]) == 64
    assert sorted(path.name for path in tmp_path.iterdir()) == ["clips.json", "source.wav"]


@pytest.mark.parametrize("rows", [
    [{"id": "bad", "path": "source.wav", "end_s": 3}],
    [{"id": "bad", "path": "source.wav", "start_s": float("nan")}],
    [{"id": "x", "path": "source.wav"}, {"id": "x", "path": "source.wav"}],
])
def test_manifest_rejects_invalid_bounds_and_duplicate_ids(tmp_path, rows):
    (tmp_path / "source.wav").write_bytes(wav(16000))
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        load_samples(manifest)


def test_length_buckets_preserve_ids_tail_and_original_indices():
    clips = [sample("long", 5), sample("short", .3), sample("mid", 1.5), sample("tail", .7)]
    batches = length_batches(clips, 2, 2)
    assert [[index for index, _ in batch] for batch in batches] == [[1, 3], [2], [0]]
    assert sorted(index for batch in batches for index, _ in batch) == list(range(4))
    assert [[clip.id for _, clip in batch] for batch in batches] == [["short", "tail"], ["mid"], ["long"]]


def inputs():
    return {"input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            "input_features": torch.zeros(2, 8, 4),
            "input_features_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])}


def test_native_batch_keeps_independent_lengths_languages_masks_and_output_order():
    seen = {}
    expected = inputs()
    class Processor:
        def apply_transcription_request(self, audio, language):
            seen["lengths"] = [len(item) for item in audio]
            seen["language"] = language
            return expected
        def batch_decode(self, generated, **kwargs):
            seen["generated"] = generated.tolist()
            assert kwargs["skip_special_tokens"] is False
            return ["language Chinese<asr_text>一<|im_end|>", "language English<asr_text>two<|im_end|>"]
    class Model:
        dtype = torch.float32
        def generate(self, **kwargs):
            seen["token_dtype"] = kwargs["input_ids"].dtype
            seen["audio_mask"] = kwargs["input_features_mask"].tolist()
            return torch.cat((kwargs["input_ids"], torch.tensor([[6], [7]])), dim=1)
    api = SimpleNamespace(set_device=lambda dev: seen.setdefault("device", dev),
                          device=lambda dev: nullcontext(),
                          synchronize=lambda dev: seen.setdefault("sync", dev))
    runner = CandidateRunner.__new__(CandidateRunner)
    runner.runtime = SimpleNamespace(device="cpu", dtype="float32", processor=Processor(),
                                     model=Model(), api=api, max_new_tokens=10, failure=None)
    raw, masks = runner.infer([sample("first", .2, "zh"), sample("second", .4, "en")], True)
    assert seen["lengths"] == [3200, 6400]
    assert seen["language"] == ["zh", "en"]
    assert seen["generated"] == [[6], [7]]
    assert seen["token_dtype"] == torch.int64
    assert seen["audio_mask"] == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert masks["input_features_mask"]["valid_lengths"] == [2, 4]
    assert seen["sync"] == "cpu"
    assert raw[0].endswith("<|im_end|>")
    assert parsed_output(raw[0]) == {"language": "Chinese", "text": "一", "has_speech": True}


@pytest.mark.parametrize("mutation", [
    lambda data: data.pop("input_features_mask"),
    lambda data: data.update(attention_mask=torch.ones(1, 3)),
    lambda data: data.update(input_features_mask=torch.zeros(2, 4)),
    lambda data: data.update(input_features_mask=torch.full((2, 4), .5)),
])
def test_batch_rejects_missing_or_invalid_masks(mutation):
    data = inputs()
    mutation(data)
    with pytest.raises(ValueError):
        validate_masks(data, 2)


class FakeRunner:
    def __init__(self, fail_load=False, fail_infer=False, interrupt=False, fail_close=False):
        self.closed = False
        self.calls = []
        self.fail_load, self.fail_infer = fail_load, fail_infer
        self.interrupt, self.fail_close = interrupt, fail_close
    def load(self):
        if self.fail_load:
            raise RuntimeError("load failed")
    def metadata(self):
        return {"fake": True}
    def infer(self, samples, native_batch):
        self.calls.append(([item.id for item in samples], native_batch))
        if self.interrupt:
            raise KeyboardInterrupt()
        if self.fail_infer:
            raise ValueError("generation failed")
        return [f"language English<asr_text>{item.id}" for item in samples], {"retained": True}
    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")


def execute(runner, report=None, **options):
    clock_values = iter(range(1000))
    return run_benchmark([sample("late", 3), sample("early", .2), sample("middle", .4)],
                         candidate=options.pop("candidate", "batch"), batch_size=2,
                         bucket_seconds=2, warmup=1, repeats=2, runner=runner,
                         report={} if report is None else report, clock=lambda: next(clock_values), **options)


def test_report_separates_warmup_and_steady_preserves_each_identity():
    runner = FakeRunner()
    report = execute(runner)
    assert runner.closed and report["close"]["status"] == "ok"
    assert len(report["records"]) == 6
    assert report["records"][0]["original_indices"] == [1, 2]
    first_outputs = report["records"][0]["outputs"]
    assert [(row["original_index"], row["sample_id"], row["parsed"]["text"]) for row in first_outputs] == [(1, "early", "early"), (2, "middle", "middle")]
    assert report["summary"]["successful_batches"] == 4
    assert report["summary"]["steady_total_inference_ms"] == 4000
    assert report["summary"]["steady_audio_seconds"] == pytest.approx(7.2)
    assert report["summary"]["steady_rtf"] == pytest.approx(4 / 7.2)
    assert all(call[1] for call in runner.calls)


@pytest.mark.parametrize("candidate,expected", [("eager", 9), ("batch", 6)])
def test_failures_remain_attributed_and_never_become_success(candidate, expected):
    runner = FakeRunner(fail_infer=True)
    report = execute(runner, candidate=candidate)
    assert runner.closed
    assert report["summary"]["failed_batches"] == expected
    assert report["summary"]["successful_batches"] == 0
    assert report["summary"]["steady_rtf"] is None
    assert all(record["sample_ids"] and record["error_type"] == "ValueError" for record in report["records"])
    assert all("outputs" not in record for record in report["records"])


@pytest.mark.parametrize("options,error", [({"fail_load": True}, RuntimeError),
                                            ({"interrupt": True}, KeyboardInterrupt),
                                            ({"fail_close": True}, RuntimeError)])
def test_load_interrupt_and_close_failures_still_close_owned_model(options, error):
    runner, report = FakeRunner(**options), {}
    with pytest.raises(error):
        execute(runner, report)
    assert runner.closed
    assert report["close"]["status"] == ("failed" if options.get("fail_close") else "ok")
    if options.get("fail_load"):
        assert report["load"]["status"] == "failed" and not runner.calls


def test_stop_request_does_not_start_new_work_and_closes():
    runner, report = FakeRunner(), {}
    with pytest.raises(InterruptedError):
        execute(runner, report, should_stop=lambda: True)
    assert runner.closed and not runner.calls


def test_host_card_lock_rejects_other_devices_and_remapped_environment(monkeypatch):
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    validate_device("npu:7")
    for device in ("npu:0", "cuda:7", "cpu"):
        with pytest.raises(ValueError):
            validate_device(device)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7")
    with pytest.raises(ValueError, match="mapping"):
        validate_device("npu:7")


def test_malformed_raw_output_keeps_evidence_and_reports_protocol_failure():
    runner = FakeRunner()
    runner.infer = lambda samples, native_batch: (["plain unprefixed transcript"] * len(samples), {})
    report = execute(runner)
    assert runner.closed
    assert report["summary"]["successful_batches"] == 0
    assert all(record["status"] == "failed" for record in report["records"])
    assert report["records"][0]["outputs"][0]["raw"] == "plain unprefixed transcript"
    assert report["records"][0]["outputs"][0]["raw_contract_ok"] is False


def test_cli_repeated_stop_signals_preserve_report_and_close_before_exit(tmp_path, monkeypatch, local_model_metadata):
    import signal
    import scripts.benchmark_qwen_candidates as benchmark
    source = tmp_path / "source.wav"
    source.write_bytes(wav(16000))
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps([{"id": "first", "path": str(source)},
                                    {"id": "second", "path": str(source)}]))
    output = tmp_path / "run.json"
    runner = FakeRunner()
    infer = runner.infer
    def interrupt_after_owned_work(samples, native_batch):
        result = infer(samples, native_batch)
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)
        return result
    runner.infer = interrupt_after_owned_work
    monkeypatch.setattr(benchmark, "CandidateRunner", lambda *args: runner)
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    previous = signal.getsignal(signal.SIGINT)
    status = benchmark.main(["--model-dir", str(local_model_metadata), "--manifest", str(manifest), "--output", str(output),
                             "--candidate", "eager", "--warmup", "0", "--repeats", "1"])
    report = json.loads(output.read_text())
    assert status == 128 + signal.SIGINT
    assert report["stop_signals"] == [signal.SIGINT, signal.SIGINT]
    assert report["close"]["status"] == "ok" and runner.closed
    assert len(runner.calls) == len(report["records"]) == 1
    assert report["records"][0]["outputs"][0]["sample_id"] == "first"
    assert signal.getsignal(signal.SIGINT) == previous
    assert not list(tmp_path.glob("*.partial"))


def test_lazy_import_failure_preserves_original_cause():
    from scripts.benchmark_qwen_candidates import error_details
    try:
        try:
            raise ModuleNotFoundError("No module named pkg_resources")
        except ModuleNotFoundError as cause:
            raise RuntimeError("lazy import wrapper") from cause
    except RuntimeError as exc:
        details = error_details(exc)
    assert details["error_type"] == "RuntimeError"
    assert [row["error_type"] for row in details["error_chain"]] == ["RuntimeError", "ModuleNotFoundError"]
    assert details["error_chain"][1]["error"] == "No module named pkg_resources"


def attention_model(propagates=True):
    from scripts.benchmark_qwen_candidates import attention_configurations
    audio = SimpleNamespace(_attn_implementation="sdpa", sub_configs={})
    text = SimpleNamespace(_attn_implementation="sdpa", sub_configs={})
    config = SimpleNamespace(_attn_implementation="sdpa", sub_configs={"audio_config": object, "text_config": object},
                             audio_config=audio, text_config=text)
    model = SimpleNamespace(config=config)
    model.named_modules = lambda: [("", model), ("audio_tower", SimpleNamespace(config=audio)),
                                    ("language_model", SimpleNamespace(config=text))]
    model.setter_calls = []
    def setter(mode):
        model.setter_calls.append(mode)
        config._attn_implementation = mode
        if propagates:
            audio._attn_implementation = text._attn_implementation = mode
    model.set_attn_implementation = setter
    return model


def test_candidate_attention_default_is_unchanged_and_eager_propagation_is_observed():
    from scripts.benchmark_qwen_candidates import set_candidate_attention
    model = attention_model()
    unchanged = set_candidate_attention(model, "default")
    assert not model.setter_calls
    assert [row["implementation"] for row in unchanged["after"]] == ["sdpa", "sdpa", "sdpa"]
    changed = set_candidate_attention(model, "eager")
    assert model.setter_calls == ["eager"]
    assert [row["implementation"] for row in changed["after"]] == ["eager", "eager", "eager"]
    assert changed["after"][1]["config_paths"] == ["config.audio_config"]
    assert changed["after"][1]["module_paths"] == ["audio_tower"]
    assert changed["after"][2]["module_paths"] == ["language_model"]


def test_candidate_attention_refuses_silent_partial_propagation():
    from scripts.benchmark_qwen_candidates import set_candidate_attention
    with pytest.raises(RuntimeError, match="did not propagate"):
        set_candidate_attention(attention_model(propagates=False), "eager")


def test_attention_switch_happens_after_load_before_compile(monkeypatch):
    import sys
    import core.qwen_asr_runtime as runtime_module
    events = []
    model = attention_model()
    model.forward = lambda: None
    setter = model.set_attn_implementation
    model.set_attn_implementation = lambda mode: (events.append(("attention", mode)), setter(mode))[-1]
    runtime = SimpleNamespace(model=model, ensure_loaded=lambda: events.append(("loaded",)))
    monkeypatch.setattr(runtime_module, "LocalQwenRuntime", lambda *args: runtime)
    monkeypatch.setitem(sys.modules, "torchair", SimpleNamespace(CompilerConfig=lambda: {},
                         get_npu_backend=lambda **kwargs: "cpu-fake-backend"))
    def compile_forward(forward, **kwargs):
        events.append(("compile", kwargs))
        assert all(row["implementation"] == "eager" for row in runner.attention_state["after"])
        return forward
    monkeypatch.setattr(torch, "compile", compile_forward)
    runner = CandidateRunner("unused", "npu:7", "bfloat16", 256, "torchair", "eager")
    runner.load()
    assert events[0:2] == [("loaded",), ("attention", "eager")]
    assert events[2] == ("compile", {"backend": "cpu-fake-backend", "dynamic": True})


@pytest.mark.parametrize("mode", ["default", "eager"])
def test_cli_routes_optional_attention_choice_only_to_candidate_runner(tmp_path, monkeypatch, mode, local_model_metadata):
    import scripts.benchmark_qwen_candidates as benchmark
    source = tmp_path / "source.wav"
    source.write_bytes(wav(16000))
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps([{"id": "test", "path": str(source)}]))
    output = tmp_path / "run.json"
    seen = []
    monkeypatch.setattr(benchmark, "CandidateRunner", lambda *args: (seen.append(args), FakeRunner())[1])
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    args = ["--model-dir", str(local_model_metadata), "--manifest", str(manifest), "--output", str(output), "--warmup", "0", "--repeats", "1"]
    if mode == "eager":
        args += ["--attention", "eager"]
    assert benchmark.main(args) == 0
    assert seen[0][-1] == mode
    assert json.loads(output.read_text())["configuration"]["attention"] == mode
