import importlib.util
import importlib.metadata
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("check_npu_environment",ROOT/"scripts/check_npu_environment.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_metadata_check_does_not_import_device_backend():
    packages,errors=module.inspect_packages(lambda name:module.REQUIRED[name])
    assert errors == [] and packages["torch-npu"] == "2.5.1"


def test_missing_attrs_has_actionable_diagnostic():
    def version(name):
        if name == "attrs":raise importlib.metadata.PackageNotFoundError(name)
        return module.REQUIRED[name]
    _,errors=module.inspect_packages(version)
    assert len(errors)==1 and "attrs" in errors[0] and "未安装" in errors[0]


def test_version_mismatch_and_local_wheel_suffix():
    _,errors=module.inspect_packages(lambda name:"2.6.0" if name=="torch" else module.REQUIRED[name])
    assert len(errors)==1 and "torch" in errors[0]
    _,errors=module.inspect_packages(lambda name:module.REQUIRED[name]+"+cpu" if name=="torch" else module.REQUIRED[name])
    assert errors==[]


def test_rebuild_refuses_existing_prefix_without_installing(tmp_path):
    marker=tmp_path/"do-not-touch";marker.write_text("original")
    result=subprocess.run(["bash",str(ROOT/"scripts/rebuild_npu_env.sh"),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==2 and "拒绝覆盖" in result.stderr
    assert marker.read_text()=="original"


def test_native_cpu_reference_uses_original_checkpoint_boundary(monkeypatch):
    """The latency baseline must never delegate to a removed engine implementation."""
    import numpy as np
    import torch
    from core import local_inference
    from scripts.bench_local_model_latency import build_runner

    loaded = []
    class Network(torch.nn.Module):
        def forward(self, features):
            return features.mean(dim=1)
    def load(name, directory):
        loaded.append((name, str(directory)))
        return Network()
    monkeypatch.setattr(local_inference, "load_native_network", load)
    monkeypatch.setattr(local_inference, "campplus_features", lambda wave: wave.view(1, -1, 1))
    runner = build_runner("speech_campplus", "/original/weights", "cpu")
    result = runner.generate(input=np.ones(320, dtype=np.float32), fs=16000)
    assert loaded == [("speech_campplus", "/original/weights")]
    assert result[0]["spk_embedding"].shape == (1, 1)
    assert result[0]["spk_embedding"].item() == 1
    assert runner.actual_precision == "fp32"
    runner.close()
    assert runner.model is None


def test_native_cpu_emotion_reference_retains_seven_class_order(monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from core import local_inference
    from scripts.bench_local_model_latency import build_runner

    class Network(torch.nn.Module):
        cfg = SimpleNamespace(normalize=True)
        def extract_features(self, source, padding_mask=None):
            assert torch.allclose(source.mean(), torch.tensor(0.), atol=1e-5)
            return {"x": torch.zeros(1, 2, 9)}
        def proj(self, features):
            return torch.arange(9, dtype=torch.float32).view(1, 9)
    monkeypatch.setattr(local_inference, "load_native_network", lambda *args: Network())
    monkeypatch.setattr(local_inference, "emotion_labels", lambda *args: list("abcdefg"))
    runner = build_runner("emotion2vec", "/original/weights", "cpu")
    result = runner.generate(input=np.arange(320, dtype=np.float32), fs=16000)[0]
    expected = torch.softmax(torch.tensor([0., 2., 3., 4., 5., 7., 8.]), dim=0)
    assert result["labels"] == list("abcdefg")
    assert torch.allclose(torch.tensor(result["scores"]), expected)
    runner.close()


def test_cann_compiler_path_is_checked_without_device_initialization():
    from types import SimpleNamespace
    checked = module.inspect_compiler_modules(lambda name: SimpleNamespace(__file__="/cann/" + name))
    assert checked == {"tbe": "/cann/tbe", "te": "/cann/te", "attr": "/cann/attr"}


def test_cann_missing_compiler_module_fails_before_runtime_probe():
    import pytest
    def missing(name):
        raise ModuleNotFoundError(name)
    with pytest.raises(ModuleNotFoundError, match="tbe"):
        module.inspect_compiler_modules(missing)


def test_benchmark_slice_uses_pyav_without_ffmpeg_cli(tmp_path, monkeypatch):
    import os
    import wave
    import numpy as np
    from scripts.bench_local_model_latency import read_audio
    from scripts.bench_vad_cpu_vs_gpu import make_slice
    source = tmp_path / "audio with space.wav"
    pcm = np.arange(16000, dtype="<i2")
    with wave.open(str(source), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", ""))
        handle.writeframes(pcm.tobytes())
    monkeypatch.setenv("PATH", "")
    audio, rate = read_audio(source, max_seconds=.1)
    assert rate == 16000 and audio.shape == (1600,)
    assert np.allclose(audio, pcm[:1600].astype(np.float32) / 32768)
    sliced = make_slice(source, .1)
    try:
        with wave.open(sliced, "rb") as handle:
            assert handle.getparams()[:4] == (1, 2, 16000, 1600)
    finally:
        os.unlink(sliced)


def test_vad_benchmark_failure_aborts_peer_barrier_and_closes_owned_model(monkeypatch):
    import pytest
    from scripts import bench_vad_process
    events = []
    class Barrier:
        def abort(self): events.append("abort")
    class Model:
        def close(self): events.append("close")
    monkeypatch.setattr(bench_vad_process, "build_model", lambda device: Model())
    def fail(*args):
        raise RuntimeError("warmup failed")
    monkeypatch.setattr(bench_vad_process, "run_vad", fail)
    with pytest.raises(RuntimeError, match="warmup failed"):
        bench_vad_process._worker(("cpu", "unused.wav", Barrier()))
    assert events == ["abort", "close"]


def test_benchmark_pyav_reads_raw_gbk_path_without_renaming(tmp_path):
    import os
    import sys
    import wave
    import numpy as np
    import pytest
    from scripts.bench_local_model_latency import read_audio
    if os.name != "posix" or sys.platform == "darwin":
        pytest.skip("raw non-UTF8 filename fixture requires a Linux-style filesystem")
    raw = os.fsencode(tmp_path) + b"/\xce\xd2\xb5\xc4 \xd2\xf4\xc6\xb5.wav"
    with open(raw, "wb") as output:
        with wave.open(output, "wb") as handle:
            handle.setparams((1, 2, 16000, 0, "NONE", ""))
            handle.writeframes(np.zeros(160, dtype="<i2").tobytes())
    audio, rate = read_audio(os.fsdecode(raw))
    assert audio.shape == (160,) and rate == 16000
    assert os.path.exists(raw)
