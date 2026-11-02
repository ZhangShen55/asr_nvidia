"""按固定课堂裁剪验证真实服务运行器、预处理、输出与多实例资源。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from funasr import AutoModel
    from funasr.models.campplus.utils import extract_feature
    from core.local_inference import TorchRunner, campplus_features, waveform_tensor
    from core.tensorrt_runner import TensorRTRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    root = Path(__file__).resolve().parents[1]
    samples = json.loads(args.samples.read_text())
    results = []
    for name, directory in [("speech_campplus", "speech_campplus_sv_zh_en_16k-common_advanced"),
                            ("emotion2vec", "emotion2vec_plus_large")]:
        model_dir = root / "model" / directory
        reference = TorchRunner(AutoModel(model=str(model_dir), device=args.device, fp16=True,
                                         disable_update=True, disable_pbar=True), args.device, fp16=True)
        eligible = [row for row in samples if name == "speech_campplus" or 1 <= row["duration_s"] <= 10]
        baselines = {}
        for row in eligible:
            audio = np.load(row["file"])
            if name == "speech_campplus":
                actual = campplus_features(waveform_tensor(audio, 16000))
                expected, _, _ = extract_feature([torch.from_numpy(audio)])
                assert torch.equal(actual, expected), "CAM++ fbank differs from FunASR"
            baselines[row["id"]] = reference.generate(input=audio, fs=16000,
                granularity="utterance", extract_embedding=False)[0]
        reference.close()
        for precision in ("fp32", "fp16"):
            runner = TensorRTRunner(name, model_dir, args.device, precision)
            rows = []
            for row in eligible:
                audio = np.load(row["file"])
                timings = []
                for _ in range(3):
                    start = time.perf_counter()
                    result = runner.generate(input=audio, fs=16000, granularity="utterance", extract_embedding=False)[0]
                    timings.append((time.perf_counter() - start) * 1000)
                baseline = baselines[row["id"]]
                if name == "speech_campplus":
                    x = result["spk_embedding"].numpy().ravel()
                    y = baseline["spk_embedding"].numpy().ravel()
                    metrics = {"cosine_vs_service_pytorch_fp16": float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))),
                               "max_abs": float(np.abs(x - y).max()), "dimension": len(x)}
                else:
                    x, y = np.array(result["scores"]), np.array(baseline["scores"])
                    assert result["labels"] == baseline["labels"]
                    metrics = {"score_max_abs": float(np.abs(x - y).max()),
                               "label_matches": bool(x.argmax() == y.argmax()),
                               "pytorch_margin": float(np.sort(y)[-1] - np.sort(y)[-2]),
                               "pytorch_label": baseline["labels"][y.argmax()],
                               "trt_label": result["labels"][x.argmax()]}
                rows.append({"id": row["id"], "duration_s": row["duration_s"],
                             "latencies_ms": timings, **metrics})
            contract = runner.contract
            for shape in (contract["profile"][0], contract["profile"][1], contract["profile"][2]):
                result = runner._execute(torch.zeros(shape, dtype=torch.float32))
                assert all(bool(torch.isfinite(value).all()) for value in result.values())
            for shape in ([1, contract["profile"][0][1] - 1, *contract["profile"][0][2:]],
                          [1, contract["profile"][2][1] + 1, *contract["profile"][2][2:]]):
                try:
                    runner._execute(torch.zeros(shape, dtype=torch.float32))
                except ValueError:
                    pass
                else:
                    raise AssertionError("out-of-profile input accepted")
            runner.close()
            results.append({"model": name, "precision": precision, "rows": rows,
                            "profile_boundaries_pass": True, "out_of_profile_rejected": True,
                            "npu_hardware_verified": False})
            print(name, precision, "validated", len(rows), flush=True)
        if name == "speech_campplus":
            runners = [TensorRTRunner(name, model_dir, args.device, "fp32") for _ in range(3)]
            assert len({id(r.context) for r in runners}) == len({r.stream.cuda_stream for r in runners}) == 3
            inputs = [np.load(eligible[index]["file"]) for index in (0, 4, 8)]
            sequential = [r.generate(input=x, fs=16000)[0]["spk_embedding"].numpy() for r, x in zip(runners, inputs)]
            with ThreadPoolExecutor(max_workers=3) as executor:
                parallel = list(executor.map(lambda pair: pair[0].generate(input=pair[1], fs=16000), zip(runners, inputs)))
            for expected, actual in zip(sequential, parallel):
                assert np.array_equal(expected, actual[0]["spk_embedding"].numpy())
            assert len({r.buffers["features"].data_ptr() for r in runners}) == 3
            for runner in runners:
                runner.close()
            results[-1]["three_contexts_streams_buffers_independent"] = True
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
