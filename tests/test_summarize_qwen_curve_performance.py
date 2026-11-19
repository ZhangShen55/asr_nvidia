import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("summarize_qwen_curve_performance",Path(__file__).resolve().parents[1]/"scripts/summarize_qwen_curve_performance.py")
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_pool_counters_subtract_warmup_and_keep_units():
    first={"speech_campplus":{"calls":128,"wait_total_ms":100,"inference_total_ms":1000,"handoff_count":120,"handoff_total_ms":2}}
    final={"speech_campplus":{"calls":256,"wait_total_ms":300,"inference_total_ms":3000,"handoff_count":240,"handoff_total_ms":5}}
    row=tool.pool_delta(final,first)["speech_campplus"]
    assert row["round_delta"]["calls"]==128 and row["round_delta"]["wait_total_ms"]==200
    assert row["mean_handoff_ms"]==3/120
    assert row["mean_inference_envelope_ms"]==2000/128


def test_physical_batch_and_parallel_stage_not_multiplied():
    data={"batches":[{"batch_id":1,"size":128,"runtime_timings":{"batch_size":128,"generate_wall_ms":8000,"synchronize_wall_ms":2000}}],
          "members":[{"start_monotonic_s":10,"ended_monotonic_s":20}],
          "enhanced":{"members":[{"speech_campplus":{"start_monotonic_s":10,"ended_monotonic_s":25,"status":"completed"}}]}}
    row=tool.one_round(data,None)
    assert row["actual_physical_batch_distribution"]=={"128":1}
    assert row["qwen_phase_sum_s_once_per_physical_batch"]["generate_wall_ms"]==8
    assert row["parallel_stage_caller_envelopes"]["qwen_callers_s"]==10
    assert row["parallel_stage_caller_envelopes"]["speech_campplus_callers_s"]==15


def test_missing_previous_is_unknown_not_cumulative_round():
    row=tool.pool_delta({"speech_campplus":{"calls":256,"wait_total_ms":99999}},None)["speech_campplus"]
    assert row["round_delta"]=={} and row["mean_pool_wait_ms"] is None
