from scripts.benchmark_asr_throughput import summarize


def test_warning_completion_is_not_strict_success():
    rows = [dict(status="completed", elapsed_s=i+1, audio_seconds=10,
                 processing_status="complete", warning_count=0, has_usable_text=True) for i in range(6)]
    rows += [dict(status="completed", elapsed_s=7, audio_seconds=10,
                  processing_status="completed_with_warnings", warning_count=2,
                  unrecoverable_segment_count=1, has_usable_text=True),
             dict(status="completed", elapsed_s=8, audio_seconds=10,
                  processing_status="completed_with_warnings", warning_count=3,
                  unrecoverable_segment_count=3, has_usable_text=False)]
    result = summarize(rows, 0, 8, False)
    assert result["complete"] is True
    assert result["processed_completed"] == 8 and result["successful"] == 6
    assert result["completion_subset_latency"]["n"] == 8
    assert result["success_subset_latency"]["n"] == 6
    assert result["warning_segments"] == 5 and result["unrecoverable_segments"] == 4
    assert result["requests_with_usable_text"] == 7
    assert result["status"] == "completed_with_warnings"


def test_failed_and_censored_requests_are_not_completions():
    result = summarize([dict(status="failed", elapsed_s=10, audio_seconds=10),
                        dict(status="cancelled", elapsed_s=420, audio_seconds=10)], 0, 420, True)
    assert result["processed_completed"] == result["successful"] == 0
    assert result["completion_subset_latency"]["p95_s"] is None
    assert result["success_subset_latency"]["p50_s"] is None
    assert not result["complete"]
