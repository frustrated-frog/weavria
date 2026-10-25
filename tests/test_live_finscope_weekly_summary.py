from examples.live_finscope_weekly_summary import (
    ModelCallSample,
    finish_reason_from_payload,
    summarize_samples,
    usage_from_payload,
)
from weavria.evaluation import TokenUsage


def test_live_finscope_metrics_do_not_invent_missing_usage() -> None:
    samples = (
        ModelCallSample(
            latency_ms=2.5,
            usage=TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            finish_reason="stop",
            http_status=200,
        ),
        ModelCallSample(
            latency_ms=1.5,
            usage=TokenUsage(),
            finish_reason="missing",
            http_status=200,
        ),
    )

    summary = summarize_samples(samples)

    assert summary["calls"] == 2
    assert summary["latency_ms"] == 4.0
    assert summary["usage_coverage"] == 0.5
    assert summary["total_tokens"] is None
    assert summary["http_statuses"] == [200, 200]


def test_live_finscope_metrics_accept_common_provider_fields_without_content() -> None:
    assert usage_from_payload(
        {"usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}}
    ) == TokenUsage(prompt_tokens=9, completion_tokens=4, total_tokens=13)
    assert usage_from_payload({"usage": {"input_tokens": 5, "output_tokens": 2}}) == TokenUsage(
        prompt_tokens=5, completion_tokens=2, total_tokens=7
    )
    assert finish_reason_from_payload({"choices": [{"finish_reason": "stop"}]}) == "stop"
    assert finish_reason_from_payload({"choices": []}) == "missing"
