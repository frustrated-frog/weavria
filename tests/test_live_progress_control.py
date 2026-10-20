from examples.live_progress_control import (
    ModelCallSample,
    aggregate_usage,
    finish_reason_from_payload,
    usage_from_payload,
)
from weavria.evaluation import TokenUsage


def test_live_usage_accepts_common_provider_names_without_retaining_payload() -> None:
    standard = usage_from_payload(
        {
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "choices": "malformed-but-independent",
        }
    )
    alternate = usage_from_payload({"usage": {"input_tokens": 4, "output_tokens": 2}})

    assert standard == TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    assert alternate == TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6)


def test_live_usage_keeps_partial_or_malformed_accounting_unknown() -> None:
    partial = aggregate_usage(
        (
            ModelCallSample(1.0, TokenUsage(prompt_tokens=2, total_tokens=3), "stop"),
            ModelCallSample(2.0, TokenUsage(), "missing"),
        )
    )

    assert partial == TokenUsage()
    assert usage_from_payload({"usage": {"prompt_tokens": -1}}) == TokenUsage()


def test_live_finish_reason_is_bounded_without_retaining_provider_payload() -> None:
    assert (
        finish_reason_from_payload(
            {"choices": [{"finish_reason": "length"}], "usage": {"prompt_tokens": -1}}
        )
        == "length"
    )
    assert finish_reason_from_payload({"choices": [{"finish_reason": "vendor_reason"}]}) == "other"
    assert finish_reason_from_payload({"choices": []}) == "missing"
    assert finish_reason_from_payload({"choices": "not-a-list"}) == "invalid_payload"
