from examples.live_progress_control import ModelCallSample, aggregate_usage, usage_from_payload
from weavria.evaluation import TokenUsage


def test_live_usage_accepts_common_provider_names_without_retaining_payload() -> None:
    standard = usage_from_payload(
        {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
    )
    alternate = usage_from_payload({"usage": {"input_tokens": 4, "output_tokens": 2}})

    assert standard == TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    assert alternate == TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6)


def test_live_usage_keeps_partial_or_malformed_accounting_unknown() -> None:
    partial = aggregate_usage(
        (
            ModelCallSample(1.0, TokenUsage(prompt_tokens=2, total_tokens=3)),
            ModelCallSample(2.0, TokenUsage()),
        )
    )

    assert partial == TokenUsage()
    assert usage_from_payload({"usage": {"prompt_tokens": -1}}) == TokenUsage()
