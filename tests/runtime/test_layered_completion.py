from typing import cast

from weavria.core import (
    CompletionAccept,
    CompletionDecision,
    CompletionRepair,
    CompletionUnverifiable,
)
from weavria.runtime import CompletionContext, LayeredCompletionPolicy


class _Policy:
    def __init__(self, version: str, decision: CompletionDecision) -> None:
        self.version = version
        self.decision = decision
        self.calls = 0

    async def verify(self, context: CompletionContext) -> CompletionDecision:
        self.calls += 1
        return self.decision


async def test_required_completion_policy_cannot_be_bypassed_by_extension() -> None:
    required = _Policy(
        "required/1",
        CompletionRepair(issue_codes=("required_output_mismatch",), guidance="修正输出"),
    )
    extension = _Policy("extension/1", CompletionAccept(outcome="verified"))
    policy = LayeredCompletionPolicy(required, extension)

    decision = await policy.verify(cast(CompletionContext, object()))

    assert isinstance(decision, CompletionRepair)
    assert extension.calls == 0


async def test_layered_completion_acceptance_keeps_the_weaker_outcome() -> None:
    required = _Policy("required/1", CompletionAccept(outcome="verified"))
    extension = _Policy("extension/1", CompletionAccept(outcome="unverified"))

    decision = await LayeredCompletionPolicy(required, extension).verify(
        cast(CompletionContext, object())
    )

    assert decision == CompletionAccept(outcome="unverified")
    assert required.calls == 1
    assert extension.calls == 1


async def test_unverifiable_required_check_cannot_skip_extension_publication_repair() -> None:
    required = _Policy(
        "required/1",
        CompletionUnverifiable(reason="No independent factual verifier"),
    )
    extension = _Policy(
        "extension/1",
        CompletionRepair(issue_codes=("redaction_required",), guidance="移除敏感内容"),
    )

    decision = await LayeredCompletionPolicy(required, extension).verify(
        cast(CompletionContext, object())
    )

    assert isinstance(decision, CompletionRepair)
    assert decision.issue_codes == ("redaction_required",)
    assert required.calls == 1
    assert extension.calls == 1


async def test_unverifiable_required_check_retains_reason_after_extension_accepts() -> None:
    required = _Policy(
        "required/1",
        CompletionUnverifiable(reason="No independent factual verifier"),
    )
    extension = _Policy("extension/1", CompletionAccept(outcome="verified"))

    decision = await LayeredCompletionPolicy(required, extension).verify(
        cast(CompletionContext, object())
    )

    assert decision == CompletionUnverifiable(reason="No independent factual verifier")
    assert extension.calls == 1
