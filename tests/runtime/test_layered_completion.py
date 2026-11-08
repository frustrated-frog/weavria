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
    # 必需结果仍有缺口时, 即使发布层愿意接受也不能把候选答案放行.
    required = _Policy(
        "required/1",
        CompletionRepair(issue_codes=("required_output_mismatch",), guidance="修正输出"),
    )
    extension = _Policy("extension/1", CompletionAccept(outcome="verified"))
    policy = LayeredCompletionPolicy(required, extension)

    decision = await policy.verify(cast(CompletionContext, object()))

    assert isinstance(decision, CompletionRepair)
    # 第一层已经否决后应立即停止, 避免额外检查覆盖原始修复原因.
    assert extension.calls == 0


async def test_layered_completion_acceptance_keeps_the_weaker_outcome() -> None:
    # 一层只能确认可交付但未验证时, 整体结果也不能升级成 verified.
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
