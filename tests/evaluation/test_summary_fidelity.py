"""配对探针验证损失归因与失败语义;替身不作为真实摘要质量证据。"""

import asyncio
import hashlib
import json

import pytest

from examples.summary_fidelity import head_tail, load_cases, prepare_summary, summary_request
from examples.summary_token_selection import replay
from weavria.core import FinishAction, ModelRequest, ModelResponse, SummaryAddress
from weavria.evaluation import (
    FidelityCase,
    FidelityCost,
    FidelityProbe,
    compare_fidelity,
    evaluate_fidelity,
    probe_request,
    score_fidelity,
)
from weavria.models import ModelCapabilities, OpenAICompatibleTokenCounter


def case() -> FidelityCase:
    source = "Retry only idempotent reads. Completed sibling results survive cancellation."
    return FidelityCase(
        case_id="cancellation",
        source_path="docs/example.md",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        task="Review retry and cancellation.",
        probes=(
            FidelityProbe(
                probe_id="retry",
                facet="condition",
                question="Can every action retry?",
                answers=("no",),
                evidence=("Retry only idempotent reads.",),
            ),
            FidelityProbe(
                probe_id="siblings",
                facet="exception",
                question="Do completed results survive?",
                answers=("yes",),
                evidence=("Completed sibling results survive cancellation.",),
            ),
            FidelityProbe(probe_id="delay", facet="unknown", question="What is the retry delay?"),
        ),
    )


def response(
    retry: str | None = "no", *, delay: str | None = None, quote: str = "Supported"
) -> str:
    return json.dumps(
        {
            "answers": [
                {"probe_id": "retry", "answer": retry, "quote": quote if retry else None},
                {"probe_id": "siblings", "answer": "yes", "quote": quote},
                {"probe_id": "delay", "answer": delay, "quote": quote if delay else None},
            ]
        }
    )


def test_probe_grading_separates_omission_incorrect_and_unsupported() -> None:
    for answer, quote, outcome in (
        (None, "Supported", "omitted"),
        ("yes", "Supported", "incorrect"),
        ("no", "Invented", "unsupported"),
        (" NO ", "Supported", "supported"),
    ):
        result = score_fidelity(case(), "Supported", response(answer, quote=quote))
        assert result.status == "scored" and result.scores[0].outcome == outcome
    unsafe = score_fidelity(case(), "Supported", response(delay="10"))
    assert unsafe.scores[-1].outcome == "unsafe_assertion"
    # 正确标签加确切引句仍只是探针支持;测试不宣称这句引文逻辑蕴含了答案。
    assert "Supported" not in unsafe.model_dump_json()


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "unknown", "extra_field", "malformed"]
)
def test_incomplete_or_ambiguous_answers_cannot_shrink_the_denominator(mutation: str) -> None:
    value = json.loads(response())
    if mutation == "duplicate":
        value["answers"].append(value["answers"][0])
    elif mutation == "missing":
        value["answers"].pop()
    elif mutation == "unknown":
        value["answers"][0]["probe_id"] = "unregistered"
    elif mutation == "extra_field":
        value["answers"][0]["confidence"] = 1
    encoded = "{" if mutation == "malformed" else json.dumps(value)
    result = score_fidelity(case(), "Supported", encoded)
    assert result.status == "invalid_output" and not result.scores


def test_labels_are_isolated_from_both_model_requests() -> None:
    original, source = load_cases()[0]
    updated = original.model_copy(
        update={
            "probes": (
                original.probes[0].model_copy(update={"answers": ("SECRET_ORACLE",)}),
                *original.probes[1:],
            )
        }
    )
    address = SummaryAddress(artifact_id="accepted", pointer="/text")
    assert summary_request(original, source, address, statement_bytes=1400) == summary_request(
        updated, source, address, statement_bytes=1400
    )
    assert probe_request(original, "Candidate") == probe_request(updated, "Candidate")
    assert "SECRET_ORACLE" not in probe_request(updated, "Candidate").model_dump_json()
    for frozen, text in load_cases():
        frozen.validate_source(text)
        with pytest.raises(ValueError, match="frozen"):
            frozen.validate_source(text + "modified")


def test_paired_baseline_prevents_misattributing_reader_failures() -> None:
    baseline = score_fidelity(case(), "Supported", response("yes"))
    candidate = score_fidelity(case(), "Supported", response(None))
    result = compare_fidelity(
        baseline,
        candidate,
        cost=FidelityCost(
            original_input_tokens=1000,
            candidate_input_tokens=200,
            generation_total_tokens=1601,
        ),
    )
    assert result.status == "inconclusive" and result.baseline_unreliable == ("retry",)
    assert result.lost == () and result.break_even_reuses == 3
    baseline = score_fidelity(case(), "Supported", response())
    result = compare_fidelity(
        baseline,
        candidate,
        cost=FidelityCost(
            original_input_tokens=1000,
            candidate_input_tokens=200,
            generation_total_tokens=1,
        ),
    )
    assert result.status == "regression" and result.lost == ("retry",)


def test_unknown_cost_and_no_token_savings_do_not_become_success() -> None:
    baseline = score_fidelity(case(), "Supported", response())
    for cost, status, reuse in (
        (FidelityCost(), "inconclusive", None),
        (
            FidelityCost(
                original_input_tokens=100, candidate_input_tokens=100, generation_total_tokens=0
            ),
            "no_savings",
            None,
        ),
        (
            FidelityCost(
                original_input_tokens=100, candidate_input_tokens=20, generation_total_tokens=0
            ),
            "eligible_on_probes",
            0,
        ),
    ):
        comparison = compare_fidelity(baseline, baseline, cost=cost)
        assert comparison.status == status and comparison.break_even_reuses == reuse
    invalid = score_fidelity(case(), "Supported", "{}")
    assert compare_fidelity(baseline, invalid, cost=FidelityCost()).status == "invalid_trial"
    with pytest.raises(ValueError, match="same frozen"):
        compare_fidelity(
            baseline, baseline.model_copy(update={"case_sha256": "0" * 64}), cost=FidelityCost()
        )


def test_json_escaping_does_not_fabricate_a_loss_or_join_unrelated_fields() -> None:
    quote = 'The default is "off".\nConditions apply.'
    represented = json.dumps({"statements": [{"text": quote}], "other": "Unrelated"})
    result = score_fidelity(case(), represented, response(quote=quote))
    assert all(s.outcome == "supported" for s in result.scores[:-1])
    joined = score_fidelity(case(), represented, response(quote=quote + "Unrelated"))
    assert joined.scores[0].outcome == "unsupported"


class BlockedModel:
    capabilities = ModelCapabilities(tool_calls=False)

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.cleaned.set()
        raise AssertionError("unreachable")


async def test_evaluation_deadline_and_parent_cancel_wait_for_cleanup() -> None:
    timeout_model = BlockedModel()
    timed = await evaluate_fidelity(case(), "source", model=timeout_model, max_duration=0.01)
    assert timed.status == "model_failure" and timed.error_category == "timeout"
    assert timeout_model.cleaned.is_set()
    model = BlockedModel()
    task = asyncio.create_task(evaluate_fidelity(case(), "source", model=model))
    await asyncio.wait_for(model.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.cleaned.is_set()


class SummaryModel:
    capabilities = ModelCapabilities(tool_calls=False)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.task.goal)
        return ModelResponse(
            action=FinishAction(
                output=json.dumps(
                    {
                        "task_revision": 1,
                        "sources": payload["sources"],
                        "statements": [
                            {
                                "kind": "condition",
                                "text": "只有已接受的当前运行成果可被恢复。",
                                "source_indices": [0],
                            }
                        ],
                    }
                )
            )
        )


async def test_live_entry_uses_production_summary_graph_without_original_body(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    frozen, source = load_cases()[0]
    representation, status = await prepare_summary(
        frozen,
        source,
        model=SummaryModel(),
        counter=session_counter,
        statement_bytes=1400,
    )
    assert status == "accepted" and representation
    view = json.loads(representation)
    assert view["depth"] == 1 and view["trust"] == "model_inference"
    assert view["sources"][0]["address"]["pointer"] == "/text"
    assert "## 搜索、去重和额度" not in representation
    # 摘录和摘要匹配的是整个模型表示大小,不会把来源元数据从成本中隐藏掉。
    assert len(head_tail(source, len(representation.encode())).encode()) <= len(
        representation.encode()
    )


async def test_offline_token_replay_checks_frozen_source_and_compares_production_policies(
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    from weavria.core import ContextSummaryView

    frozen, source = load_cases()[0]
    representation, status = await prepare_summary(
        frozen, source, model=SummaryModel(), counter=session_counter, statement_bytes=1400
    )
    assert status == "accepted" and representation
    view = ContextSummaryView.model_validate_json(representation)
    report = await replay(frozen, source, view, session_counter)
    assert report.candidates == 1 and report.comparisons[0].decision == "saves_tokens"
    changed = view.model_copy(
        update={"sources": (view.sources[0].model_copy(update={"value_sha256": "a" * 64}),)}
    )
    with pytest.raises(ValueError, match="frozen source"):
        await replay(frozen, source, changed, session_counter)


@pytest.mark.parametrize("failure", ["invalid_source", "statement_budget", "invalid_generation"])
async def test_generation_failures_are_not_repaired_or_graded_as_summaries(
    session_counter: OpenAICompatibleTokenCounter,
    failure: str,
) -> None:
    class InvalidSummary(SummaryModel):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            generated = await super().generate(request)
            assert isinstance(generated.action, FinishAction)
            value = json.loads(generated.action.output)
            if failure == "invalid_source":
                value["sources"][0]["artifact_id"] = "unaccepted"
            elif failure == "statement_budget":
                value["statements"][0]["text"] = "条件" * 300
            else:
                value = {"unexpected": "value"}
            return ModelResponse(action=FinishAction(output=json.dumps(value)))

    model = InvalidSummary()
    frozen, source = load_cases()[0]
    representation, status = await prepare_summary(
        frozen,
        source,
        model=model,
        counter=session_counter,
        statement_bytes=1400,
    )
    assert representation is None and status == failure and model.calls == 1


def test_head_tail_respects_utf8_budget_without_learning_gold_labels() -> None:
    source = '条件🙂 "quote"\n' * 1000
    for budget in (32, 97, 256):
        excerpt = head_tail(source, budget)
        assert len(excerpt.encode()) <= budget and "omitted" in excerpt
    assert head_tail("short", 32) == "short"
