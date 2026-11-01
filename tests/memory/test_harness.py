"""运行真实存储与编译链路。脚本提取器仅验证接线,不证明自然语言形成质量。"""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelResponse,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolCallAction,
)
from weavria.harness import (
    AgentHarness,
    CompletionCriterion,
    HarnessMemory,
    MemoryFormationError,
    ModelTaskInterpreter,
    TaskSpecification,
    TaskUnderstandingEngine,
    TaskUnderstandingInput,
)
from weavria.memory import (
    MemoryAccess,
    MemoryCandidate,
    MemoryCitation,
    MemoryExtraction,
    MemoryFormationInput,
    MemoryJudgment,
    MemoryReconciliation,
    MemoryReview,
    MemoryScope,
)
from weavria.models import FakeModelProvider, FakeSessionProvider, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionBusyError
from weavria.tools import ToolRegistry


class Extractor:
    version = "fixture/1"

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False
        self.block = False
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def extract(
        self, request: MemoryFormationInput, *, max_duration: float
    ) -> MemoryExtraction:
        self.calls += 1
        if self.fail:
            raise MemoryFormationError("private content must not appear in report")
        if self.block:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        doc = request.documents[-1]
        if doc.text != "documentation language 中文":
            return MemoryExtraction()
        return MemoryExtraction(
            candidates=(
                MemoryCandidate(
                    subject="project",
                    predicate="documentation_language",
                    value="中文",
                    text=doc.text,
                    scope_kind="project",
                    retention="durable",
                    time_kind="current",
                    citations=(MemoryCitation(source_id=doc.source.source_id, quote=doc.text),),
                ),
            )
        )


class Reviewer:
    version = "fixture/1"

    async def review(
        self,
        request: MemoryFormationInput,
        extraction: MemoryExtraction,
        snapshot: MemoryReconciliation,
        *,
        max_duration: float,
    ) -> MemoryReview:
        return MemoryReview(
            judgments=(
                MemoryJudgment(
                    candidate_index=0,
                    supported=True,
                    scope_supported=True,
                    conditions_preserved=True,
                    time_supported=True,
                    relation="new",
                ),
            )
        )


class Interpreter:
    version = "fixture/1"

    def __init__(self) -> None:
        self.inputs: list[TaskUnderstandingInput] = []

    async def interpret(
        self, request: TaskUnderstandingInput, *, max_duration: float
    ) -> TaskSpecification:
        self.inputs.append(request)
        return TaskSpecification(
            route="direct",
            intent="Answer the user",
            goal=request.task.goal,
            completion_criteria=(CompletionCriterion(kind="response", description="Answer"),),
        )


def config(
    tmp_path: Path, access: MemoryAccess, scope: MemoryScope, extractor: Extractor
) -> HarnessMemory:
    return HarnessMemory(tmp_path / "memory.db", access, scope, "author", extractor, Reviewer())


def harness(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    model: FakeSessionProvider,
    memory: HarnessMemory | None,
    interpreter: Interpreter | None = None,
) -> AgentHarness:
    return AgentHarness(
        tmp_path / "session.db",
        model=model,
        tools=ToolRegistry(()),
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        token_counter=session_counter,
        memory=memory,
        understanding=TaskUnderstandingEngine(interpreter) if interpreter else None,
        limits=RunLimits(max_request_bytes=64000),
    )


def finish() -> ModelTurn:
    return ModelTurn(decision=FinishAction(output="done"))


async def test_formation_then_new_session_understanding_and_decision_recall(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    extractor, interpreter = Extractor(), Interpreter()
    model = FakeSessionProvider((finish(), finish()))
    memory = config(tmp_path, access, scope, extractor)
    async with harness(tmp_path, session_counter, model, memory, interpreter) as agent:
        first = await agent.create("first")
        result = await agent.send(first, "documentation language 中文", input_id="a")
        assert result.session.report.status is RunStatus.COMPLETED
        assert result.memory.status == "completed"
        assert result.memory.cursor == result.memory.through_sequence
        second = await agent.create("second")
        await agent.send(second, "documentation language?", input_id="b")
        supply = interpreter.inputs[-1].context_supply
        assert supply is not None and len(supply.bundles) == 1
        assert "中文" in supply.model_dump_json()
        # 验证实际理解适配器也把供给编码进任务数据,并保留 v1 的旧序列化形状。
        recalled = interpreter.inputs[-1]
        assert recalled.schema_version == 2
        assert TaskUnderstandingInput.model_validate_json(recalled.model_dump_json()) == recalled
        old = TaskUnderstandingInput(task=TaskInput(goal="hello"))
        assert "context_supply" not in old.model_dump()
        with pytest.raises(ValidationError, match="requires understanding input v2"):
            TaskUnderstandingInput(task=recalled.task, context_supply=supply)
        specification = await interpreter.interpret(recalled, max_duration=1)
        wire_model = FakeModelProvider(
            (
                ModelResponse(
                    action=ToolCallAction(
                        call_id="interpret",
                        tool_name="declare_task_specification",
                        arguments=specification.model_dump(mode="json"),
                    )
                ),
            )
        )
        await ModelTaskInterpreter(wire_model).interpret(recalled, max_duration=1)
        assert any(
            "Scoped recalled context" in c and "中文" in c
            for c in wire_model.requests[0].task.constraints
        )
        assert "中文" not in wire_model.requests[0].instructions

        assert model.requests[-1].bundles[0].content["group"] == supply.bundles[0].content["group"]
        assert "中文" not in model.requests[-1].current.instructions
        # 重试已提交输入只续接游标,不重跑决策或已经消费的提取。
        calls = extractor.calls
        repeated = await agent.send(second, "documentation language?", input_id="b")
        assert repeated.memory.processed_events == 0
        assert len(model.requests) == 2 and extractor.calls == calls


async def test_failed_formation_preserves_answer_and_reopens_without_model_reexecution(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    extractor = Extractor()
    extractor.fail = True
    memory = config(tmp_path, access, scope, extractor)
    async with harness(
        tmp_path, session_counter, FakeSessionProvider((finish(),)), memory
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "documentation language 中文", input_id="a")
        assert result.session.report.status is RunStatus.COMPLETED
        assert result.memory.status == "failed"
        assert result.memory.error_type == "MemoryFormationError"
        assert "private" not in repr(result.memory)
    extractor.fail = False
    model = FakeSessionProvider(())
    async with harness(tmp_path, session_counter, model, memory) as agent:
        repeated = await agent.send("s", "documentation language 中文", input_id="a")
        assert repeated.session == result.session
        assert repeated.memory.status == "completed"
        assert model.requests == ()


@pytest.mark.parametrize("close", [False, True])
async def test_cancel_formation_joins_request_and_retains_committed_run(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
    close: bool,
) -> None:
    extractor = Extractor()
    extractor.block = True
    memory = config(tmp_path, access, scope, extractor)
    async with harness(
        tmp_path, session_counter, FakeSessionProvider((finish(),)), memory
    ) as agent:
        await agent.create("s")
        task = asyncio.create_task(agent.send("s", "documentation language 中文", input_id="a"))
        await asyncio.wait_for(extractor.entered.wait(), 5)
        with pytest.raises(SessionBusyError):
            await agent.send("s", "another input")
        if close:
            await agent.close()
        else:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert extractor.cancelled.is_set()
    extractor.block = False
    async with harness(tmp_path, session_counter, FakeSessionProvider(()), memory) as agent:
        result = await agent.send("s", "documentation language 中文", input_id="a")
        assert result.session.report.status is RunStatus.COMPLETED
        assert result.memory.status == "completed"


async def test_bounded_formation_and_explicit_drain(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    memory = replace(config(tmp_path, access, scope, Extractor()), max_events=1)
    async with harness(
        tmp_path, session_counter, FakeSessionProvider((finish(),)), memory
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "hello")
        assert result.memory.status == "deferred"
        assert result.memory.processed_events == 1
        report = result.memory
        for _ in range(100):
            report = await agent.flush_memory("s")
            if report.status == "completed":
                break
        assert report.status == "completed"
        assert report.cursor == report.through_sequence


async def test_formation_timeout_is_deferred_not_run_failure(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    extractor = Extractor()
    extractor.block = True
    memory = replace(config(tmp_path, access, scope, extractor), max_duration=0.1)
    async with harness(
        tmp_path, session_counter, FakeSessionProvider((finish(),)), memory
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "hello")
        assert result.session.report.status is RunStatus.COMPLETED
        assert result.memory.status == "deferred"
        assert result.memory.error_type == "TimeoutError"
        assert extractor.cancelled.is_set()


@pytest.mark.parametrize("enabled", [False, True])
async def test_optional_read_only_memory(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
    enabled: bool,
) -> None:
    memory = HarnessMemory(tmp_path / "memory.db", access, scope, "author") if enabled else None
    async with harness(
        tmp_path, session_counter, FakeSessionProvider((finish(),)), memory
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "hello")
        assert result.memory.status == ("read_only" if enabled else "disabled")


@pytest.mark.parametrize("profile", ["basic", "fact_sets", "conditions"])
def test_model_memory_profile_assembles_compatible_components(
    tmp_path: Path, access: MemoryAccess, scope: MemoryScope, profile: str
) -> None:
    from typing import Literal, cast

    from weavria.harness import MemoryReviewFeatures, ModelMemoryReviewer

    memory = HarnessMemory.from_model(
        tmp_path / "memory.db",
        access,
        scope,
        "author",
        model=FakeModelProvider(()),
        configuration_id="fixture",
        profile=cast(Literal["basic", "fact_sets", "conditions"], profile),
    )
    assert isinstance(memory.reviewer, ModelMemoryReviewer)
    assert memory.admission is not None
    expected = MemoryReviewFeatures(profile != "basic", profile == "conditions")
    assert memory.reviewer.features == memory.admission.features == expected
    if profile != "basic":
        # 原来的入口可接受此配置,直到调用模型后才因准入不支持而失败。
        with pytest.raises(ValueError, match="not supported by admission"):
            replace(memory, admission=None)


@pytest.mark.parametrize("interruption", ["none", "steering", "cancel"])
async def test_steering_retrieves_new_background_before_reinterpretation(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
) -> None:
    from weavria.context import ContextDemand
    from weavria.core import ContextSupply, SessionRequest
    from weavria.harness import MemoryContextSource

    entered, release = asyncio.Event(), asyncio.Event()
    retrieving, continue_read, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = MemoryContextSource.retrieve

    async def delayed(
        self: MemoryContextSource,
        *,
        session_id: str,
        run_id: str,
        task: TaskInput,
        demands: tuple[ContextDemand, ...],
    ) -> ContextSupply:
        supply = await original(
            self, session_id=session_id, run_id=run_id, task=task, demands=demands
        )
        if task.revision == 2 and interruption != "none":
            retrieving.set()
            try:
                await continue_read.wait()
            finally:
                cleaned.set()
        return supply

    monkeypatch.setattr(MemoryContextSource, "retrieve", delayed)

    class Model(FakeSessionProvider):
        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            if request.current.task.goal == "hello":
                entered.set()
                await release.wait()
            return await super().generate_session(request)

    interpreter = Interpreter()
    model = Model((finish(), finish(), finish()))
    memory = config(tmp_path, access, scope, Extractor())
    async with harness(tmp_path, session_counter, model, memory, interpreter) as agent:
        await agent.create("saved")
        await agent.send("saved", "documentation language 中文")
        await agent.create("work")
        task = asyncio.create_task(agent.send("work", "hello"))
        await asyncio.wait_for(entered.wait(), 5)
        await agent.steer("work", "documentation language?", input_id="change")
        release.set()
        if interruption != "none":
            await asyncio.wait_for(retrieving.wait(), 5)
            if interruption == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert cleaned.is_set()
                assert all(r.current.task.revision == 1 for r in model.requests)
                return
            await agent.steer("work", "documentation language for this project?", input_id="newer")
            continue_read.set()
        result = await asyncio.wait_for(task, 5)
    assert result.session.report.status is RunStatus.COMPLETED
    revised = interpreter.inputs[-1]
    expected_revision = 3 if interruption == "steering" else 2
    assert revised.task.revision == expected_revision
    assert revised.context_supply is not None
    assert revised.context_supply.report.task_revision == expected_revision
    assert "中文" in revised.context_supply.model_dump_json()
    assert model.requests[-1].current.task.revision == expected_revision
    if interruption == "steering":
        assert all(r.current.task.revision != 2 for r in model.requests)
    assert "中文" in str(model.requests[-1].bundles)
    # 首次无关目标的背景不可代替新任务的查询结果。
    initial = interpreter.inputs[1]
    assert initial.task.revision == 1
    assert initial.context_supply is not None
    assert (
        initial.context_supply.report.demand_sha256 != revised.context_supply.report.demand_sha256
    )


async def test_parent_deadline_defers_post_memory_and_retry_finishes_it(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    extractor = Extractor()
    extractor.block = True
    memory = config(tmp_path, access, scope, extractor)
    model = FakeSessionProvider((finish(),))
    agent = AgentHarness(
        tmp_path / "session.db",
        model=model,
        tools=ToolRegistry(()),
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        token_counter=session_counter,
        memory=memory,
        max_request_duration=0.4,
    )
    async with agent:
        await agent.create("s")
        # 提取器的局部上限是 30 秒,父请求 0.4 秒必须先截断它。
        result = await asyncio.wait_for(
            agent.send("s", "documentation language 中文", input_id="a"), 3
        )
        assert result.session.report.status is RunStatus.COMPLETED
        assert result.memory.status == "deferred"
        assert extractor.cancelled.is_set()
        extractor.block = False
        repeated = await agent.send("s", "documentation language 中文", input_id="a")
        assert repeated.session == result.session
        assert repeated.memory.status == "completed"
        assert len(model.requests) == 1


async def test_catchup_cannot_spend_session_reserved_time(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    extractor = Extractor()
    extractor.fail = True
    memory = config(tmp_path, access, scope, extractor)
    async with harness(
        tmp_path, session_counter, FakeSessionProvider((finish(),)), memory
    ) as first:
        await first.create("s")
        result = await first.send("s", "documentation language 中文")
        assert result.memory.status == "failed"
    extractor.fail, extractor.block = False, True
    model = FakeSessionProvider((finish(),))
    async with AgentHarness(
        tmp_path / "session.db",
        model=model,
        tools=ToolRegistry(()),
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        token_counter=session_counter,
        memory=memory,
        max_request_duration=0.6,
        memory_catchup_fraction=0.1,
    ) as agent:
        result = await asyncio.wait_for(agent.send("s", "hello"), 3)
        assert result.prior_memory.status == "deferred"
        assert result.session.report.status is RunStatus.COMPLETED
        assert len(model.requests) == 1
        assert extractor.cancelled.is_set()


async def test_enhanced_review_is_admitted_through_unified_send(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    from weavria.harness import ConservativeMemoryAdmission, MemoryReviewFeatures
    from weavria.memory import (
        MemoryAlignment,
        MemoryFieldMatch,
        MemoryGenerationAttempt,
        MemoryGenerationReport,
        MemoryResolution,
    )

    class RevisingExtractor(Extractor):
        async def extract(
            self, request: MemoryFormationInput, *, max_duration: float
        ) -> MemoryExtraction:
            doc = request.documents[-1]
            if not doc.text.startswith("documentation language "):
                return MemoryExtraction()
            return MemoryExtraction(
                candidates=(
                    MemoryCandidate(
                        subject="project",
                        predicate="documentation_language",
                        value=doc.text.split()[-1],
                        text=doc.text,
                        scope_kind="project",
                        retention="durable",
                        time_kind="current",
                        citations=(MemoryCitation(source_id=doc.source.source_id, quote=doc.text),),
                    ),
                )
            )

    class RevisingReviewer(Reviewer):
        features = MemoryReviewFeatures(set_reconciliation=True)

        async def review(
            self,
            request: MemoryFormationInput,
            extraction: MemoryExtraction,
            snapshot: MemoryReconciliation,
            *,
            max_duration: float,
        ) -> MemoryReview:
            if not snapshot.versions:
                return await super().review(
                    request, extraction, snapshot, max_duration=max_duration
                )
            target = snapshot.versions[0]
            citations = extraction.candidates[0].citations
            return MemoryReview(
                schema_version=5,
                judgments=(
                    MemoryJudgment(
                        candidate_index=0,
                        supported=True,
                        scope_supported=True,
                        conditions_preserved=True,
                        time_supported=True,
                        relation="correction",
                        target_id=target.assertion.memory_id,
                    ),
                ),
                alignments=(
                    MemoryAlignment(
                        candidate_index=0,
                        target_id=target.assertion.memory_id,
                        target_revision=target.revision,
                        matches=tuple(
                            MemoryFieldMatch(
                                candidate_path=p,
                                target_path=p,
                                citations=citations,
                            )
                            for p in ("/subject", "/predicate")
                        ),
                    ),
                ),
                resolutions=(
                    MemoryResolution(
                        candidate_index=0,
                        target_ids=(target.assertion.memory_id,),
                        citations=citations,
                    ),
                ),
                generation=MemoryGenerationReport(
                    stage="review",
                    policy_version="fixture/1",
                    attempts=(
                        MemoryGenerationAttempt(
                            number=1,
                            reason="accepted",
                            request_sha256="a" * 64,
                            elapsed_ms=0,
                        ),
                    ),
                ),
            )

    memory = HarnessMemory(
        tmp_path / "memory.db",
        access,
        scope,
        "author",
        RevisingExtractor(),
        RevisingReviewer(),
        admission=ConservativeMemoryAdmission(set_reconciliation=True),
    )
    model = FakeSessionProvider((finish(), finish(), finish()))
    async with harness(tmp_path, session_counter, model, memory, Interpreter()) as agent:
        await agent.create("s")
        assert (await agent.send("s", "documentation language 中文")).memory.status == "completed"
        assert (
            await agent.send("s", "documentation language English")
        ).memory.status == "completed"
        await agent.create("recall")
        await agent.send("recall", "documentation language?")
        assert "English" in str(model.requests[-1].bundles)


async def test_harness_preserves_optional_completion_policy(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    from weavria.core import CompletionAccept, CompletionDecision, CompletionRepair
    from weavria.runtime import CompletionContext

    class Check:
        version = "fixture/1"

        async def verify(self, context: CompletionContext) -> CompletionDecision:
            if context.candidate.output == "draft":
                return CompletionRepair(
                    issue_codes=("missing_result",), guidance="Deliver corrected result"
                )
            return CompletionAccept(outcome="verified")

    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="draft")), finish()))
    async with AgentHarness(
        tmp_path / "session.db",
        model=model,
        tools=ToolRegistry(()),
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        token_counter=session_counter,
        completion=Check(),
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "answer")
    assert len(model.requests) == 2
    assert result.session.report.status is RunStatus.COMPLETED
