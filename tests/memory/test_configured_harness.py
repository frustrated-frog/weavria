"""同一推荐入口贯穿真实适配器和存储;脚本模型只证明接线,不证明任务收益。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, JsonValue

from weavria.context import ContextCompilerConfig
from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelRequest,
    ModelResponse,
    ModelTurn,
    RecoverContext,
    RecoveryNotice,
    RunError,
    RunStatus,
    SessionContextCompiled,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
    ToolSucceededObservation,
)
from weavria.harness import (
    AgentHarness,
    CapabilityNeed,
    CompletionCriterion,
    HarnessConfig,
    HarnessMemory,
    InformationNeed,
    TaskSpecification,
    WorkingState,
    WorkNote,
    WorkReference,
    WorkUpdate,
)
from weavria.memory import (
    MemoryAccess,
    MemoryCandidate,
    MemoryCitation,
    MemoryFormationInput,
    MemoryJudgment,
    MemoryScope,
)
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCallError,
    ModelCapabilities,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolFault, ToolRegistry, TypedTool

PREFERENCE = "documentation language 中文"
QUESTION = "Read the document using our documentation language"


def configuration() -> HarnessConfig:
    return HarnessConfig(
        configuration_id="configured-chain/1",
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        limits=RunLimits(max_request_bytes=64000),
        working_records=True,
        memory_write="basic",
    )


class ReasoningModel:
    """只替换模型端口,不替换理解、提取、审阅、准入或检索算法。"""

    capabilities = ModelCapabilities(tool_calls=True)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.time_kind: Literal["current", "unknown"] = "current"
        self.fail_formation = False
        self.reject_candidate = False
        self.block_formation = False
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        arguments: dict[str, JsonValue]
        if "declare_task_specification" in request.instructions:
            execute = request.task.goal == QUESTION
            specification = TaskSpecification(
                route="execute" if execute else "direct",
                intent="Read document" if execute else "Answer",
                goal=request.task.goal,
                information_needs=(InformationNeed(description="document", source="external"),)
                if execute
                else (),
                capability_needs=tuple(
                    CapabilityNeed(name=n, purpose="Read document", necessity="required")
                    for n in ("a_source", "b_source")
                )
                if execute
                else (),
                completion_criteria=(
                    CompletionCriterion(kind="response", description="Answer"),
                    *(
                        CompletionCriterion(
                            kind="capability_result", description="Read", capability=n
                        )
                        for n in ("a_source", "b_source")
                        if execute
                    ),
                ),
            )
            name, arguments = "declare_task_specification", specification.model_dump(mode="json")
        elif "propose_memory" in request.instructions:
            if self.fail_formation:
                raise ModelCallError(
                    RunError(category=ErrorCategory.MODEL_FAILURE, message="private provider error")
                )
            if self.block_formation:
                self.entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
            payload = json.loads(request.task.goal)
            formation = MemoryFormationInput.model_validate_json(json.dumps(payload["input"]))
            # 来源取实际接受的用户事件,不预填数据库或伪造正确记忆。
            doc = next(
                d for d in formation.documents if d.source.source_id == formation.primary_source_id
            )
            candidates = (
                ()
                if doc.text != PREFERENCE
                else (
                    MemoryCandidate(
                        subject=formation.scope.key,
                        predicate="documentation_language",
                        value="中文",
                        text=doc.text,
                        scope_kind="project",
                        retention="durable",
                        time_kind=self.time_kind,
                        citations=(MemoryCitation(source_id=doc.source.source_id, quote=doc.text),),
                    ),
                )
            )
            name = "propose_memory"
            arguments = {
                "schema_version": 1,
                "candidates": [c.model_dump(mode="json") for c in candidates],
            }
        else:
            assert "review_memory" in request.instructions
            judgment = MemoryJudgment(
                candidate_index=0,
                supported=not self.reject_candidate,
                scope_supported=True,
                conditions_preserved=True,
                time_supported=True,
                relation="new",
            )
            name = "review_memory"
            arguments = {
                "schema_version": 2,
                "judgments": [judgment.model_dump(mode="json")],
                "alignments": [],
            }
        return ModelResponse(
            action=ToolCallAction(
                call_id=f"reasoning-{len(self.requests)}", tool_name=name, arguments=arguments
            )
        )


class Input(BaseModel):
    key: str


class Output(BaseModel):
    value: str


def call(name: str, arguments: dict[str, JsonValue], call_id: str | None = None) -> ModelTurn:
    return ModelTurn(
        decision=ToolBatch(
            calls=(ToolCallAction(call_id=call_id or name, tool_name=name, arguments=arguments),)
        )
    )


async def test_configured_chain_forms_recalls_executes_updates_and_recovers(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    calls: list[str] = []

    async def primary(arguments: Input, context: ToolContext) -> Output:
        calls.append("a_source")
        raise ToolFault(ErrorCategory.TOOL_FAILURE, "unavailable", retryable=False)

    async def backup(arguments: Input, context: ToolContext) -> Output:
        calls.append("b_source")
        return Output(value="accepted " * 5000 + "MIDDLE" + "accepted " * 5000 + "TAIL")

    registry = ToolRegistry(
        tuple(
            TypedTool(
                name=n,
                version="1",
                description="Read document",
                input_type=Input,
                output_type=Output,
                execute=f,
            )
            for n, f in (("a_source", primary), ("b_source", backup))
        )
    )

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

        def __init__(self) -> None:
            self.requests: list[SessionRequest] = []
            self.step = 0
            self.error: Exception | None = None

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            # Runtime 会归一化模型异常;测试另外保留断言,让失败能定位到交互步骤。
            try:
                return await self.respond(request)
            except Exception as exc:
                self.error = exc
                raise

        async def respond(self, request: SessionRequest) -> ModelTurn:
            self.requests.append(request)
            if request.current.task.goal != QUESTION:
                return ModelTurn(decision=FinishAction(output="记住了"))
            self.step += 1
            assert request.bundles and "中文" in request.bundles[0].model_dump_json()
            assert "中文" not in request.current.instructions
            assert request.current.recovery_enabled
            view = request.current.task_model_view
            assert view is not None
            state = WorkingState.model_validate_json(json.dumps(view.progress["working_state"]))
            exchanges = [e for e in request.current.history if isinstance(e, ToolExchange)]
            if self.step == 1:
                return call("a_source", {"key": "document"})
            if self.step == 2:
                assert exchanges, request.current.model_dump_json()
                assert isinstance(exchanges[-1].observation, ToolFailedObservation)
                return call("b_source", {"key": "document"})
            if self.step == 3:
                # 需求完成会退休业务工具,但成果读取和工作更新仍必须可用。
                assert '"name":"b_source"' not in request.current.instructions
                assert "read_artifact" in request.current.instructions
                assert "propose_work" in request.current.instructions
                source = next(e for e in exchanges if e.action.call_id == "b_source")
                assert isinstance(source.observation, ToolSucceededObservation)
                output = source.observation.output
                assert isinstance(output, dict) and isinstance(output["artifact_id"], str)
                assert output["representation"] != "full"
                needs = view.definition["needs"]
                assert isinstance(needs, list) and isinstance(needs[0], dict)
                need_id = needs[0]["id"]
                assert isinstance(need_id, str)
                update = WorkUpdate(
                    task_revision=1,
                    base_revision=0,
                    notes=(
                        WorkNote(
                            note_id="result",
                            kind="result",
                            summary="Check the source tail",
                            need_ids=(need_id,),
                            sources=(
                                WorkReference(artifact_id=output["artifact_id"], pointer="/value"),
                            ),
                        ),
                    ),
                    next_need_ids=(need_id,),
                    next_query="document tail",
                )
                return call("propose_work", update.model_dump(mode="json"))
            assert (
                state.last_notice == "accepted" and state.sources[0].artifact.call_id == "b_source"
            )
            if self.step == 4:
                # 通过控制面恢复原文片段,不会重新执行来源工具。
                return ModelTurn(
                    schema_version=2,
                    decision=RecoverContext(
                        call_id="b_source",
                        source_sha256=state.sources[0].artifact.sha256,
                        pointer="/value",
                        start=89980,
                        length=30,
                    ),
                )
            if self.step == 5:
                assert isinstance(request.current.control, RecoveryNotice)
                assert request.current.control.status == "recovered"
                assert "TAIL" in request.model_dump_json()
                # 完成阶段的 Artifact 导航按现有协议从首页开启,随后按游标续读。
                # 任意位置的精确回取由上一步 RecoverContext 承担。
                return call(
                    "read_artifact",
                    {
                        "artifact_id": state.sources[0].artifact.artifact_id,
                        "pointer": "/value",
                        "start": 0,
                        "length": 64,
                    },
                    call_id=f"readback-{self.step}",
                )
            assert self.step == 6
            assert any(
                e.action.tool_name == "read_artifact" and "accepted" in e.model_dump_json()
                for e in exchanges
            ), (request.current.control, request.current.feedback)
            return ModelTurn(decision=FinishAction(output="已按中文约定读取到 TAIL"))

    reasoning, model = ReasoningModel(), Model()
    async with AgentHarness.from_config(
        tmp_path / "session.db",
        config=configuration(),
        model=model,
        tools=registry,
        token_counter=session_counter,
        reasoning_model=reasoning,
        reasoning_counter=session_counter,
        memory=HarnessMemory(tmp_path / "memory.db", access, scope, "author"),
    ) as agent:
        await agent.create("first")
        first = await agent.send("first", PREFERENCE, input_id="preference")
        assert first.memory.status == "completed", first.memory
        assert first.memory.extracted_candidates == first.memory.admitted_candidates == 1
        assert first.memory.rejected_candidates == first.memory.deferred_candidates == 0
        await agent.create("second")
        result = await agent.send("second", QUESTION, input_id="read")
        if model.error is not None:
            raise model.error
        assert result.session.report.status is RunStatus.COMPLETED, result.session.report
        assert result.memory.status == "completed"
        assert result.memory.extracted_candidates == result.memory.admitted_candidates == 0
        before = len(model.requests), len(reasoning.requests)
        repeated = await agent.send("second", QUESTION, input_id="read")
        assert repeated.session == result.session
        assert repeated.memory.processed_events == repeated.memory.admitted_candidates == 0
        assert (len(model.requests), len(reasoning.requests)) == before
        events = await agent.sessions.store.events("second", limit=256)
    interpretation = next(r for r in reasoning.requests if r.task.goal == QUESTION)
    assert any(
        "Scoped recalled context" in c and "中文" in c for c in interpretation.task.constraints
    )
    assert "中文" not in interpretation.instructions
    compilations = [
        e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)
    ]
    assert compilations and all(c.task_model_view is not None for c in compilations)
    assert all(c.retrieval is not None and c.supply is not None for c in compilations)
    assert any(c.recovery_ranges for c in compilations)
    assert calls == ["a_source", "b_source"]


@pytest.mark.parametrize("read_only", [False, True])
async def test_disabled_features_make_no_reasoning_calls(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
    read_only: bool,
) -> None:
    reasoning = FakeModelProvider(())
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="hello")),))
    config = replace(
        configuration(),
        task_understanding=False,
        working_records=False,
        memory_write="off",
        history_retrieval=False,
        context_recovery=False,
    )
    memory = HarnessMemory(tmp_path / "memory.db", access, scope, "author") if read_only else None
    async with AgentHarness.from_config(
        tmp_path / "session.db",
        config=config,
        model=model,
        tools=ToolRegistry(()),
        token_counter=session_counter,
        reasoning_model=reasoning,
        memory=memory,
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "hello")
        events = await agent.sessions.store.events("s")
    assert result.session.report.status is RunStatus.COMPLETED
    assert result.memory.status == ("read_only" if read_only else "disabled")
    assert not reasoning.requests
    assert model.requests[0].current.task_model_view is None
    assert not model.requests[0].current.recovery_enabled
    assert all(
        e.data.compilation.retrieval is None
        for e in events
        if isinstance(e.data, SessionContextCompiled)
    )


@pytest.mark.parametrize("failure", ["provider", "cancel", "reject"])
async def test_configured_formation_failure_and_cancellation_preserve_execution(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
    failure: str,
) -> None:
    reasoning = ReasoningModel()
    reasoning.fail_formation = failure == "provider"
    reasoning.block_formation = failure == "cancel"
    reasoning.reject_candidate = failure == "reject"
    model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="done")),))
    async with AgentHarness.from_config(
        tmp_path / "session.db",
        config=configuration(),
        model=model,
        tools=ToolRegistry(()),
        token_counter=session_counter,
        reasoning_model=reasoning,
        reasoning_counter=session_counter,
        memory=HarnessMemory(tmp_path / "memory.db", access, scope, "author"),
    ) as agent:
        await agent.create("s")
        if failure == "cancel":
            task = asyncio.create_task(agent.send("s", PREFERENCE, input_id="one"))
            await asyncio.wait_for(reasoning.entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert reasoning.cancelled.is_set()
        else:
            result = await agent.send("s", PREFERENCE, input_id="one")
            assert result.session.report.status is RunStatus.COMPLETED
            if failure == "provider":
                assert result.memory.status == "failed"
                assert result.memory.error_type == "MemoryFormationError"
                assert result.memory.admitted_candidates == 0
            else:
                assert result.memory.status == "completed"
                assert result.memory.extracted_candidates == result.memory.rejected_candidates == 1
                assert result.memory.admitted_candidates == 0
        reasoning.fail_formation = reasoning.block_formation = False
        # 同一 input_id 只续接形成;取消/失败都不能重复已提交的模型执行。
        repeated = await agent.send("s", PREFERENCE, input_id="one")
        assert repeated.session.report.status is RunStatus.COMPLETED
        assert repeated.memory.status == "completed"
        if failure != "reject":
            assert (
                repeated.prior_memory.admitted_candidates + repeated.memory.admitted_candidates == 1
            )
        assert len(model.requests) == 1


def test_configuration_rejects_missing_dependencies_before_opening_storage(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    with pytest.raises(ValueError, match="working records require"):
        replace(configuration(), task_understanding=False)
    with pytest.raises(ValueError, match="reasoning model and counter"):
        AgentHarness.from_config(
            tmp_path / "session.db",
            config=configuration(),
            model=FakeSessionProvider(()),
            tools=ToolRegistry(()),
            token_counter=session_counter,
        )
    with pytest.raises(ValueError, match="storage, scope and a formation model"):
        AgentHarness.from_config(
            tmp_path / "session.db",
            config=replace(configuration(), task_understanding=False, working_records=False),
            model=FakeSessionProvider(()),
            tools=ToolRegistry(()),
            token_counter=session_counter,
        )
    assert not (tmp_path / "session.db").exists()


@pytest.mark.parametrize("include_unknown", [False, True])
async def test_configured_unknown_time_memory_survives_reopen_and_respects_host_choice(
    tmp_path: Path,
    access: MemoryAccess,
    scope: MemoryScope,
    session_counter: OpenAICompatibleTokenCounter,
    include_unknown: bool,
) -> None:
    reasoning = ReasoningModel()
    reasoning.time_kind = "unknown"
    memory = HarnessMemory(
        tmp_path / "memory.db", access, scope, "author", include_unknown_time=include_unknown
    )
    # 走真实模型适配器的抽取/审阅与存储,只替换模型响应;不预写正确记忆。
    for session, text in (("seed", PREFERENCE), ("recall", "What is documentation language?")):
        model = FakeSessionProvider((ModelTurn(decision=FinishAction(output="answer")),))
        async with AgentHarness.from_config(
            tmp_path / "sessions.db",
            config=configuration(),
            model=model,
            tools=ToolRegistry(()),
            token_counter=session_counter,
            reasoning_model=reasoning,
            reasoning_counter=session_counter,
            memory=memory,
        ) as agent:
            await agent.create(session)
            result = await agent.send(session, text, input_id="input")
            assert result.session.report.status is RunStatus.COMPLETED
            if session == "seed":
                assert result.memory.admitted_candidates == 1
            else:
                assert bool(model.requests[0].bundles) is include_unknown
                if include_unknown:
                    payload = model.requests[0].bundles[0].model_dump_json()
                    assert '"valid_from":null' in payload and "中文" in payload
