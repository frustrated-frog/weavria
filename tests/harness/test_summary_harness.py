"""推荐入口真实持久链路。脚本只替换模型,不替换压缩图、编译器或原文回取。"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelResponse,
    ModelTurn,
    RecoverContext,
    SessionContextCompiled,
    SessionRequest,
    SummaryAddress,
    SummaryDraft,
    SummaryStatement,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.harness import (
    AgentHarness,
    CapabilityNeed,
    CompletionCriterion,
    HarnessConfig,
    TaskSpecification,
)
from weavria.models import FakeModelProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class ReadInput(BaseModel):
    index: int


class ReadOutput(BaseModel):
    text: str


class SummaryModel:
    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self, hierarchy: bool = False) -> None:
        self.hierarchy = hierarchy
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        turn = len(self.requests)
        assert request.schema_version == 9
        assert (
            request.summary_advice is None
            or request.summary_advice.task_revision == request.current.task.revision
        )
        if turn == 1:
            assert "compact_context" in request.model_dump_json()
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(call_id="read", tool_name="read", arguments={"index": 0}),
                    )
                )
            )
        if turn == 2:
            exchange = next(
                h
                for h in request.current.history
                if isinstance(h, ToolExchange) and h.action.call_id == "read"
            )
            assert isinstance(exchange.observation, ToolSucceededObservation)
            output = exchange.observation.output
            assert isinstance(output, dict) and isinstance(output["artifact_id"], str)
            draft = SummaryDraft(
                task_revision=request.current.task.revision,
                sources=(SummaryAddress(artifact_id=output["artifact_id"], pointer="/text"),),
                statements=(
                    SummaryStatement(
                        kind="condition",
                        text=(
                            "cleanup cancellation keeps sibling results; "
                            "retry only idempotent reads"
                        )
                        * (8 if self.hierarchy else 1),
                        source_indices=(0,),
                    ),
                ),
            )
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="compact",
                            tool_name="compact_context",
                            arguments=draft.model_dump(mode="json"),
                        ),
                    )
                )
            )
        if turn == 3 and self.hierarchy:
            node = next(v for v in request.summaries if v.statements)
            draft = SummaryDraft(
                task_revision=request.current.task.revision,
                sources=(SummaryAddress(artifact_id=node.artifact.artifact_id),),
                statements=(
                    SummaryStatement(
                        kind="condition",
                        text=(
                            "cleanup cancellation keeps sibling results; "
                            "retry only idempotent reads"
                        ),
                        source_indices=(0,),
                    ),
                ),
            )
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="merge",
                            tool_name="compact_context",
                            arguments=draft.model_dump(mode="json"),
                        ),
                    )
                )
            )
        if turn == (4 if self.hierarchy else 3):
            summaries = [v for v in request.summaries if v.statements]
            assert len(summaries) == (2 if self.hierarchy else 1)
            assert max(v.depth for v in summaries) == (2 if self.hierarchy else 1)
            assert summaries[0].status == "available"
            leaf = summaries[0].sources[0]
            return ModelTurn(
                schema_version=2,
                decision=RecoverContext(
                    call_id=leaf.artifact.call_id,
                    source_sha256=leaf.artifact.sha256,
                    pointer=leaf.address.pointer,
                    length=180,
                ),
            )
        assert turn == (5 if self.hierarchy else 4)
        assert "exact original qualification" in request.model_dump_json()
        return ModelTurn(decision=FinishAction(output="verified original qualification"))


@pytest.mark.parametrize("understanding,hierarchy", ((False, False), (True, False), (False, True)))
@pytest.mark.parametrize("token_savings", (False, True))
async def test_harness_persists_summary_request_and_recovers_source(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    understanding: bool,
    hierarchy: bool,
    token_savings: bool,
) -> None:
    async def read(arguments: ReadInput, context: ToolContext) -> ReadOutput:
        return ReadOutput(
            text=(
                "exact original qualification: When a caller abandons an operation, preserve "
                "completed concurrent children. Repeat a read only when it has no extra effect.\n"
                + "Implementation discussion without additional conclusions.\n"
                * 100
            )
        )

    registry = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read original source",
                input_type=ReadInput,
                output_type=ReadOutput,
                execute=read,
            ),
        )
    )
    model = SummaryModel(hierarchy)
    config = HarnessConfig(
        configuration_id="semantic-compaction-contract/1",
        task_understanding=understanding,
        context=ContextCompilerConfig(
            window_tokens=32000,
            output_tokens=256,
            semantic_compaction=True,
            summary_guidance=True,
            summary_selection="token_savings" if token_savings else "coverage",
            policy="full" if hierarchy else "coverage",
        ),
        limits=RunLimits(max_request_bytes=32000, max_model_calls=8),
    )
    path = tmp_path / "session.sqlite"
    reasoning = FakeModelProvider(
        (
            ModelResponse(
                action=ToolCallAction(
                    call_id="interpret",
                    tool_name="declare_task_specification",
                    arguments=TaskSpecification(
                        route="execute",
                        intent="Inspect cancellation source",
                        goal="Read source and explain",
                        capability_needs=(
                            CapabilityNeed(
                                name="read", purpose="Inspect source", necessity="required"
                            ),
                        ),
                        completion_criteria=(
                            CompletionCriterion(kind="response", description="Explain"),
                            CompletionCriterion(
                                kind="capability_result",
                                description="Read source",
                                capability="read",
                            ),
                        ),
                    ).model_dump(mode="json"),
                )
            ),
        )
    )
    async with AgentHarness.from_config(
        path,
        config=config,
        model=model,
        tools=registry,
        token_counter=session_counter,
        reasoning_model=reasoning if understanding else None,
        reasoning_counter=session_counter if understanding else None,
    ) as agent:
        await agent.create("s")
        result = await agent.send(
            "s", "cleanup cancellation sibling results retry idempotent reads", input_id="one"
        )
        assert result.session.report.execution.output == "verified original qualification", (
            result.session.report.model_dump_json(),
            len(model.requests),
            model.requests[-1].model_dump_json(),
        )
        events = await agent.sessions.store.events("s", limit=256)
        reports = [e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)]
        assert reports and all(r.summaries is not None for r in reports)
        assert any(r.summaries and r.summaries.selected == 1 for r in reports)
        assert any(r.recovery_ranges for r in reports)
        assert all((r.summary_selection is not None) == token_savings for r in reports)
        if token_savings:
            assert all(r.schema_version == 17 for r in reports)
    async with AgentHarness.from_config(
        path,
        config=config,
        model=model,
        tools=registry,
        token_counter=session_counter,
        reasoning_model=reasoning if understanding else None,
        reasoning_counter=session_counter if understanding else None,
    ) as agent:
        repeated = await agent.send(
            "s", "cleanup cancellation sibling results retry idempotent reads", input_id="one"
        )
        assert repeated.session.run_id == result.session.run_id
        assert len(model.requests) == (5 if hierarchy else 4)


class PreparationModel:
    """按框架建议动作前进,陈述用替身固定。它不证明真实模型的策略选择。"""

    capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

    def __init__(self) -> None:
        self.requests: list[SessionRequest] = []

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        self.requests.append(request)
        assert request.schema_version == 10
        turn = len(self.requests)
        if turn == 1:
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(call_id="read", tool_name="read", arguments={"index": 0}),
                    )
                )
            )
        advice = request.summary_advice
        assert advice, request.model_dump_json()
        if turn == 2:
            assert advice.status == "recover" and advice.recovery, request.model_dump_json()
            assert not advice.sources
            return ModelTurn(schema_version=2, decision=advice.recovery)
        if turn == 3:
            assert advice.status == "compact" and advice.statement_budget_bytes >= 512
            assert "qualified cancellation" in request.model_dump_json()
            draft = SummaryDraft(
                task_revision=request.current.task.revision,
                sources=advice.sources,
                statements=(
                    SummaryStatement(
                        kind="condition",
                        text="qualified cancellation retains completed sibling results",
                        source_indices=tuple(range(len(advice.sources))),
                    ),
                ),
            )
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="compact",
                            tool_name="compact_context",
                            arguments=draft.model_dump(mode="json"),
                        ),
                    )
                )
            )
        assert turn == 4
        assert request.summaries
        return ModelTurn(decision=FinishAction(output="qualified cancellation verified"))


async def test_harness_prepares_compacts_and_reopens_without_hidden_model_calls(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    async def read(arguments: ReadInput, context: ToolContext) -> ReadOutput:
        return ReadOutput(text="qualified cancellation retains completed sibling results. " * 900)

    registry = ToolRegistry(
        (
            TypedTool(
                name="read",
                version="1",
                description="Read source",
                input_type=ReadInput,
                output_type=ReadOutput,
                execute=read,
            ),
        )
    )
    config = HarnessConfig(
        configuration_id="summary-preparation-contract/1",
        task_understanding=False,
        context=ContextCompilerConfig(
            window_tokens=20000,
            output_tokens=256,
            semantic_compaction=True,
            summary_guidance=True,
            summary_preparation=True,
            max_passages=1,
        ),
        limits=RunLimits(max_request_bytes=24000, max_model_calls=8),
    )
    model = PreparationModel()
    path = tmp_path / "preparation.sqlite"
    async with AgentHarness.from_config(
        path, config=config, model=model, tools=registry, token_counter=session_counter
    ) as agent:
        await agent.create("s")
        result = await agent.send("s", "qualified cancellation sibling results", input_id="one")
        assert result.session.report.output == "qualified cancellation verified", (
            result.session.report.model_dump_json(),
            [r.model_dump_json() for r in model.requests],
        )
        events = await agent.sessions.store.events("s", limit=256)
        reports = [e.data.compilation for e in events if isinstance(e.data, SessionContextCompiled)]
        assert all(r.schema_version == 16 for r in reports)
        assert any(r.summary_advice and r.summary_advice.status == "recover" for r in reports)
        assert any(r.summary_advice and r.summary_advice.status == "compact" for r in reports)
    async with AgentHarness.from_config(
        path, config=config, model=model, tools=registry, token_counter=session_counter
    ) as agent:
        replay = await agent.send("s", "qualified cancellation sibling results", input_id="one")
        assert replay.session.run_id == result.session.run_id
        assert len(model.requests) == 4
