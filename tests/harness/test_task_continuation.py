"""真实 Harness 入口, 使用脚本模型验证跨会话授权、事实复用和任务 CAS。"""

import asyncio
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, JsonValue

from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelResponse,
    ModelTurn,
    SessionRequest,
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
    InformationNeed,
    TaskSpecification,
)
from weavria.models import FakeModelProvider, ModelCapabilities, OpenAICompatibleTokenCounter
from weavria.runtime import RunLimits
from weavria.sessions import SessionConflictError
from weavria.tools import PolicyDecision, ToolContext, ToolDescriptor, ToolRegistry, TypedTool


class Empty(BaseModel):
    pass


class Value(BaseModel):
    value: str


class Access:
    def __init__(self, allowed: set[str]) -> None:
        self.allowed = allowed

    async def authorize(
        self, task_id: str, session_id: str, operation: Literal["read", "write"]
    ) -> bool:
        return task_id in {"compare", "copy"} and session_id in self.allowed


class SourcePolicy:
    revoked = False

    async def authorize(
        self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
    ) -> PolicyDecision:
        allowed = not self.revoked or descriptor.name != "read_alpha"
        return PolicyDecision(allowed, "test/1", "allowed" if allowed else "revoked")


def interpretation(tool: str) -> ModelResponse:
    spec = TaskSpecification(
        route="execute",
        intent="Compare two public documents",
        goal="Compare two public documents",
        capability_needs=(
            CapabilityNeed(name=tool, purpose="Read the missing document", necessity="required"),
        ),
        information_needs=(
            InformationNeed(
                description=f"Find the value from {tool}",
                source="external",
                freshness="stable",
                required=False,
                required_fields=("/value",),
                capability_names=(tool,),
            ),
        ),
        completion_criteria=(
            CompletionCriterion(
                kind="capability_result", description="Read source", capability=tool
            ),
            CompletionCriterion(kind="response", description="Report result"),
        ),
    )
    return ModelResponse(
        action=ToolCallAction(
            call_id="interpret",
            tool_name="declare_task_specification",
            arguments=spec.model_dump(mode="json"),
        ),
    )


@pytest.mark.parametrize(
    ("reconciliation", "paged"), [(False, False), (True, False), (False, True)]
)
async def test_new_sessions_reuse_accepted_result_and_cannot_overwrite_task_head(
    tmp_path: Path, session_counter: OpenAICompatibleTokenCounter, reconciliation: bool, paged: bool
) -> None:
    calls: list[str] = []
    both_reading = asyncio.Event()
    continuation_count = 0

    async def alpha(args: Empty, context: ToolContext) -> Value:
        calls.append("alpha")
        return Value(value="alpha allows 10" + (" supplemental" * 250 if paged else ""))

    async def beta(args: Empty, context: ToolContext) -> Value:
        calls.append("beta")
        return Value(value="beta allows 20")

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            nonlocal continuation_count
            assert request.current.task.goal == "Compare two public documents"
            view = request.current.task_model_view
            assert view is not None
            navigation = view.definition["navigation_context"]
            if navigation:
                assert isinstance(navigation, list)
                item = navigation[0]
                assert isinstance(item, dict)
                payload = item["payload"]
                assert isinstance(payload, dict)
                prior_needs = payload["prior_needs"]
                assert isinstance(prior_needs, list) and len(prior_needs) == 1
                need = prior_needs[0]
                assert isinstance(need, dict)
                if reconciliation:
                    assert need["reuse"] == "reusable"
                    evidence = need["evidence"]
                    assert isinstance(evidence, list) and len(evidence) == 1
                    assert isinstance(evidence[0], dict)
                    assert evidence[0]["json_pointer"] == "/value"
                else:
                    assert "reuse" not in need
                sources = payload["sources"]
                assert isinstance(sources, list)
                source = sources[0]
                assert isinstance(source, dict)
                assert len(sources) == 1
                assert source["content"] == (None if paged else {"value": "alpha allows 10"})
                reference = source["reference"]
                assert isinstance(reference, dict)
                feedback = request.current.feedback
                if feedback is not None:
                    if paged and feedback.calls[0].call_id == "inherited-page":
                        assert "alpha allows 10" in request.current.model_dump_json()
                        return ModelTurn(
                            decision=ToolBatch(
                                calls=(
                                    ToolCallAction(
                                        call_id="read", tool_name="read_beta", arguments={}
                                    ),
                                )
                            )
                        )
                    return ModelTurn(
                        decision=FinishAction(output=f"{reference['artifact_id']}: 10 vs 20")
                    )
                continuation_count += 1
                if continuation_count == 2:
                    both_reading.set()
                await both_reading.wait()
                if paged:
                    return ModelTurn(
                        decision=ToolBatch(
                            calls=(
                                ToolCallAction(
                                    call_id="inherited-page",
                                    tool_name="read_task_source",
                                    arguments={
                                        "artifact_id": reference["artifact_id"],
                                        "pointer": "/value",
                                        "length": 100,
                                    },
                                ),
                            )
                        )
                    )
                tool = "read_beta"
            else:
                feedback = request.current.feedback
                if feedback is not None:
                    if paged and feedback.calls[0].call_id == "read":
                        exchange = request.current.history[0]
                        assert isinstance(exchange, ToolExchange)
                        archive = exchange.observation
                        assert isinstance(archive, ToolSucceededObservation)
                        assert isinstance(archive.output, dict)
                        return ModelTurn(
                            decision=ToolBatch(
                                calls=(
                                    ToolCallAction(
                                        call_id="local-page",
                                        tool_name="read_artifact",
                                        arguments={
                                            "artifact_id": archive.output["artifact_id"],
                                            "pointer": "/value",
                                            "length": 100,
                                        },
                                    ),
                                )
                            )
                        )
                    return ModelTurn(decision=FinishAction(output="Alpha documented, beta pending"))
                tool = "read_alpha"
            return ModelTurn(
                decision=ToolBatch(
                    calls=(ToolCallAction(call_id="read", tool_name=tool, arguments={}),)
                )
            )

    registry = ToolRegistry(
        tuple(
            TypedTool(
                name=name,
                version="1",
                description=name,
                input_type=Empty,
                output_type=Value,
                execute=execute,
            )
            for name, execute in (("read_alpha", alpha), ("read_beta", beta))
        )
    )
    profile = HarnessConfig(
        configuration_id="task-continuation-test/1",
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        limits=RunLimits(max_request_bytes=64000),
    )
    reasoning = FakeModelProvider(
        tuple(interpretation(tool) for tool in ("read_alpha", "read_beta", "read_beta"))
    )
    access = Access({"a", "b", "c"})
    policy = SourcePolicy()
    async with AgentHarness.from_config(
        tmp_path / "task.db",
        config=profile,
        model=Model(),
        tools=registry,
        token_counter=session_counter,
        reasoning_model=reasoning,
        reasoning_counter=session_counter,
        policy=policy,
    ) as harness:
        for session in ("a", "b", "c"):
            await harness.create(session)
        initial = await harness.send("a", "Compare two public documents")
        task = await harness.register_task("compare", "a", initial.session.run_id, access=access)
        records = await harness.sessions.store.run_records("a", initial.session.run_id)
        assert len(records) == (2 if paged else 1), initial.session.report
        old = records[0]
        if paged:
            assert records[1].exchange.action.tool_name == "read_artifact"
            assert records[1].artifact is not None
        ref = old.artifact
        assert ref is not None
        denied = Access({"b"})
        with pytest.raises(PermissionError):
            await harness.continue_task(
                "compare",
                "b",
                expected_version=1,
                access=denied,
                evidence_reconciliation=reconciliation,
            )
        with pytest.raises(PermissionError):
            await harness.read_task_artifact(
                "compare", "b", ref.run_id, ref.artifact_id, access=denied
            )

        outcomes = await asyncio.wait_for(
            asyncio.gather(
                *(
                    harness.continue_task(
                        "compare",
                        session,
                        expected_version=task.version,
                        access=access,
                        evidence_reconciliation=reconciliation,
                    )
                    for session in ("b", "c")
                ),
                return_exceptions=True,
            ),
            timeout=10,
        )
        failures = [x for x in outcomes if isinstance(x, BaseException)]
        assert len(failures) == 1 and isinstance(failures[0], SessionConflictError), outcomes
        assert str(failures[0]) == "task_version_changed"
        (winner,) = (x for x in outcomes if not isinstance(x, BaseException))
        assert winner.task.version == 2
        assert winner.session.run_id != initial.session.run_id
        assert winner.session.report.output == f"{ref.artifact_id}: 10 vs 20"
        assert calls == ["alpha", "beta", "beta"]
        assert await harness.sessions.store.task_version("compare") == winner.task
        with pytest.raises(SessionConflictError, match="task_source_already_continued"):
            await harness.register_task(
                "copy", winner.session.session_id, winner.session.run_id, access=access
            )
        source = await harness.read_task_artifact(
            "compare",
            winner.session.session_id,
            ref.run_id,
            ref.artifact_id,
            access=access,
        )
        assert source.value == {
            "value": "alpha allows 10" + (" supplemental" * 250 if paged else "")
        }
        policy.revoked = True
        with pytest.raises(PermissionError, match="task_source_authorization_revoked"):
            await harness.read_task_artifact(
                "compare", winner.session.session_id, ref.run_id, ref.artifact_id, access=access
            )
        with pytest.raises(SessionConflictError, match="task_version_changed"):
            await harness.continue_task("compare", "b", expected_version=1, access=access)
