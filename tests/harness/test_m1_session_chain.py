"""M1 的两个入口组合缺口。替换模型响应,保留真实理解适配器与会话生命周期。"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from weavria.artifacts import InMemoryArtifactStore
from weavria.context import ContextCompilerConfig
from weavria.core import (
    FinishAction,
    ModelResponse,
    ModelTurn,
    RunStatus,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
)
from weavria.harness import (
    AgentHarness,
    ClarificationAnswerOption,
    ClarificationQuestionCandidate,
    ClarificationResolution,
    CompletionCriterion,
    HarnessConfig,
    IntentHypothesis,
    IntentUncertainty,
    TaskSpecification,
    WorkingState,
    WorkNote,
    WorkUpdate,
)
from weavria.harness.work import WorkProposal, work_tool
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCapabilities,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


def config() -> HarnessConfig:
    return HarnessConfig(
        configuration_id="m1-session-chain/1",
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        limits=RunLimits(max_request_bytes=64000),
        working_records=True,
    )


def test_recommended_entry_requires_progress_projection_with_adaptive_capabilities() -> None:
    base = replace(config(), working_records=False)
    with pytest.raises(ValueError, match="adaptive capabilities require task model projection"):
        replace(base, project_task_view=False)
    baseline = replace(base, project_task_view=False, adaptive_capabilities=False)
    assert not baseline.project_task_view


def declaration(specification: TaskSpecification) -> ModelResponse:
    return ModelResponse(
        action=ToolCallAction(
            call_id="interpret",
            tool_name="declare_task_specification",
            arguments=specification.model_dump(mode="json"),
        )
    )


def direct() -> TaskSpecification:
    return TaskSpecification(
        route="direct",
        intent="Answer",
        goal="Explain a principle",
        completion_criteria=(CompletionCriterion(kind="response", description="Answer"),),
    )


async def test_recommended_entry_clarifies_and_resumes_persisted_work(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    question = "Which language should I use?"
    ambiguity = "Output language is unspecified"
    clarify = TaskSpecification(
        route="clarify",
        intent="Resolve language",
        goal="Ask for language",
        ambiguities=(ambiguity,),
        uncertainty=IntentUncertainty(
            hypotheses=tuple(
                IntentHypothesis(
                    hypothesis_id=n,
                    interpretation=f"Answer in {n}",
                    intended_route="direct",
                    outcome_key=n,
                    support_weight=50,
                )
                for n in ("chinese", "english")
            ),
            candidate_questions=(
                ClarificationQuestionCandidate(
                    question_id="language",
                    question_text=question,
                    target_ambiguities=(ambiguity,),
                    answer_options=tuple(
                        ClarificationAnswerOption(
                            answer_id=n,
                            label=n,
                            hypothesis_ids=(n,),
                        )
                        for n in ("chinese", "english")
                    ),
                ),
            ),
        ),
        completion_criteria=(
            CompletionCriterion(kind="clarification", description="Ask language"),
        ),
    )
    resolved = direct().model_copy(
        update={
            "clarification_resolution": ClarificationResolution(
                question_id="language", answer_id="chinese"
            ),
            "resolved_outcome_key": "chinese",
        }
    )
    reasoning = FakeModelProvider((declaration(clarify), declaration(resolved)))
    model = FakeSessionProvider(
        (
            ModelTurn(decision=FinishAction(output=question)),
            ModelTurn(decision=FinishAction(output="中文说明")),
        )
    )
    path = tmp_path / "session.db"
    async with AgentHarness.from_config(
        path,
        config=config(),
        model=model,
        tools=ToolRegistry(()),
        token_counter=session_counter,
        reasoning_model=reasoning,
        reasoning_counter=session_counter,
    ) as agent:
        await agent.create("s")
        first = await agent.send("s", "Explain a principle; first ask which language", input_id="a")
        assert first.session.report.output == question
        pending = await agent.sessions.store.pending_clarifications("s")
        assert len(pending) == 1 and pending[0].pending_clarification is not None
        work_id = pending[0].work_id
        clarification_id = pending[0].pending_clarification.clarification_id
    # 关闭并重开所有存储后续接;不能依赖仍在内存里的准备对象。
    async with AgentHarness.from_config(
        path,
        config=config(),
        model=model,
        tools=ToolRegistry(()),
        token_counter=session_counter,
        reasoning_model=reasoning,
        reasoning_counter=session_counter,
    ) as agent:
        result = await agent.send("s", "Chinese", input_id="b", clarification_id=clarification_id)
        head = await agent.sessions.store.semantic_head("s", work_id)
        assert result.session.report.status is RunStatus.COMPLETED
        assert result.session.report.output == "中文说明"
        assert head is not None and head.work_id == work_id and head.work_status == "succeeded"
        assert head.version.task_revision == 2
        assert not await agent.sessions.store.pending_clarifications("s")
        repeated = await agent.send("s", "Chinese", input_id="b", clarification_id=clarification_id)
        assert repeated.session == result.session and len(model.requests) == 2
    assert len(reasoning.requests) == 2
    assert all(r.current.task_model_view is not None for r in model.requests)


@pytest.mark.parametrize("compact", [False, True])
async def test_recommended_entry_steering_invalidates_old_work_and_model_answer(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
    compact: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    notes = (
        WorkNote(note_id="old-plan", kind="plan", summary="Check the source constraints."),
        WorkNote(note_id="finding", kind="result", summary="The source restricts the first case."),
        WorkNote(note_id="rejected", kind="rejected", summary="The broader reading conflicts."),
        WorkNote(note_id="issue", kind="issue", summary="Verify whether the second case applies."),
        WorkNote(note_id="hypothesis", kind="hypothesis", summary="The rule could be local."),
    )
    profile = config()
    profile = replace(profile, context=replace(profile.context, work_summary_compaction=compact))

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            view = request.current.task_model_view
            assert view is not None
            payload = view.progress["working_state"]
            assert isinstance(payload, dict)
            if payload.get("schema_version") == 3:
                assert compact
                assert payload["task_revision"] == 1 and payload["revision"] == 1
                assert payload["notes"] == [
                    note.model_dump(mode="json", exclude={"schema_version"}, exclude_defaults=True)
                    for note in notes
                ]
                assert payload["next_query"] == "Verify the second case"
                # 同一完整模型请求只替换工作表示, 用生产计量器比较而非用字符猜 Token。
                full = WorkingState(
                    task_revision=1,
                    revision=1,
                    processed_records=1,
                    notes=notes,
                    next_query="Verify the second case",
                    last_notice="accepted",
                )
                baseline_view = view.model_copy(
                    update={
                        "progress": {**view.progress, "working_state": full.model_dump(mode="json")}
                    }
                )
                baseline = request.model_copy(
                    update={
                        "current": request.current.model_copy(
                            update={"task_model_view": baseline_view}
                        )
                    }
                )
                assert len(request.model_dump_json().encode()) < len(
                    baseline.model_dump_json().encode()
                )
                assert session_counter.count(request) < session_counter.count(baseline)
                # 直接验收工具的版本绑定, 不要求已经建议收尾的 Runtime 再跑一个工作轮。
                store = InMemoryArtifactStore()
                executor = ToolExecutor(
                    "binding", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy()
                )
                record = await executor.execute(
                    ToolCallAction(
                        call_id="update",
                        tool_name="propose_work",
                        arguments={"next_query": "Read appendix"},
                    ),
                    max_duration=1,
                    task_view=view,
                )
                assert record.artifact is not None
                saved = await store.get("binding", record.artifact.artifact_id)
                proposal = WorkProposal.model_validate_json(saved.content_json)
                assert (proposal.update.task_revision, proposal.update.base_revision) == (1, 1)
            else:
                work = WorkingState.model_validate_json(json.dumps(payload))
                if request.current.task.revision > 1:
                    assert work.task_revision == 2 and work.last_notice == "revised"
                    assert work.notes == () and work.invalidated_note_ids == tuple(
                        n.note_id for n in notes
                    )
                    return ModelTurn(decision=FinishAction(output="Updated answer"))
                if work.revision:
                    assert not compact
                    assert work.notes == notes
                    assert work.next_query == "Verify the second case"

            if payload["revision"] == 0:
                update = WorkUpdate(
                    task_revision=1,
                    base_revision=0,
                    notes=notes,
                    next_query="Verify the second case",
                )
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="note",
                                tool_name="propose_work",
                                arguments=update.model_dump(mode="json"),
                            ),
                        )
                    )
                )
            entered.set()
            await release.wait()
            return ModelTurn(decision=FinishAction(output="Stale answer"))

    reasoning = FakeModelProvider((declaration(direct()), declaration(direct())))
    async with AgentHarness.from_config(
        tmp_path / "session.db",
        config=profile,
        model=Model(),
        tools=ToolRegistry(()),
        token_counter=session_counter,
        reasoning_model=reasoning,
        reasoning_counter=session_counter,
    ) as agent:
        await agent.create("s")
        task = asyncio.create_task(agent.send("s", "Explain a principle", input_id="a"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await agent.steer("s", "Instead explain another principle", input_id="change")
            release.set()
            result = await asyncio.wait_for(task, 5)
        finally:
            # 断言失败也不能把测试拥有的 send 任务留给事件循环销毁。
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert result.session.report.status is RunStatus.COMPLETED, result.session.report
        assert result.session.report.output == "Updated answer"
    assert len(reasoning.requests) == 2 and reasoning.requests[-1].task.revision == 2
