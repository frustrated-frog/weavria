"""M1 的两个入口组合缺口。替换模型响应,保留真实理解适配器与会话生命周期。"""

import asyncio
import json
from pathlib import Path

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
from weavria.models import (
    FakeModelProvider,
    FakeSessionProvider,
    ModelCapabilities,
    OpenAICompatibleTokenCounter,
)
from weavria.runtime import RunLimits
from weavria.tools import ToolRegistry


def config() -> HarnessConfig:
    return HarnessConfig(
        configuration_id="m1-session-chain/1",
        context=ContextCompilerConfig(window_tokens=64000, output_tokens=256),
        limits=RunLimits(max_request_bytes=64000),
        working_records=True,
    )


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


async def test_recommended_entry_steering_invalidates_old_work_and_model_answer(
    tmp_path: Path,
    session_counter: OpenAICompatibleTokenCounter,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Model:
        capabilities = ModelCapabilities(tool_calls=True, multi_tool_calls=True)

        async def generate_session(self, request: SessionRequest) -> ModelTurn:
            view = request.current.task_model_view
            assert view is not None
            work = WorkingState.model_validate_json(json.dumps(view.progress["working_state"]))
            if request.current.task.revision > 1:
                assert work.task_revision == 2 and work.last_notice == "revised"
                assert work.notes == () and work.invalidated_note_ids == ("old-plan",)
                return ModelTurn(decision=FinishAction(output="Updated answer"))
            if work.revision == 0:
                update = WorkUpdate(
                    task_revision=1,
                    base_revision=0,
                    notes=(WorkNote(note_id="old-plan", kind="plan", summary="Initial plan"),),
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
        config=config(),
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
