"""真实准入闭环: 使用与工具名称无关的分页契约,拒绝后一次重提即可执行。"""

from dataclasses import replace

import pytest
from pydantic import BaseModel, Field

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    AdmissionNotice,
    ControlRequest,
    EvidenceAccessView,
    FeedbackBounds,
    FinishAction,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import FakeControlProvider
from weavria.runtime import AgentLoop, RunLimits
from weavria.runtime.feedback_budget import ReadFeedbackBudget
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class PageInput(BaseModel):
    size: int = Field(ge=1, le=4000)


class PageOutput(BaseModel):
    text: str


async def test_feedback_rejection_provides_a_feasible_single_page() -> None:
    executed: list[int] = []

    async def page(args: PageInput, context: ToolContext) -> PageOutput:
        executed.append(args.size)
        return PageOutput(text="x" * args.size)

    tool = TypedTool(
        name="load_page",
        version="1",
        description="Load a bounded page",
        input_type=PageInput,
        output_type=PageOutput,
        execute=page,
        feedback_delivery="complete",
        feedback_bound_version="page/1",
        feedback_data_bound=lambda args, bounds: args.size + 11,
        feedback_page_parameter="size",
    )

    class Model(FakeControlProvider):
        async def generate_turn(self, request: ControlRequest) -> ModelTurn:
            self._requests.append(request.model_dump_json())
            if len(self.requests) == 1:
                size = 4000
            elif isinstance(request.control, AdmissionNotice):
                assert not executed
                assert request.control.page_parameter == "size"
                assert request.control.page_call_index == 0
                assert request.control.page_maximum is not None
                assert 0 < request.control.page_maximum < 4000
                size = request.control.page_maximum
            else:
                return ModelTurn(decision=FinishAction(output="done"))
            return ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id=f"read-{len(self.requests)}",
                            tool_name="load_page",
                            arguments={"size": size},
                        ),
                    )
                )
            )

    provider = Model(())
    agent = AgentLoop(
        run_id="page",
        task=TaskInput(goal="Read a page"),
        instructions="Read",
        model=provider,
        artifacts=InMemoryArtifactStore(),
        tools=ToolRegistry((tool,)),
        limits=RunLimits(max_request_bytes=6000),
    )
    result = await agent.execute()
    assert result.status is RunStatus.COMPLETED
    assert len(executed) == 1
    assert len(provider.requests) == 3


def test_no_page_fits_but_another_offered_read_can_be_reproduced_from_the_notice() -> None:
    async def read(arguments: PageInput, context: ToolContext) -> PageOutput:
        return PageOutput(text="x" * arguments.size)

    source = TypedTool(
        name="source",
        version="1",
        description="Read a source",
        input_type=PageInput,
        output_type=PageOutput,
        execute=read,
        artifact_read_inputs=(),
    )
    page = TypedTool(
        name="page",
        version="1",
        description="Read a page",
        input_type=PageInput,
        output_type=PageOutput,
        execute=read,
        feedback_delivery="complete",
        feedback_bound_version="page/1",
        feedback_data_bound=lambda args, bounds: 3000 + args.size,
        feedback_page_parameter="size",
        artifact_read_inputs=(),
    )
    tools = ToolRegistry((source, page))
    access = EvidenceAccessView.from_records((), (), task_revision=1, request_sha256="f" * 64)
    budget = ReadFeedbackBudget(
        tools,
        "use sources",
        TaskInput(goal="read"),
        None,
        (),
        FeedbackBounds(),
        10000,
        access,
        None,
    )
    single = ToolCallAction(tool_name="source", call_id="source-1", arguments={"size": 20})
    big = ToolCallAction(tool_name="page", call_id="page-1", arguments={"size": 1000})
    batch = ToolBatch(calls=(big, single))
    measured, _ = budget.measure(ToolBatch(calls=(single,)), (tools.action_identity(single),))
    tight = replace(budget, budget_bytes=measured.required_bytes)
    visible = {"page", "source"}
    retry = tight.retry(batch, visible)
    assert retry is not None and retry.feasible and retry.call == single
    too_small = replace(tight, budget_bytes=measured.required_bytes - 1)
    blocked = too_small.retry(batch, visible)
    assert blocked is not None and not blocked.feasible and blocked.page == (0, "size", 0)
    hidden = tight.retry(batch, {"page"})
    assert hidden is not None and not hidden.feasible
    # v7 将可原样重提的参数随通知送达,模型无需记住上次被拒的提案。
    notice = AdmissionNotice(
        schema_version=7,
        reason="feedback_budget",
        proposal_sha256="a" * 64,
        proposed_calls=2,
        required_bytes=10001,
        budget_bytes=10000,
        remaining_tool_calls=2,
        recommended_max_calls=1,
        retry_read=retry.call,
    )
    assert AdmissionNotice.model_validate_json(notice.model_dump_json()).retry_read == single
    with pytest.raises(ValueError, match="singleton"):
        AdmissionNotice.model_validate(notice.model_copy(update={"retry_read": None}))
    with pytest.raises(ValueError, match="v7"):
        AdmissionNotice.model_validate(notice.model_copy(update={"schema_version": 1}))
