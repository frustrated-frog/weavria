"""真实准入闭环: 使用与工具名称无关的分页契约,拒绝后一次重提即可执行。"""

from pydantic import BaseModel, Field

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    AdmissionNotice,
    ControlRequest,
    FinishAction,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
)
from weavria.models import FakeControlProvider
from weavria.runtime import AgentLoop, RunLimits
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
