import asyncio

from pydantic import BaseModel, JsonValue

from weavria.artifacts import ArtifactSnapshot, InMemoryArtifactStore
from weavria.core import (
    ErrorCategory,
    FinishAction,
    ModelTurn,
    RunStatus,
    TaskInput,
    ToolBatch,
    ToolCallAction,
    ToolFailedObservation,
)
from weavria.models import FakeTurnProvider
from weavria.runtime import AgentLoop, ToolScheduling
from weavria.tools import ToolContext, ToolRegistry, TypedTool, artifact_read_tool


class Input(BaseModel):
    read: bool


class Output(BaseModel):
    text: str


async def test_sibling_artifact_cannot_be_read_even_after_it_was_published() -> None:
    published = asyncio.Event()
    identity: list[str] = []

    class Store(InMemoryArtifactStore):
        async def put(self, run_id: str, call_id: str, value: JsonValue) -> ArtifactSnapshot:
            snapshot = await super().put(run_id, call_id, value)
            if call_id == "source":
                identity.append(snapshot.reference.artifact_id)
                published.set()
            return snapshot

    async def read(arguments: Input, context: ToolContext) -> Output:
        if arguments.read:
            await published.wait()
            await context.read_artifact(identity[0])
        return Output(text="source")

    tool = TypedTool(
        name="read",
        version="1",
        description="Read",
        input_type=Input,
        output_type=Output,
        execute=read,
    )
    model = FakeTurnProvider(
        (
            ModelTurn(
                decision=ToolBatch(
                    calls=(
                        ToolCallAction(
                            call_id="source", tool_name="read", arguments={"read": False}
                        ),
                        ToolCallAction(
                            call_id="sibling", tool_name="read", arguments={"read": True}
                        ),
                    )
                )
            ),
            ModelTurn(decision=FinishAction(output="done")),
        )
    )
    agent = AgentLoop(
        run_id="run",
        task=TaskInput(goal="Read"),
        instructions="Read",
        model=model,
        tools=ToolRegistry((tool,)),
        artifacts=Store(),
        scheduling=ToolScheduling(2, frozenset({"read"})),
    )
    assert (await agent.execute()).status is RunStatus.COMPLETED
    failure = agent.state.records[1].exchange.observation
    assert isinstance(failure, ToolFailedObservation)
    assert failure.error.category is ErrorCategory.POLICY_DENIED
    assert not agent.state.records[1].dependencies


async def test_later_turn_can_read_accepted_source_and_context_remains_dependency_closed() -> None:
    from weavria.core import ModelRequest, ToolExchange, ToolSucceededObservation
    from weavria.models import ModelCapabilities

    async def read(arguments: Input, context: ToolContext) -> Output:
        return Output(text="authoritative result")

    tool = TypedTool(
        name="read",
        version="1",
        description="Read",
        input_type=Input,
        output_type=Output,
        execute=read,
    )

    class Recover:
        capabilities = ModelCapabilities(True, True)

        async def generate_turn(self, request: ModelRequest) -> ModelTurn:
            if not request.history:
                action = ToolCallAction(
                    call_id="source", tool_name="read", arguments={"read": False}
                )
                return ModelTurn(decision=ToolBatch(calls=(action,)))
            exchanges = [e for e in request.history if isinstance(e, ToolExchange)]
            if not any(e.action.call_id == "recover" for e in exchanges):
                entry = exchanges[0]
                assert isinstance(entry.observation, ToolSucceededObservation)
                output = entry.observation.output
                assert isinstance(output, dict)
                return ModelTurn(
                    decision=ToolBatch(
                        calls=(
                            ToolCallAction(
                                call_id="recover",
                                tool_name="read_artifact",
                                arguments={
                                    "artifact_id": output["artifact_id"],
                                    "pointer": "/text",
                                },
                            ),
                        )
                    )
                )
            return ModelTurn(decision=FinishAction(output="done"))

    agent = AgentLoop(
        run_id="run",
        task=TaskInput(goal="Read"),
        instructions="Read",
        model=Recover(),
        tools=ToolRegistry((tool, artifact_read_tool())),
        artifacts=InMemoryArtifactStore(),
    )
    assert (await agent.execute()).status is RunStatus.COMPLETED
    assert agent.state.records[1].dependencies == ("source",)
    assert all(d.selected for d in agent.plans[-1].decisions)
