"""Local persistent conversation driver; deterministic interpreter, not an LLM agent."""

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from weavria.context import EvidenceMode
from weavria.core import (
    ConversationItem,
    FinishAction,
    ModelTurn,
    SessionRequest,
    ToolBatch,
    ToolCallAction,
    ToolExchange,
    ToolFailedObservation,
)
from weavria.models import ModelCapabilities
from weavria.sessions import SessionEngine
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class FileInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str


class FileOutput(BaseModel):
    document: JsonValue


def source_tool(root: Path) -> TypedTool[FileInput, FileOutput]:
    root = root.resolve()

    async def read(arguments: FileInput, context: ToolContext) -> FileOutput:
        path = (root / arguments.name).resolve()
        if not path.is_relative_to(root) or path.stat().st_size > 32000:
            raise ValueError("fixture is outside the bounded source directory")
        return FileOutput(
            document=TypeAdapter[JsonValue](JsonValue).validate_json(path.read_bytes())
        )

    return TypedTool(
        name="read_file",
        version="1",
        description="Read a local JSON fixture",
        input_type=FileInput,
        output_type=FileOutput,
        execute=read,
    )


class ConversationInterpreter:
    """Resolve a tiny command language from actual observations; no expected answers supplied."""

    capabilities = ModelCapabilities(True, True)

    def __init__(self) -> None:
        self._next_call = 0

    async def generate_session(self, request: SessionRequest) -> ModelTurn:
        await asyncio.sleep(0)
        current = [e for e in request.current.history if isinstance(e, ToolExchange)]
        if current:
            observation = current[-1].observation
            if isinstance(observation, ToolFailedObservation):
                return self.finish(json.dumps({"failed": observation.error.category.value}))
            output = observation.output
            assert isinstance(output, dict)
            data = output.get("data")
            if current[-1].action.tool_name == "read_session_history" and isinstance(data, dict):
                data = data.get("data")
            if request.current.task.goal.startswith("recall ") and isinstance(data, dict):
                keyword = request.current.task.goal.removeprefix("recall ")
                if isinstance(data.get("text"), str):
                    return self.finish(str(data["text"]))
                cards = data.get("turns")
                if isinstance(cards, list):
                    items = tuple(
                        ConversationItem.model_validate_json(json.dumps(c)) for c in cards
                    )
                    match = self._recall(items, keyword)
                    if match is not None:
                        return match
                before = data.get("next_before_ordinal")
                if isinstance(before, int) and before > 1:
                    return self.call("read_session_history", {"before_ordinal": before})
                return self.finish("No matching historical answer")

            if isinstance(data, dict) and "text" in data and request.current.task.goal == "second":
                answer = json.loads(str(data["text"]))
                return self.finish(json.dumps(answer["choices"][1], ensure_ascii=False))
            return self.finish(
                json.dumps(data if data is not None else output, ensure_ascii=False, sort_keys=True)
            )
        goal = request.current.task.goal
        if goal.startswith("options "):
            choices = TypeAdapter(list[str]).validate_json(goal.removeprefix("options "))
            return self.finish(json.dumps({"choices": choices}, ensure_ascii=False))
        if goal == "second":
            prior = request.conversation.turns[-1]
            answer = prior.assistant
            if answer is None:
                return self.finish("No previous accepted answer")
            if not answer.complete:
                return self.call(
                    "read_session_history",
                    {"run_id": prior.run_id, "field": "assistant", "length": 4000},
                )
            data = json.loads(answer.text)
            return self.finish(json.dumps(data["choices"][1], ensure_ascii=False))
        if goal.startswith("recall "):
            found = self._recall(request.conversation.turns, goal.removeprefix("recall "))
            return found if found is not None else self.call("read_session_history", {})
        if goal.startswith("read "):
            return self.call("read_file", {"name": goal.removeprefix("read ")})
        if goal == "reuse":
            source = next(
                c.artifact
                for t in reversed(request.conversation.turns)
                for c in reversed(t.calls)
                if c.artifact is not None
            )
            return self.call("read_session_source", {"source": source.model_dump(mode="json")})
        return self.finish("Supported: options [JSON strings], second, read <fixture.json>, reuse")

    def _recall(self, items: tuple[ConversationItem, ...], keyword: str) -> ModelTurn | None:
        for item in items:
            if keyword in item.user.text and item.assistant is not None:
                if item.assistant.complete:
                    return self.finish(item.assistant.text)
                return self.call("read_session_history", {"run_id": item.run_id})
        return None

    def call(self, name: str, arguments: dict[str, JsonValue]) -> ModelTurn:
        self._next_call += 1
        return ModelTurn(
            decision=ToolBatch(
                calls=(
                    ToolCallAction(
                        call_id=f"read-{self._next_call}", tool_name=name, arguments=arguments
                    ),
                )
            )
        )

    @staticmethod
    def finish(output: str) -> ModelTurn:
        return ModelTurn(decision=FinishAction(output=output))


async def execute(args: argparse.Namespace) -> None:
    path = Path(str(args.db))
    root = Path(__file__).parent / "data" / "loop" / "clean"
    async with SessionEngine(
        path,
        model=ConversationInterpreter(),
        tools=ToolRegistry((source_tool(root),)),
        mode=EvidenceMode.FULL,
    ) as engine:
        session = str(args.session)
        if args.create:
            await engine.create(session)
        result = await engine.send(session, str(args.prompt), input_id=args.input_id)
        print(result.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--session", default="demo")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--input-id")
    parser.add_argument("--prompt", required=True)
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
