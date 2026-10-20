"""Frozen semantic-frontier progress experiment with an exact-repeat baseline."""

import json

from pydantic import BaseModel, ConfigDict

from weavria.evaluation import ProgressAttempt, ProgressControlCase, evaluate_progress_control
from weavria.tools import ToolContext, ToolRegistry, TypedTool


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: str


async def _unused(arguments: _Input, context: ToolContext) -> _Output:
    raise AssertionError("the progress-control experiment cannot execute tools")


def _registry() -> ToolRegistry:
    return ToolRegistry(
        (
            TypedTool(
                name="read_source",
                version="1",
                description="Read a source for one required fact",
                input_type=_Input,
                output_type=_Output,
                execute=_unused,
            ),
        )
    )


def main() -> None:
    unchanged = "a" * 64
    report = evaluate_progress_control(
        (
            ProgressControlCase(
                case_id="volatile_argument_churn",
                goal="Read one stable fact",
                attempts=tuple(
                    ProgressAttempt(
                        need_state_sha256=unchanged,
                        tool_names=("read_source",),
                        arguments=({"query": f"variant-{index}"},),
                        expected_disposition="block" if index == 4 else "allow",
                    )
                    for index in range(1, 5)
                ),
            ),
            ProgressControlCase(
                case_id="effective_need_progress",
                goal="Collect four distinct required fields",
                attempts=tuple(
                    ProgressAttempt(
                        need_state_sha256=character * 64,
                        tool_names=("read_source",),
                        arguments=({"query": character},),
                        expected_disposition="allow",
                    )
                    for character in ("a", "b", "c", "d")
                ),
            ),
            ProgressControlCase(
                case_id="batch_shape_regrouping",
                goal="Bound one source family across batch shapes",
                attempts=(
                    ProgressAttempt(
                        need_state_sha256=unchanged,
                        tool_names=("read_source", "read_source", "read_source"),
                        arguments=tuple({"query": f"variant-{index}"} for index in range(3)),
                        expected_disposition="allow",
                    ),
                    ProgressAttempt(
                        need_state_sha256=unchanged,
                        tool_names=("read_source", "read_source"),
                        arguments=tuple({"query": f"variant-{index}"} for index in range(3, 5)),
                        expected_disposition="block",
                    ),
                ),
            ),
        ),
        _registry(),
    )
    payload = report.model_dump(mode="json")
    payload["scope"] = (
        "deterministic mechanism experiment; it neither executes tools nor claims "
        "real-model task-completion gains"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
