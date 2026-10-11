"""Inspect the contracts without a model, tool execution, or an Agent Loop."""

from weavria.core import (
    AgentRequest,
    AgentResult,
    FinishAction,
    Message,
    MessageRole,
    RunStatus,
    ToolCallAction,
    ToolSucceededObservation,
    parse_action,
    transition_status,
)


def main() -> None:
    """Walk through a manually supplied add-tool decision and result."""
    request = AgentRequest(messages=(Message(role=MessageRole.USER, content="计算 2 + 3"),))
    print("Request:", request.model_dump_json())
    status = transition_status(RunStatus.CREATED, RunStatus.RUNNING)

    action = parse_action(
        {
            "kind": "tool_call",
            "call_id": "call-1",
            "tool_name": "add",
            "arguments": {"left": 2, "right": 3},
        }
    )
    assert isinstance(action, ToolCallAction)
    print("Validated action:", action.model_dump_json())

    # This is supplied demonstration data, not an executed or validated tool result.
    observation = ToolSucceededObservation(
        call_id=action.call_id, tool_name=action.tool_name, output={"value": 5}
    )
    print("Observation:", observation.model_dump_json())

    finish = parse_action({"kind": "finish", "output": "2 + 3 = 5"})
    assert isinstance(finish, FinishAction)
    status = transition_status(status, RunStatus.COMPLETED)
    assert status is RunStatus.COMPLETED
    result = AgentResult(run_id="run-1", status=status, output=finish.output)
    print("Result:", result.model_dump_json())


if __name__ == "__main__":
    main()
