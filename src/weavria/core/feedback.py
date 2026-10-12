"""Minimum observation projections and deterministic upper bounds; no execution facts created."""

from .actions import ToolCallAction
from .control import FeedbackBounds
from .errors import ErrorCategory, RunError
from .execution import ToolRecord
from .model_io import ToolExchange
from .outcomes import ToolFailedObservation, ToolSucceededObservation


def minimum_exchange(record: ToolRecord) -> ToolExchange:
    """Keep identity, failure category and recoverable source; full record stays authoritative."""
    action = record.exchange.action
    if record.artifact is not None:
        source = record.artifact
        return ToolExchange(
            action=action,
            observation=ToolSucceededObservation(
                call_id=action.call_id,
                tool_name=action.tool_name,
                output={
                    "artifact_id": source.artifact_id,
                    "sha256": source.sha256,
                    "size_bytes": source.size_bytes,
                },
            ),
        )
    observation = record.exchange.observation
    assert isinstance(observation, ToolFailedObservation)
    return ToolExchange(
        action=action,
        observation=ToolFailedObservation(
            call_id=action.call_id,
            tool_name=action.tool_name,
            error=RunError(
                category=observation.error.category,
                message=observation.error.category.value,
                retryable=observation.error.retryable,
            ),
        ),
    )


def maximum_exchange_bytes(action: ToolCallAction, bounds: FeedbackBounds) -> int:
    """Bound only the published minimum projection, not backend result or policy text.

    Identity JSON bytes and source size are enforced by the executor. Failure
    messages are projected to stable categories. This is a sizing calculation;
    none of the templates are recorded or delivered as observations.
    """
    success = ToolSucceededObservation(
        call_id=action.call_id,
        tool_name=action.tool_name,
        output={
            "artifact_id": "x" * (bounds.max_artifact_id_json_bytes - 2),
            "sha256": "0" * 64,
            "size_bytes": bounds.max_result_bytes,
        },
    )
    sizes = [len(success.model_dump_json().encode())]
    for category in ErrorCategory:
        failure = ToolFailedObservation(
            call_id=action.call_id,
            tool_name=action.tool_name,
            error=RunError(category=category, message=category.value, retryable=False),
        )
        sizes.append(len(failure.model_dump_json().encode()))
    # Wire field names are fixed by ToolExchange v1; values are measured with
    # the same Pydantic encoder as the final request, including Unicode escapes.
    wrapper = len(b'{"schema_version":1,"kind":"tool_exchange","action":,"observation":}')
    return wrapper + len(action.model_dump_json().encode()) + max(sizes)
