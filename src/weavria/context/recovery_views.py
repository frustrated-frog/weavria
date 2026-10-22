"""Verify that each accepted lease is actually readable in the model request."""

from pydantic import JsonValue

from weavria.artifacts import ArtifactAccessError, resolve_pointer, verify_recovery_value
from weavria.core import ControlRequest, RecoveryRange, ToolExchange, ToolSucceededObservation

from .selection import ContextBuildError


def validate_recovery_view(request: ControlRequest, ranges: tuple[RecoveryRange, ...]) -> None:
    """Reject omitted, reference-only, altered or shortened protected originals."""
    exchanges = {e.action.call_id: e for e in request.history if isinstance(e, ToolExchange)}
    for part in ranges:
        exchange = exchanges.get(part.call_id)
        output = (
            exchange.observation.output
            if exchange and isinstance(exchange.observation, ToolSucceededObservation)
            else None
        )
        if not isinstance(output, dict) or output.get("sha256") != part.source_sha256:
            raise ContextBuildError("recovery view does not bind its accepted source")
        try:
            if output.get("representation") == "full" and "data" in output:
                verify_recovery_value(resolve_pointer(output["data"], part.pointer), part)
                continue
            if output.get("representation") == "excerpts" and _contains(
                output.get("passages"), part
            ):
                continue
        except (ArtifactAccessError, KeyError, IndexError, ValueError) as exc:
            raise ContextBuildError("recovery view differs from its original range") from exc
        raise ContextBuildError("recovery view omitted a protected original range")


def _contains(passages: JsonValue, part: RecoveryRange) -> bool:
    if not isinstance(passages, list):
        return False
    for item in passages:
        if not isinstance(item, dict) or item.get("pointer") != part.pointer or "value" not in item:
            continue
        start, end = item.get("start"), item.get("end")
        if type(start) is not int or type(end) is not int:
            continue
        value = item["value"]
        if part.kind == "text":
            if (
                not isinstance(value, str)
                or len(value) != end - start
                or not start <= part.start < part.end <= end
            ):
                continue
            local = part.model_copy(update={"start": part.start - start, "end": part.end - start})
            verify_recovery_value(value, local)
        else:
            if start or end:
                continue
            verify_recovery_value(value, part)
        return True
    return False
