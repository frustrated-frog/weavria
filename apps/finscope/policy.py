"""Application-local discovery and execution policy for FinScope read capabilities."""

from pydantic import JsonValue

from weavria.harness import CapabilityDescriptor
from weavria.tools import PolicyDecision, ToolDescriptor

FINSCOPE_TOOL_NAMES = frozenset(
    {
        "read_market_pulse",
        "read_finscope_catalogue",
        "read_finscope_resource",
        "search_finscope",
    }
)
_FORBIDDEN_ARGUMENT_KEYS = frozenset({"method", "path", "url", "sql", "statement"})


class FinScopeReadOnlyPolicy:
    """Allow only the adapter's fixed, low-risk GET-backed capability set."""

    version = "finscope-read-only/1"

    def can_discover(self, descriptor: CapabilityDescriptor) -> bool:
        return (
            descriptor.name in FINSCOPE_TOOL_NAMES
            and descriptor.discoverable
            and descriptor.availability == "available"
            and _safe_metadata(descriptor)
        )

    async def authorize(
        self, run_id: str, descriptor: ToolDescriptor, arguments: JsonValue
    ) -> PolicyDecision:
        del run_id
        safe_arguments = not (
            isinstance(arguments, dict) and _FORBIDDEN_ARGUMENT_KEYS.intersection(arguments)
        )
        allowed = (
            descriptor.name in FINSCOPE_TOOL_NAMES and _safe_metadata(descriptor) and safe_arguments
        )
        if allowed:
            reason = "fixed FinScope read capability"
        elif not safe_arguments:
            reason = "caller-controlled methods and paths are forbidden"
        else:
            reason = "outside the FinScope read-only permission ceiling"
        return PolicyDecision(allowed, self.version, reason)


def _safe_metadata(descriptor: CapabilityDescriptor | ToolDescriptor) -> bool:
    return (
        descriptor.behavior == "read_only"
        and descriptor.idempotency == "replay_safe"
        and descriptor.risk == "low"
        and not descriptor.approval_required
        and descriptor.sensitivity in {"public", "private"}
    )
