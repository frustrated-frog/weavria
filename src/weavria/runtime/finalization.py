"""Bounded synthesis readiness over the exact compiled evidence view."""

from dataclasses import dataclass
from typing import Literal, Protocol

from weavria.core import ControlRequest, EvidenceDeliveryManifest

from .completion import CompletionContext


@dataclass(frozen=True, slots=True)
class FinalizationAssessment:
    """Synthesis assessment; eligible means early readiness, not exhausted acquisition.

    ``mode`` describes evidence actually delivered to the candidate request. A limited
    view alone does not authorize Runtime to stop exploration before its stop boundary.
    """

    eligible: bool
    mode: Literal["complete", "limited", "unavailable"]
    required_call_ids: tuple[str, ...] = ()
    missing_need_ids: tuple[str, ...] = ()


class FinalizationPolicy(Protocol):
    @property
    def version(self) -> str: ...

    async def assess(
        self,
        context: CompletionContext,
        *,
        request: ControlRequest | None,
        manifest: EvidenceDeliveryManifest | None,
    ) -> FinalizationAssessment: ...
