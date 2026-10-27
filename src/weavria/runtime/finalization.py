"""Bounded synthesis readiness over the exact compiled evidence view."""

from dataclasses import dataclass
from typing import Literal, Protocol

from weavria.core import ControlRequest, EvidenceDeliveryManifest

from .completion import CompletionContext


@dataclass(frozen=True, slots=True)
class FinalizationAssessment:
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
