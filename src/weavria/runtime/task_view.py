"""Host projection port; Runtime does not interpret Harness task schemas."""

from typing import Literal, Protocol

from weavria.core import SemanticSnapshot, TaskInput, TaskModelView


class TaskModelProjector(Protocol):
    """Keep definition, Work identity and capacity stable within one task revision.

    Progress, accepted-source versions and phase may change. A view is a model input,
    never an authority to execute tools or accept completion.
    """

    @property
    def version(self) -> str: ...

    def project(
        self,
        task: TaskInput,
        snapshot: SemanticSnapshot,
        *,
        phase: Literal["decision", "synthesis"],
        max_instruction_bytes: int,
    ) -> TaskModelView: ...
