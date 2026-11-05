"""Result snapshots and storage ports."""

from .attachments import read_owned_attachment
from .pointer import resolve_pointer
from .ranges import read_recovery_range, verify_recovery_value
from .store import ArtifactAccessError, ArtifactSnapshot, ArtifactStore, InMemoryArtifactStore

__all__ = [
    "ArtifactAccessError",
    "ArtifactSnapshot",
    "ArtifactStore",
    "InMemoryArtifactStore",
    "read_owned_attachment",
    "read_recovery_range",
    "resolve_pointer",
    "verify_recovery_value",
]
