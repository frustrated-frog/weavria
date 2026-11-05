"""附件的身份验证;父结果的授权由调用边界负责,本模块不以 ID 授权。"""

import json
from collections.abc import Awaitable, Callable

from weavria.core import ArtifactRef

from .pointer import resolve_pointer
from .store import ArtifactAccessError, ArtifactSnapshot


async def read_owned_attachment(
    parent: ArtifactSnapshot,
    pointer: str,
    read: Callable[[str, str], Awaitable[ArtifactSnapshot]],
) -> ArtifactSnapshot:
    """Read an exact child of an already-authorized parent, requiring the same producer."""
    try:
        reference = ArtifactRef.model_validate_json(
            json.dumps(resolve_pointer(parent.value, pointer))
        )
    except (ValueError, KeyError, IndexError) as exc:
        raise ArtifactAccessError("invalid attachment address") from exc
    if (
        reference.run_id != parent.reference.run_id
        or reference.call_id != parent.reference.call_id
        or reference.artifact_id == parent.reference.artifact_id
    ):
        raise ArtifactAccessError("attachment belongs to another producer")
    stored = await read(reference.run_id, reference.artifact_id)
    snapshot = ArtifactSnapshot(stored.reference_json, stored.content_json)
    if snapshot.reference != reference:
        raise ArtifactAccessError("attachment identity differs from its accepted parent")
    return snapshot
