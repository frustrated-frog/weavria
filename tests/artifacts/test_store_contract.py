import math

import pytest
from pydantic import JsonValue

from weavria.artifacts import ArtifactAccessError, ArtifactSnapshot, InMemoryArtifactStore


async def test_round_trip_identity_is_stable_and_nested_values_are_snapshots() -> None:
    store = InMemoryArtifactStore()
    value: dict[str, JsonValue] = {"source": "资料", "nested": [1, 2]}
    first = await store.put("run", "call", value)
    second = await store.put("run", "call", value)
    assert first == second
    value["nested"] = [99]
    assert first.value == {"source": "资料", "nested": [1, 2]}
    returned = first.value
    assert isinstance(returned, dict)
    returned["source"] = "changed"
    assert (await store.get("run", first.reference.artifact_id)).value != returned
    assert "资料" not in repr(first)


async def test_run_and_producing_call_are_part_of_identity_and_access_scope() -> None:
    store = InMemoryArtifactStore()
    first = await store.put("a", "call", "same")
    other = await store.put("b", "call", "same")
    third = await store.put("a", "different-call", "same")
    assert (
        len({first.reference.artifact_id, other.reference.artifact_id, third.reference.artifact_id})
        == 3
    )
    with pytest.raises(ArtifactAccessError):
        await store.get("b", first.reference.artifact_id)


async def test_failed_oversized_put_does_not_publish_a_result() -> None:
    store = InMemoryArtifactStore(max_result_bytes=4)
    with pytest.raises(ArtifactAccessError):
        await store.put("run", "call", "中文")
    small = await store.put("run", "call", 1)
    assert small.value == 1


async def test_tampered_snapshot_is_rejected() -> None:
    first = await InMemoryArtifactStore().put("run", "call", {"answer": 5})
    with pytest.raises(ArtifactAccessError):
        ArtifactSnapshot(first.reference_json, '{"answer":6}')


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
async def test_nonfinite_json_is_not_silently_changed_to_null(value: float) -> None:
    with pytest.raises(ValueError):
        await InMemoryArtifactStore().put("run", "call", value)
