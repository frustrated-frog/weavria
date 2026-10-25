"""Deterministic local memory evidence, separate from model capability evaluation."""

from datetime import UTC, datetime

import pytest

from weavria.memory import MemoryAccess, MemoryAssertion, MemoryScope, MemorySource


@pytest.fixture
def scope() -> MemoryScope:
    return MemoryScope(tenant_id="tenant", kind="project", key="project")


@pytest.fixture
def access(scope: MemoryScope) -> MemoryAccess:
    return MemoryAccess(principal_id="host", readable_scopes=(scope,), writable_scopes=(scope,))


@pytest.fixture
def source(scope: MemoryScope) -> MemorySource:
    return MemorySource(
        source_id="input-1",
        scope=scope,
        reference="session-one/event-1",
        content_sha256="a" * 64,
        origin="user",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def assertion(scope: MemoryScope, source: MemorySource) -> MemoryAssertion:
    return MemoryAssertion(
        memory_id="preference-1",
        scope=scope,
        subject="user",
        predicate="language",
        value="中文",
        text="用户偏好中文回复",
        source_ids=(source.source_id,),
        authority="self-report",
        confidence=1.0,
        epistemic_status="stated",
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
