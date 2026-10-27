"""FinScope workspace result-semantics regression tests."""

import pytest
from pydantic import JsonValue

from apps.finscope.completion import FinScopeWorkspaceEvidenceExtractor
from apps.finscope.workspace import FinScopeReadOutput
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    ToolAuthorization,
    ToolCallAction,
    ToolExchange,
    ToolRecord,
    ToolSucceededObservation,
)
from weavria.harness import InformationRequirement


async def _evidence(
    *,
    tool_name: str,
    arguments: dict[str, JsonValue],
    data: JsonValue,
):
    call_id = "workspace-read"
    store = InMemoryArtifactStore()
    snapshot = await store.put(
        "run",
        call_id,
        FinScopeReadOutput(
            source_id="finscope:test:scope",
            resource_type="test",
            endpoint="/api/test",
            data=data,
        ).model_dump(mode="json"),
    )
    action = ToolCallAction(call_id=call_id, tool_name=tool_name, arguments=arguments)
    authorization = ToolAuthorization(
        tool_version="1",
        policy_version="test-policy/1",
        allowed=True,
        reason="test read",
        normalized_arguments=arguments,
    )
    record = ToolRecord(
        exchange=ToolExchange(
            action=action,
            observation=ToolSucceededObservation(
                call_id=call_id,
                tool_name=tool_name,
                output={
                    "artifact_id": snapshot.reference.artifact_id,
                    "sha256": snapshot.reference.sha256,
                    "size_bytes": snapshot.reference.size_bytes,
                    "policy_version": authorization.policy_version,
                },
            ),
        ),
        artifact=snapshot.reference,
        authorization=authorization,
    )
    requirement = InformationRequirement(
        need_id="workspace-finding",
        description="Report the bounded workspace finding",
        source="external",
        freshness="current",
        required_fields=("/data",),
        capability_names=(tool_name,),
        completion_mode="report_findings",
        minimum_proof="semantic",
        origin_ref="test-origin",
    )
    return await FinScopeWorkspaceEvidenceExtractor().evaluate(
        requirement,
        record,
        snapshot,
    )


@pytest.mark.parametrize(
    "data",
    (
        [],
        {"items": [], "page": 0, "size": 20, "total": 0},
    ),
)
async def test_empty_first_page_is_complete_negative_evidence(data: JsonValue) -> None:
    evidence = await _evidence(
        tool_name="search_finscope",
        arguments={"scope": "companies", "query": "missing", "page": 0, "limit": 20},
        data=data,
    )

    assert evidence.result_kind == "empty"
    assert evidence.scope_completeness == "complete"


async def test_nonempty_truncated_first_page_remains_partial() -> None:
    evidence = await _evidence(
        tool_name="search_finscope",
        arguments={"scope": "news_window", "query": "company", "page": 0, "limit": 1},
        data={"items": [{"id": 1}], "page": 0, "size": 1, "total": 3},
    )

    assert evidence.result_kind == "found"
    assert evidence.scope_completeness == "partial"


async def test_isolated_later_empty_page_does_not_claim_complete_scope() -> None:
    evidence = await _evidence(
        tool_name="search_finscope",
        arguments={"scope": "news_window", "query": "company", "page": 2, "limit": 20},
        data={"items": [], "page": 2, "size": 20, "total": 3},
    )

    assert evidence.result_kind == "empty"
    assert evidence.scope_completeness == "partial"


async def test_missing_fixed_resource_is_complete_not_found_evidence() -> None:
    evidence = await _evidence(
        tool_name="read_finscope_resource",
        arguments={"kind": "article", "identifier": "404"},
        data=None,
    )

    assert evidence.result_kind == "not_found"
    assert evidence.scope_completeness == "complete"
