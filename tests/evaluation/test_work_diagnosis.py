"""验证归因探针不伪造证据、不提升证明强度;明确记录现有未限定版本的缺口。"""

from pathlib import Path

import pytest

from examples.work_context_probe import full_source_view
from examples.work_recheck import load_dataset, registry
from examples.work_recheck_diagnosis import with_version_scope
from weavria.artifacts import InMemoryArtifactStore
from weavria.core import (
    ControlRequest,
    ConversationView,
    SessionRequest,
    TaskInput,
    ToolCallAction,
    ToolExchange,
    ToolSucceededObservation,
)
from weavria.harness.requirements import EvidenceResolver, InformationRequirement, NeedStatus
from weavria.tools import ReadOnlyPolicy, ToolExecutor


async def test_freshness_and_source_version_are_independent_diagnostic_axes() -> None:
    case = load_dataset(Path("tests/fixtures/work_recheck/tasks.json")).cases[1]
    store = InMemoryArtifactStore()
    executor = ToolExecutor("r", registry(case), store, ReadOnlyPolicy())
    records = tuple(
        [
            await executor.execute(
                ToolCallAction(
                    call_id=version, tool_name="read_source", arguments={"version": version}
                ),
                max_duration=1,
            )
            for version in ("original", "current")
        ]
    )
    requirement = InformationRequirement(
        need_id="current-document",
        description="Read the current document version",
        source="external",
        freshness="current",
        required_fields=("/text",),
        capability_names=("read_source",),
        origin_ref="request",
    )

    async def resolve(need: InformationRequirement, *, original_only: bool = False):
        return await EvidenceResolver().resolve(
            work_id="w",
            task_revision=1,
            frame_revision=1,
            requirements=(need,),
            records=records[:1] if original_only else records,
            read_artifact=lambda aid: store.get("r", aid),
        )

    actual = await resolve(requirement)
    assert actual.states[0].missing_fields == ("$freshness",)
    stable = requirement.model_copy(update={"freshness": "stable"})
    assert (await resolve(stable)).states[0].acquisition_complete
    # 当前无范围配置的行为记录,不是应推广的语义承诺:只读旧文档也会算已获取。
    unscoped = await resolve(stable, original_only=True)
    assert unscoped.states[0].status is NeedStatus.ACQUIRED
    scoped = with_version_scope(stable, "current")
    wrong = await resolve(scoped, original_only=True)
    assert wrong.bindings == () and not wrong.states[0].acquisition_complete
    correct = await resolve(scoped)
    assert {a.producer_call_id for b in correct.bindings for a in b.addresses} == {"current"}
    assert not correct.states[0].acquisition_complete  # 匹配版本不等于 scoped_semantic 证明。
    assert requirement.freshness == "current" and not requirement.evidence_match_plans


async def test_full_source_probe_only_changes_verified_delivered_material() -> None:
    case = load_dataset(Path("tests/fixtures/work_recheck/tasks.json")).cases[0]
    store = InMemoryArtifactStore()
    executor = ToolExecutor("r", registry(case), store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(
            call_id="original", tool_name="read_source", arguments={"version": "original"}
        ),
        max_duration=1,
    )
    assert record.artifact is not None
    receipt = record.artifact
    observation = ToolSucceededObservation(
        call_id="original",
        tool_name="read_source",
        output={
            "artifact_id": receipt.artifact_id,
            "sha256": receipt.sha256,
            "size_bytes": receipt.size_bytes,
            "representation": "excerpts",
            "passages": [{"pointer": "/text", "start": 0, "end": 10, "value": case.original[:10]}],
        },
    )
    request = SessionRequest(
        conversation=ConversationView(session_id="s", through_sequence=0),
        current=ControlRequest(
            instructions="unchanged instructions",
            task=TaskInput(goal="unchanged goal"),
            history=(ToolExchange(action=record.exchange.action, observation=observation),),
        ),
    )
    full = await full_source_view(request, case)
    assert full.conversation == request.conversation
    assert full.current.model_dump(exclude={"history"}) == request.current.model_dump(
        exclude={"history"}
    )
    exchange = full.current.history[0]
    assert isinstance(exchange, ToolExchange) and isinstance(
        exchange.observation, ToolSucceededObservation
    )
    assert exchange.action == record.exchange.action
    assert exchange.observation.output == {
        "artifact_id": receipt.artifact_id,
        "sha256": receipt.sha256,
        "size_bytes": receipt.size_bytes,
        "data": {"text": case.original, "version": "original"},
    }
    with pytest.raises(ValueError, match="digest"):
        await full_source_view(request, case.model_copy(update={"original": "tampered source"}))
