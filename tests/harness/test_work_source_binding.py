"""原文选择器走真实工具授权和 Artifact,不以模型摘要代替定位依据。"""

import pytest
from pydantic import JsonValue

from weavria.artifacts import InMemoryArtifactStore
from weavria.core import ErrorCategory, ToolCallAction, ToolFailedObservation
from weavria.harness.work import WorkProposal, work_tool
from weavria.tools import ReadOnlyPolicy, ToolExecutor, ToolRegistry


@pytest.mark.parametrize("selector", ["quote", "range"])
async def test_unicode_quote_and_legacy_range_bind_the_same_original_source(selector: str) -> None:
    store = InMemoryArtifactStore()
    text = "===== 数据.md =====\r\n金额🙂为20元。\r\n结尾"
    source = await store.put("r", "read", {"text": text})
    phrase = "金额🙂为20元。"
    start = text.index(phrase)
    ref: dict[str, JsonValue] = {"artifact_id": source.reference.artifact_id, "pointer": "/text"}
    ref.update(
        {"quote": phrase} if selector == "quote" else {"start": start, "end": start + len(phrase)}
    )
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(
            call_id="record",
            tool_name="propose_work",
            arguments={
                "task_revision": 1,
                "base_revision": 0,
                "notes": [{"kind": "result", "summary": "金额20元", "sources": [ref]}],
            },
        ),
        max_duration=1,
        accepted_artifacts=(source.reference,),
    )
    assert record.artifact is not None
    proposal = WorkProposal.model_validate_json(
        (await store.get("r", record.artifact.artifact_id)).content_json
    )
    bound = proposal.update.notes[0].sources[0]
    assert (bound.start, bound.end) == (start, start + len(phrase))
    assert text[bound.start : bound.end] == phrase
    assert proposal.sources[0].source_label == "数据.md"
    assert proposal.sources[0].artifact == source.reference
    assert record.dependencies == ("read",)
    assert "quote" not in bound.model_dump()  # 输入便利格式不进入持久化引用契约。
    # 换成规范坐标后再次提交同一内容,自动 ID 不能因输入写法不同而变化。
    canonical = await executor.execute(
        ToolCallAction(
            call_id="canonical",
            tool_name="propose_work",
            arguments={
                "task_revision": 1,
                "base_revision": 0,
                "notes": [
                    {
                        "kind": "result",
                        "summary": "金额20元",
                        "sources": [bound.model_dump(mode="json")],
                    }
                ],
            },
        ),
        max_duration=1,
        accepted_artifacts=(source.reference,),
    )
    assert canonical.artifact is not None
    accepted = WorkProposal.model_validate_json(
        (await store.get("r", canonical.artifact.artifact_id)).content_json
    )
    assert accepted.update.notes == proposal.update.notes


@pytest.mark.parametrize(
    "text,selector,diagnostic",
    [
        ("重复 重复", {"quote": "重复"}, "exactly once"),
        ("aaa", {"quote": "aa"}, "exactly once"),
        ("金额20元", {"quote": "金额是20元"}, "exactly once"),
        ("金额20元", {"quote": "20元", "pointer": "/data/text"}, '"/text"'),
        ("金额20元", {"quote": "20元", "pointer": ""}, '"/text"'),
        ("金额20元", {"start": 0, "end": 11}, "<= 5 characters"),
        ({"amount": 20}, {"quote": "20"}, "text field"),
        ("金额20元", {"quote": "20元", "start": 0, "end": 5}, None),
    ],
)
async def test_ambiguous_invalid_or_mixed_source_selectors_are_not_silently_repaired(
    text: JsonValue, selector: dict[str, JsonValue], diagnostic: str | None
) -> None:
    store = InMemoryArtifactStore()
    source = await store.put("r", "read", {"text": text})
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(
            call_id="record",
            tool_name="propose_work",
            arguments={
                "task_revision": 1,
                "base_revision": 0,
                "notes": [
                    {
                        "kind": "result",
                        "summary": "Fact",
                        "sources": [
                            {
                                "artifact_id": source.reference.artifact_id,
                                "pointer": "/text",
                                **selector,
                            }
                        ],
                    }
                ],
            },
        ),
        max_duration=1,
        accepted_artifacts=(source.reference,),
    )
    assert record.artifact is None
    error = record.exchange.observation
    assert isinstance(error, ToolFailedObservation)
    assert error.error.category is ErrorCategory.INVALID_TOOL_ARGUMENTS
    if diagnostic is not None:
        assert "notes[0].sources[0]" in error.error.message
        assert diagnostic in error.error.message


async def test_quote_resolution_cannot_read_an_unaccepted_artifact() -> None:
    store = InMemoryArtifactStore()
    source = await store.put("r", "hidden", {"text": "private phrase"})
    executor = ToolExecutor("r", ToolRegistry((work_tool(),)), store, ReadOnlyPolicy())
    record = await executor.execute(
        ToolCallAction(
            call_id="unauthorized",
            tool_name="propose_work",
            arguments={
                "task_revision": 1,
                "base_revision": 0,
                "notes": [
                    {
                        "kind": "result",
                        "summary": "Fact",
                        "sources": [
                            {
                                "artifact_id": source.reference.artifact_id,
                                "pointer": "/bad",
                                "quote": "private",
                            }
                        ],
                    }
                ],
            },
        ),
        max_duration=1,
        accepted_artifacts=(),
    )
    assert record.artifact is None
    assert isinstance(record.exchange.observation, ToolFailedObservation)
    assert record.exchange.observation.error.category is ErrorCategory.POLICY_DENIED
    assert "Available root pointers" not in record.exchange.observation.error.message


def test_quote_selector_requires_explicit_pointer_in_the_model_contract() -> None:
    from pydantic import ValidationError

    from weavria.harness import WorkQuoteReference

    assert "pointer" in WorkQuoteReference.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        WorkQuoteReference.model_validate_json('{"artifact_id":"source","quote":"fact"}')
