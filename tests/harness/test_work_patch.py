"""局部协议的输入及定位边界;真实执行和回放由 test_work_recheck 覆盖。"""

import json

import pytest
from pydantic import JsonValue, ValidationError

from weavria.context import ContextCompilerConfig
from weavria.harness import HarnessConfig, WorkPatch, WorkUpdate
from weavria.harness.work_patch import locate_quote


@pytest.mark.parametrize(
    "document, quote, pointer",
    [
        ("text", "text", ""),
        ({"a/b~": [0, "Hello 原文"]}, "原文", "/a~1b~0/1"),
    ],
)
def test_exact_quote_location(document: JsonValue, quote: str, pointer: str) -> None:
    assert locate_quote(document, quote) == pointer


@pytest.mark.parametrize(
    "document, quote",
    [
        ({"a": "text", "b": "text"}, "text"),
        ("aaa", "aa"),
        ({"field": "missing"}, "field"),  # 键名不作为证据文本。
        (["text", *range(1024)], "text"),
        ({"a": "text", "b": "x" * 65536}, "text"),
    ],
)
def test_ambiguous_missing_or_partial_scan_never_claims_unique(
    document: JsonValue,
    quote: str,
) -> None:
    with pytest.raises(ValueError):
        locate_quote(document, quote)


def test_deep_documents_require_explicit_pointer() -> None:
    value: JsonValue = "text"
    for _ in range(34):
        value = [value]
    with pytest.raises(ValueError, match="capacity"):
        locate_quote(value, "text")


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"remove": ["a", "a"]},
        {"upsert": [{"note_id": "a"}, {"note_id": "a"}]},
        {"remove": ["a"], "review": ["a"]},
        {"base_revision": 10, "review": ["a"]},
        {"upsert": [{"note_id": "a", "depends_on": [{"note_id": "b", "revision": 2}]}]},
        {"upsert": [{"note_id": "a", "summary": "secret" * 600}]},
        {"upsert": [{"note_id": f"n{i}", "kind": "plan", "summary": "中" * 230} for i in range(6)]},
    ],
)
def test_patch_rejects_duplicate_intents_versions_and_oversized_input(
    value: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError):
        WorkPatch.model_validate_json(json.dumps(value))


def test_explicit_review_selection_is_versioned_and_roundtrips() -> None:
    with pytest.raises(ValidationError):
        WorkUpdate(task_revision=1, base_revision=0, recheck_note_ids=())
    update = WorkUpdate(schema_version=4, task_revision=1, base_revision=0, recheck_note_ids=())
    assert WorkUpdate.model_validate_json(update.model_dump_json()) == update
    legacy = WorkUpdate(task_revision=1, base_revision=0)
    assert "recheck_note_ids" not in legacy.model_dump()


def test_patch_configuration_requires_working_records() -> None:
    base = {
        "context": ContextCompilerConfig(window_tokens=32000, output_tokens=256),
        "configuration_id": "patch-test",
    }
    with pytest.raises(ValueError, match="working records"):
        HarnessConfig(**base, work_updates="patch")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown"):
        HarnessConfig(**base, work_updates="merge")  # type: ignore[arg-type]


def test_replace_tool_does_not_expose_or_accept_host_recheck_metadata() -> None:
    from weavria.harness.work import work_tool
    from weavria.tools import ToolFault

    tool = work_tool()
    properties = tool.descriptor.input_schema["properties"]
    assert isinstance(properties, dict) and "recheck_note_ids" not in properties
    with pytest.raises(ToolFault):
        tool.validate_arguments({"recheck_note_ids": [], "task_revision": 1, "base_revision": 0})


def test_review_may_repeat_upsert_intent_but_never_conflicting_removal() -> None:
    WorkPatch.model_validate_json('{"upsert":[{"note_id":"a"}],"review":["a"]}')
    with pytest.raises(ValidationError):
        WorkPatch.model_validate_json('{"upsert":[{"note_id":"a"}],"remove":["a"]}')
    with pytest.raises(ValidationError):
        WorkPatch.model_validate_json('{"review":["a","a"]}')
