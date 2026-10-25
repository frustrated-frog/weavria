from pydantic import BaseModel, ConfigDict, JsonValue

from weavria.harness.evidence_contract import (
    schema_allows_value_at_pointer,
    schema_supports_pointer,
)


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    data: dict[str, JsonValue]


def test_schema_grounding_preserves_declared_dynamic_container() -> None:
    schema = _Envelope.model_json_schema()

    assert schema_supports_pointer(schema, "/source_id")
    assert schema_supports_pointer(schema, "/data/businessDate")
    assert schema_supports_pointer(schema, "/data/recentRegimes/0/evidence")
    assert not schema_supports_pointer(schema, "/businessDate")
    assert not schema_supports_pointer(schema, "/recentRegimes/0")


def test_schema_grounding_understands_closed_objects_and_pointer_escaping() -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"a/b": {"type": "object", "properties": {"~key": {}}}},
        "additionalProperties": False,
    }

    assert schema_supports_pointer(schema, "/a~1b/~0key")
    assert not schema_supports_pointer(schema, "/a~2b")
    assert not schema_supports_pointer(schema, "/missing")


def test_schema_grounding_rejects_explicitly_forbidden_property() -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"forbidden": False},
        "additionalProperties": False,
    }

    assert not schema_supports_pointer(schema, "/forbidden")


def test_schema_literal_grounding_rejects_enum_and_type_contradictions() -> None:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "focus": {"enum": ["summary", "details"], "type": "string"},
            "limit": {"type": "integer"},
        },
        "additionalProperties": False,
    }

    assert schema_allows_value_at_pointer(schema, "/focus", "summary")
    assert not schema_allows_value_at_pointer(schema, "/focus", "unknown")
    assert schema_allows_value_at_pointer(schema, "/limit", 3)
    assert not schema_allows_value_at_pointer(schema, "/limit", True)
    assert not schema_allows_value_at_pointer(schema, "/missing", "summary")
