from copy import deepcopy

from pydantic import JsonValue

from weavria.tools import model_schema_view


def test_model_schema_keeps_constraints_and_business_keys_without_mutating_registration() -> None:
    schema: dict[str, JsonValue] = {
        "title": "PythonModel",
        "description": "Python type documentation",
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "value"],
        "properties": {
            "title": {"title": "Title", "type": "string", "minLength": 1},
            "value": {
                "title": "Value",
                "description": "This is essential parameter guidance",
                "anyOf": [{"type": "integer", "minimum": 3}, {"type": "null"}],
                "default": None,
            },
            "mode": {"type": "string", "enum": ["safe", "other"], "default": "safe"},
            "opaque": {"default": {"title": "business data"}, "const": {"title": "keep"}},
            "items": {"type": "array", "items": {"$ref": "#/$defs/title"}},
        },
        "$defs": {"title": {"title": "Type", "type": "string", "maxLength": 16}},
    }
    original = deepcopy(schema)
    view = model_schema_view(schema)
    assert schema == original
    assert "title" not in view and "description" not in view
    properties = view["properties"]
    assert isinstance(properties, dict)
    assert properties["title"] == {"type": "string", "minLength": 1}
    assert properties["value"] == {
        "description": "This is essential parameter guidance",
        "anyOf": [{"type": "integer", "minimum": 3}, {"type": "null"}],
    }
    original_properties = original["properties"]
    assert isinstance(original_properties, dict)
    for key in ("mode", "opaque", "items"):
        assert properties[key] == original_properties[key]
    assert view["required"] == original["required"]
    assert view["additionalProperties"] is False
    assert view["$defs"] == {"title": {"type": "string", "maxLength": 16}}
