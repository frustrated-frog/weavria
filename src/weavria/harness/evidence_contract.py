"""Ground low-trust evidence selectors against host-owned JSON Schemas."""

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue


def schema_supports_pointer(schema: Mapping[str, JsonValue], pointer: str) -> bool:
    """Return whether a JSON Pointer can be produced by the declared output schema.

    An unconstrained schema is deliberately treated as compatible: the host has not
    declared enough structure to reject the selector. Closed objects and typed arrays
    remain strict. Local ``$ref`` values, unions and recursive ``JsonValue`` schemas are
    traversed without executing application code.
    """
    if pointer == "":
        return True
    try:
        tokens = _pointer_tokens(pointer)
    except ValueError:
        return False
    root = cast(dict[str, object], dict(schema))
    return _supports(root, root, tokens, depth=0)


def unsupported_evidence_fields(
    fields: tuple[str, ...],
    schemas: tuple[Mapping[str, JsonValue], ...],
) -> tuple[str, ...]:
    """Return selectors unsupported by every candidate producer contract."""
    if not schemas:
        return fields
    return tuple(
        field
        for field in fields
        if not any(schema_supports_pointer(schema, field) for schema in schemas)
    )


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must start with a slash")
    result: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        decoded: list[str] = []
        while index < len(raw):
            if raw[index] != "~":
                decoded.append(raw[index])
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise ValueError("invalid JSON Pointer escape")
            decoded.append("~" if raw[index + 1] == "0" else "/")
            index += 2
        result.append("".join(decoded))
    return tuple(result)


def _supports(
    root: Mapping[str, object],
    node: object,
    tokens: tuple[str, ...],
    *,
    depth: int,
) -> bool:
    if depth > len(tokens) * 8 + 32:
        return False
    if node is False:
        return False
    if not tokens or node is True:
        return True
    current = _mapping(node)
    if current is None:
        return False
    if not current:
        return True

    reference = current.get("$ref")
    if isinstance(reference, str):
        target = _resolve_local_reference(root, reference)
        if target is None:
            return True
        siblings = {key: value for key, value in current.items() if key != "$ref"}
        if siblings and not _supports(root, siblings, tokens, depth=depth + 1):
            return False
        return _supports(root, target, tokens, depth=depth + 1)

    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = current.get(keyword)
        if isinstance(branches, list):
            typed_branches = cast(list[object], branches)
            checks = (_supports(root, branch, tokens, depth=depth + 1) for branch in typed_branches)
            return all(checks) if keyword == "allOf" else any(checks)

    token, remaining = tokens[0], tokens[1:]
    declared_type = current.get("type")
    allowed_types: set[str] = set()
    if isinstance(declared_type, str):
        allowed_types.add(declared_type)
    elif isinstance(declared_type, list) and all(
        isinstance(item, str) for item in cast(list[object], declared_type)
    ):
        allowed_types.update(cast(list[str], declared_type))
    properties = _mapping(current.get("properties"))
    object_possible = (
        "object" in allowed_types
        or properties is not None
        or "additionalProperties" in current
        or not allowed_types
    )
    if object_possible:
        if properties is not None and token in properties:
            return _supports(root, properties[token], remaining, depth=depth + 1)
        # Tool schemas are generated from typed output models. Once named properties
        # exist, undeclared values are not a dependable serialized contract even when
        # vanilla JSON Schema would default this keyword to ``true``.
        additional = current.get("additionalProperties", properties is None)
        if additional is True:
            return True
        if _mapping(additional) is not None:
            return _supports(root, additional, remaining, depth=depth + 1)

    array_possible = "array" in allowed_types or "items" in current
    if array_possible and (token == "-" or token.isdigit()):
        items = current.get("items", {})
        if isinstance(items, list):
            typed_items = cast(list[object], items)
            index = int(token) if token.isdigit() else len(typed_items)
            if index >= len(typed_items):
                return False
            return _supports(root, typed_items[index], remaining, depth=depth + 1)
        return _supports(root, items, remaining, depth=depth + 1)
    return False


def _resolve_local_reference(root: Mapping[str, object], reference: str) -> object | None:
    if not reference.startswith("#"):
        return None
    pointer = reference[1:]
    if pointer == "":
        return root
    try:
        tokens = _pointer_tokens(pointer)
    except ValueError:
        return None
    current: object = root
    for token in tokens:
        current_mapping = _mapping(current)
        if current_mapping is None or token not in current_mapping:
            return None
        current = current_mapping[token]
    return current


def _mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)
