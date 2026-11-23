"""Pure RFC 6901 addressing of artifact values."""

from pydantic import JsonValue


def resolve_pointer(value: JsonValue, pointer: str) -> JsonValue:
    """Resolve RFC 6901 paths without accepting malformed escapes or negative indices."""
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with slash")
    for raw in pointer[1:].split("/"):
        index = 0
        while index < len(raw):
            if raw[index] == "~":
                if index + 1 >= len(raw) or raw[index + 1] not in "01":
                    raise ValueError("invalid JSON pointer escape")
                index += 1
            index += 1
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and (
            key == "0" or (key.isascii() and key.isdecimal() and not key.startswith("0"))
        ):
            value = value[int(key)]
        else:
            raise ValueError("JSON pointer does not address a value")
    return value
