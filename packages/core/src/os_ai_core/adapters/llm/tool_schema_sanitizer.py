from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


UNSUPPORTED_KEYS_BY_DEFAULT = {
    "$schema",
    "$id",
    "examples",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
}


class ToolSchemaTooLargeError(ValueError):
    pass


class ToolSchemaSanitizer:
    def __init__(self, max_bytes: int | None = None) -> None:
        self._max_bytes = max_bytes

    def sanitize(self, schema: dict[str, Any] | Any) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}, "additionalProperties": True}
        clean = self._strip_unsupported(deepcopy(schema))
        if clean.get("type") != "object":
            clean = {"type": "object", "properties": {}, "additionalProperties": True}
        clean.setdefault("properties", {})
        self._validate_size(clean)
        return clean

    def _strip_unsupported(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._strip_unsupported(item)
                for key, item in value.items()
                if key not in UNSUPPORTED_KEYS_BY_DEFAULT
            }
        if isinstance(value, list):
            return [self._strip_unsupported(item) for item in value]
        return value

    def _validate_size(self, schema: dict[str, Any]) -> None:
        if self._max_bytes is None:
            return
        size = len(json.dumps(schema, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        if size > self._max_bytes:
            raise ToolSchemaTooLargeError(f"tool schema is too large: {size} bytes")
