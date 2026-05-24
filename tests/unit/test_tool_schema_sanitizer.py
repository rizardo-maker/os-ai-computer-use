from __future__ import annotations

import pytest

from os_ai_core.adapters.llm.tool_schema_sanitizer import ToolSchemaSanitizer, ToolSchemaTooLargeError


def test_tool_schema_sanitizer_strips_unsupported_keys_without_mutating_source() -> None:
    source = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "default": "/tmp/a",
                "examples": ["/tmp/b"],
            }
        },
    }

    clean = ToolSchemaSanitizer().sanitize(source)

    assert "$schema" not in clean
    assert "default" not in clean["properties"]["path"]
    assert "examples" not in clean["properties"]["path"]
    assert "$schema" in source
    assert "default" in source["properties"]["path"]


def test_tool_schema_sanitizer_replaces_non_object_schema() -> None:
    clean = ToolSchemaSanitizer().sanitize({"type": "string"})

    assert clean == {"type": "object", "properties": {}, "additionalProperties": True}


def test_tool_schema_sanitizer_rejects_too_large_schema() -> None:
    sanitizer = ToolSchemaSanitizer(max_bytes=10)

    with pytest.raises(ToolSchemaTooLargeError):
        sanitizer.sanitize({"type": "object", "properties": {"text": {"type": "string"}}})
