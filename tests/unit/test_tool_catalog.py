from __future__ import annotations

from os_ai_llm.types import ToolDescriptor
from os_ai_core.domain.tools.catalog import ToolCatalog, ToolCatalogSnapshot


def test_tool_catalog_snapshot_is_immutable_and_lookupable() -> None:
    descriptors = [ToolDescriptor(name="echo", kind="function", params={})]

    snapshot = ToolCatalogSnapshot.create(version=7, tools=descriptors)
    descriptors.append(ToolDescriptor(name="late", kind="function", params={}))

    assert snapshot.version == 7
    assert tuple(tool.name for tool in snapshot.tools) == ("echo",)
    assert snapshot.get("echo") is snapshot.tools[0]
    assert snapshot.get("missing") is None


def test_tool_catalog_replaces_snapshots_with_monotonic_versions() -> None:
    catalog = ToolCatalog()

    first = catalog.replace([ToolDescriptor(name="echo", kind="function", params={})])
    second = catalog.replace([ToolDescriptor(name="search", kind="function", params={})])

    assert first.version == 1
    assert second.version == 2
    assert catalog.snapshot is second
    assert second.get("search") is not None
    assert second.get("echo") is None
