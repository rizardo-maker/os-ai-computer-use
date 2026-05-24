from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from os_ai_llm.types import ToolDescriptor


@dataclass(frozen=True)
class ToolCatalogSnapshot:
    version: int
    created_at: datetime
    tools: tuple[ToolDescriptor, ...]

    @staticmethod
    def create(version: int, tools: list[ToolDescriptor]) -> "ToolCatalogSnapshot":
        return ToolCatalogSnapshot(
            version=version,
            created_at=datetime.now(timezone.utc),
            tools=tuple(tools),
        )

    def get(self, name: str) -> ToolDescriptor | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


class ToolCatalog:
    def __init__(self) -> None:
        self._version = 0
        self._snapshot = ToolCatalogSnapshot.create(0, [])

    @property
    def snapshot(self) -> ToolCatalogSnapshot:
        return self._snapshot

    def replace(self, tools: list[ToolDescriptor]) -> ToolCatalogSnapshot:
        self._version += 1
        self._snapshot = ToolCatalogSnapshot.create(self._version, tools)
        return self._snapshot

    def clear(self) -> ToolCatalogSnapshot:
        return self.replace([])
