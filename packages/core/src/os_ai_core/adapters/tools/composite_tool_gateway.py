from __future__ import annotations

from os_ai_llm.types import TextPart, ToolCall, ToolDescriptor, ToolResult
from os_ai_core.application.ports.tools import ToolExecutionContext, ToolGateway, ToolProvider
from os_ai_core.domain.tools.catalog import ToolCatalogSnapshot


class CompositeToolGateway(ToolGateway):
    def __init__(self, providers: list[ToolProvider]) -> None:
        self._providers = list(providers)
        self._snapshot: ToolCatalogSnapshot | None = None
        self._version = 0
        self._dirty = True

    def list_tools(self) -> list[ToolDescriptor]:
        return list(self.snapshot().tools)

    def snapshot(self) -> ToolCatalogSnapshot:
        if self._snapshot is not None and not self._dirty:
            return self._snapshot
        descriptors: list[ToolDescriptor] = []
        seen: set[str] = set()
        for provider in self._providers:
            try:
                provider_tools = provider.list_tools()
            except Exception:
                continue
            for descriptor in provider_tools:
                if descriptor.name in seen:
                    raise ValueError(f"duplicate tool name: {descriptor.name}")
                seen.add(descriptor.name)
                descriptors.append(descriptor)
        self._version += 1
        self._snapshot = ToolCatalogSnapshot.create(self._version, descriptors)
        self._dirty = False
        return self._snapshot

    def mark_dirty(self) -> None:
        self._dirty = True

    def handle_tools_list_changed(self, provider_id: str | None = None) -> None:
        for provider in self._providers:
            if provider_id is not None and provider.provider_id != provider_id:
                continue
            handler = getattr(provider, "mark_tools_changed", None)
            if callable(handler):
                handler()
        self.mark_dirty()

    def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        for provider in self._providers:
            if not provider.can_execute(call):
                continue
            result = provider.execute(call, context)
            result.metadata.setdefault("provider_id", provider.provider_id)
            if call.name != "computer":
                result.metadata.setdefault("provider_tool_type", "function")
            return result
        return ToolResult(
            tool_call_id=call.id,
            content=[TextPart(text=f"error: unknown tool '{call.name}'")],
            is_error=True,
            metadata={"error_code": "unknown_tool"},
        )
