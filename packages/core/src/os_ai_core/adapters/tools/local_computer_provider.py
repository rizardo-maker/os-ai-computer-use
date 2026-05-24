from __future__ import annotations

from os_ai_llm.types import ToolCall, ToolDescriptor, ToolResult
from os_ai_core.application.ports.tools import ToolExecutionContext, ToolProvider
from os_ai_core.tools.registry import ToolRegistry


class LocalComputerToolProvider(ToolProvider):
    def __init__(
        self,
        registry: ToolRegistry,
        descriptors: list[ToolDescriptor] | None = None,
        strict_provider_metadata: bool = True,
    ) -> None:
        self._registry = registry
        self._descriptors = list(descriptors or [])
        self._strict_provider_metadata = strict_provider_metadata

    @property
    def provider_id(self) -> str:
        return "local.computer"

    def list_tools(self) -> list[ToolDescriptor]:
        if not self._descriptors and "computer" in self._registry.names():
            return [ToolDescriptor(name="computer", kind="computer_use", params={})]
        return list(self._descriptors)

    def can_execute(self, call: ToolCall) -> bool:
        return call.name in self._registry.names()

    def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        registry_call = self._to_registry_call(call)
        result = self._registry.execute(registry_call, cancel_token=context.cancel_token)
        result.metadata.update(call.metadata)
        result.metadata.setdefault("provider_tool_type", "computer_use")
        result.metadata.setdefault("provider_id", self.provider_id)
        return result

    def _to_registry_call(self, call: ToolCall) -> ToolCall:
        if not self._strict_provider_metadata:
            return ToolCall(id=call.id, name=call.name, args={**call.args, **call.metadata})
        if call.name != "computer":
            return call
        legacy_args = dict(call.args)
        for key in ("_openai_batch", "_openai_actions"):
            if key in call.metadata:
                legacy_args[key] = call.metadata[key]
        return ToolCall(id=call.id, name=call.name, args=legacy_args)
