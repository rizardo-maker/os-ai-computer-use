from __future__ import annotations

from os_ai_llm.types import Message, ToolCall, ToolDescriptor, ToolResult


class LegacyLLMMapper:
    def map_messages(self, messages: list[Message]) -> list[Message]:
        return list(messages)

    def map_tools(self, tools: list[ToolDescriptor]) -> list[ToolDescriptor]:
        return list(tools)

    def map_tool_call(self, call: ToolCall) -> ToolCall:
        return call

    def map_tool_result(self, result: ToolResult) -> ToolResult:
        return result
