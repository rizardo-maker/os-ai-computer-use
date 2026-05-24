from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from os_ai_llm.types import LLMResponse, Message, ToolCall, ToolDescriptor, ToolResult


@dataclass(frozen=True)
class LLMRequest:
    messages: list[Message]
    tools: list[ToolDescriptor]
    system: str | None = None
    max_tokens: int = 1024
    tool_choice: str = "auto"
    allow_parallel_tools: bool = True
    provider_context: dict[str, Any] | None = field(default=None)


class LLMGateway(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    def append_tool_result(
        self,
        messages: list[Message],
        tool_call: ToolCall,
        result: ToolResult,
    ) -> list[Message]:
        ...

    def provider_name(self) -> str:
        ...

    def model_name(self) -> str:
        ...
