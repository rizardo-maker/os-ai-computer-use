from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from os_ai_llm.types import ToolCall, ToolDescriptor, ToolResult
from os_ai_core.application.ports.approval import ApprovalPort


@dataclass(frozen=True)
class ToolExecutionContext:
    job_id: str
    conversation_id: str | None = None
    timeout_seconds: float | None = None
    cancel_token: Any | None = None
    approval: ApprovalPort | None = None


class ToolGateway(Protocol):
    def list_tools(self) -> list[ToolDescriptor]:
        ...

    def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        ...


class ToolProvider(Protocol):
    @property
    def provider_id(self) -> str:
        ...

    def list_tools(self) -> list[ToolDescriptor]:
        ...

    def can_execute(self, call: ToolCall) -> bool:
        ...

    def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        ...
