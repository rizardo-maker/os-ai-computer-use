from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from os_ai_llm.types import ToolCall
from os_ai_core.domain.tools.models import ToolRisk


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ApprovalRequest:
    job_id: str
    tool_call: ToolCall
    risk: ToolRisk
    summary: str
    expires_in_seconds: int = 60


class ApprovalPort(Protocol):
    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        ...
