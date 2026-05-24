from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ApprovalState(str, Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ApprovalScope:
    job_id: str
    tool_call_id: str
    tool_name: str
    expires_at: datetime


@dataclass(frozen=True)
class ApprovalGrant:
    scope: ApprovalScope
    state: ApprovalState
    approved_by: str | None = None

    def can_execute(self, job_id: str, tool_call_id: str, now: datetime) -> bool:
        return (
            self.state is ApprovalState.APPROVED
            and self.scope.job_id == job_id
            and self.scope.tool_call_id == tool_call_id
            and now < self.scope.expires_at
        )
