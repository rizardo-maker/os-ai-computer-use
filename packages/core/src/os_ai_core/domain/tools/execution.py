from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from os_ai_llm.types import ToolCall, ToolResult
from os_ai_core.domain.tools.policies import ToolPolicyDecision


class ToolExecutionStatus(str, Enum):
    PLANNED = "planned"
    POLICY_CHECKED = "policy_checked"
    APPROVAL_PENDING = "approval_pending"
    EXECUTING = "executing"
    FINISHED = "finished"
    REJECTED = "rejected"


@dataclass
class ToolExecution:
    call: ToolCall
    status: ToolExecutionStatus = ToolExecutionStatus.PLANNED
    policy_decision: ToolPolicyDecision | None = None
    result: ToolResult | None = None

    def apply_policy(self, decision: ToolPolicyDecision) -> None:
        if self.status is not ToolExecutionStatus.PLANNED:
            raise ValueError("policy can be applied only once")
        self.policy_decision = decision
        if not decision.allowed:
            self.status = ToolExecutionStatus.REJECTED
        elif decision.requires_approval:
            self.status = ToolExecutionStatus.APPROVAL_PENDING
        else:
            self.status = ToolExecutionStatus.POLICY_CHECKED

    def mark_executing(self) -> None:
        if self.status is not ToolExecutionStatus.POLICY_CHECKED:
            raise ValueError("tool cannot execute before policy allows it")
        self.status = ToolExecutionStatus.EXECUTING

    def finish(self, result: ToolResult) -> None:
        if self.status is not ToolExecutionStatus.EXECUTING:
            raise ValueError("tool result cannot be recorded before execution")
        if result.tool_call_id != self.call.id:
            raise ValueError("tool result id does not match tool call id")
        self.result = result
        self.status = ToolExecutionStatus.FINISHED
