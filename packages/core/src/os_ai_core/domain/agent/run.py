from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentRunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {
    AgentRunStatus.SUCCEEDED,
    AgentRunStatus.FAILED,
    AgentRunStatus.CANCELLED,
}


@dataclass
class AgentRun:
    run_id: str
    status: AgentRunStatus = AgentRunStatus.CREATED
    step_index: int = 0

    def start(self) -> None:
        if self.status is not AgentRunStatus.CREATED:
            raise ValueError(f"cannot start run from {self.status.value}")
        self.status = AgentRunStatus.RUNNING

    def next_step(self) -> int:
        if self.status is not AgentRunStatus.RUNNING:
            raise ValueError("cannot advance non-running agent run")
        self.step_index += 1
        return self.step_index

    def finish(self, status: AgentRunStatus) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("finish requires terminal status")
        if self.status in TERMINAL_STATUSES:
            raise ValueError("agent run already finished")
        self.status = status
