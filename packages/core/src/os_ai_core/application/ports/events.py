from __future__ import annotations

from typing import Protocol

from os_ai_core.domain.agent.events import AgentEvent


class EventSink(Protocol):
    def emit(self, event: AgentEvent) -> None:
        ...
