from __future__ import annotations

from os_ai_core.application.ports.events import EventSink
from os_ai_core.domain.agent.events import AgentEvent


class NoopEventSink(EventSink):
    def emit(self, event: AgentEvent) -> None:
        return
