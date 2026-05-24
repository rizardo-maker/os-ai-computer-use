from __future__ import annotations

from os_ai_core.application.ports.events import EventSink
from os_ai_core.domain.agent.events import AgentEvent


class RecordingEventSink(EventSink):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)
