from __future__ import annotations

from os_ai_core.application.ports.events import EventSink
from os_ai_core.domain.agent.events import AgentEvent


class CliEventSink(EventSink):
    def __init__(self, show_assistant_text: bool = False) -> None:
        self.show_assistant_text = show_assistant_text

    def emit(self, event: AgentEvent) -> None:
        if self.show_assistant_text and event.kind == "assistant_text":
            text = str(event.payload.get("text", "")).strip()
            if text:
                print(text)
