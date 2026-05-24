from __future__ import annotations

from typing import Any, Callable

from os_ai_llm.types import ImagePart, TextPart
from os_ai_core.adapters.events.tool_call_event_mapper import ToolCallEventMapper
from os_ai_core.application.ports.events import EventSink
from os_ai_core.domain.agent.events import AgentEvent


class CallbackEventSink(EventSink):
    def __init__(
        self,
        callback: Callable[[str, dict[str, Any]], None] | None,
        tool_call_mapper: ToolCallEventMapper | None = None,
    ) -> None:
        self._callback = callback
        self._tool_call_mapper = tool_call_mapper or ToolCallEventMapper()

    def emit(self, event: AgentEvent) -> None:
        if self._callback is None:
            return
        try:
            self._emit_legacy(event)
        except Exception:
            return

    def _emit_legacy(self, event: AgentEvent) -> None:
        if event.kind == "assistant_text":
            self._callback("assistant_text", {"text": event.payload.get("text", "")})
        elif event.kind == "progress":
            self._callback("progress", dict(event.payload))
        elif event.kind == "usage":
            self._callback("usage", dict(event.payload))
        elif event.kind == "tool_started":
            call = event.payload["call"]
            for payload in self._tool_call_mapper.map_started(call):
                self._callback("tool_call", payload)
        elif event.kind in {"tool_finished", "tool_failed"}:
            result = event.payload["result"]
            self._emit_tool_result(result)

    def _emit_tool_result(self, result: Any) -> None:
        has_image = any(isinstance(part, ImagePart) for part in result.content)
        if has_image:
            for part in result.content:
                if isinstance(part, ImagePart):
                    self._callback(
                        "tool_result_image",
                        {"media_type": part.media_type, "data": part.data_base64},
                    )
            return
        for part in result.content:
            if isinstance(part, TextPart):
                self._callback("tool_result_text", {"text": part.text})
                return
