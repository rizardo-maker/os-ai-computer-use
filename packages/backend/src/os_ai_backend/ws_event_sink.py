from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from fastapi import WebSocket

from os_ai_core.adapters.events.callback_event_sink import CallbackEventSink
from os_ai_core.application.ports.events import EventSink
from os_ai_core.domain.agent.events import AgentEvent


SendEvent = Callable[[WebSocket, str, dict[str, Any]], Awaitable[None]]


class WebSocketEventSink(EventSink):
    def __init__(
        self,
        websocket: WebSocket,
        loop: asyncio.AbstractEventLoop,
        job_id: str,
        send_event: SendEvent,
    ) -> None:
        self._websocket = websocket
        self._loop = loop
        self._job_id = job_id
        self._send_event = send_event
        self._legacy = CallbackEventSink(self.emit_legacy)

    def emit(self, event: AgentEvent) -> None:
        self._legacy.emit(event)

    def emit_legacy(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "assistant_text":
            self._schedule("event.log", {"level": "info", "message": payload.get("text", ""), "jobId": self._job_id})
        elif kind == "tool_call":
            self._schedule(
                "event.action",
                {"name": payload.get("name"), "status": "start", "meta": payload.get("args", {}), "jobId": self._job_id},
            )
        elif kind == "tool_result_text":
            self._schedule("event.action", {"name": "tool_result", "status": "ok", "meta": payload, "jobId": self._job_id})
        elif kind == "tool_result_image":
            self._schedule(
                "event.screenshot",
                {
                    "mime": payload.get("media_type", "image/jpeg"),
                    "data": payload.get("data", ""),
                    "ts": None,
                    "jobId": self._job_id,
                },
            )
        elif kind == "progress":
            self._schedule("event.progress", {**payload, "jobId": self._job_id})
        elif kind == "usage":
            self._schedule("event.usage", {**payload, "jobId": self._job_id})

    def _schedule(self, method: str, payload: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(self._send_event(self._websocket, method, payload), self._loop)
