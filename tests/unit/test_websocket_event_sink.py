from __future__ import annotations

import asyncio

from os_ai_backend.ws_event_sink import WebSocketEventSink
from os_ai_core.domain.agent.events import AgentEvent


async def _send_event(websocket, method, payload):
    websocket.sent.append((method, payload))


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent = []


def test_websocket_event_sink_maps_progress_to_ws_event() -> None:
    loop = asyncio.new_event_loop()
    websocket = FakeWebSocket()
    sink = WebSocketEventSink(websocket, loop, "job-1", _send_event)

    try:
        sink.emit(AgentEvent.progress("job-1", "iteration_start", 0))
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()

    assert websocket.sent == [("event.progress", {"stage": "iteration_start", "iteration": 0, "jobId": "job-1"})]
