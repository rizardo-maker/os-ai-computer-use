from __future__ import annotations

from os_ai_core.adapters.events.noop_event_sink import NoopEventSink
from os_ai_core.adapters.events.recording_event_sink import RecordingEventSink
from os_ai_core.domain.agent.events import AgentEvent


def test_noop_event_sink_accepts_events() -> None:
    NoopEventSink().emit(AgentEvent.progress("job", "stage", 1))


def test_recording_event_sink_records_events() -> None:
    sink = RecordingEventSink()
    event = AgentEvent.progress("job", "stage", 1)

    sink.emit(event)

    assert sink.events == [event]
