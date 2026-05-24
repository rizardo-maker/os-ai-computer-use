from __future__ import annotations

from os_ai_cli.events import CliEventSink
from os_ai_core.domain.agent.events import AgentEvent


def test_cli_event_sink_can_print_assistant_text(capsys) -> None:
    sink = CliEventSink(show_assistant_text=True)

    sink.emit(AgentEvent.assistant_text("job", "hello"))

    assert capsys.readouterr().out == "hello\n"


def test_cli_event_sink_is_quiet_by_default(capsys) -> None:
    sink = CliEventSink()

    sink.emit(AgentEvent.assistant_text("job", "hello"))

    assert capsys.readouterr().out == ""
