from __future__ import annotations

from os_ai_core.domain.agent.events import AgentEvent


def test_tool_catalog_changed_event_is_internal_domain_event() -> None:
    event = AgentEvent.tool_catalog_changed("job", version=3)

    assert event.kind == "tool_catalog_changed"
    assert event.payload == {"version": 3}
