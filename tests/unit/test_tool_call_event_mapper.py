from __future__ import annotations

from os_ai_llm.types import ToolCall
from os_ai_core.adapters.events.tool_call_event_mapper import ToolCallEventMapper


def test_tool_call_event_mapper_expands_openai_batch_actions() -> None:
    call = ToolCall(
        id="call-1",
        name="computer",
        args={"type": "drag"},
        metadata={
            "_openai_actions": [
                {"type": "move", "x": 10, "y": 20},
                {"type": "click", "button": "left"},
            ]
        },
    )

    events = ToolCallEventMapper().map_started(call)

    assert events == [
        {"name": "computer", "args": {"type": "move", "x": 10, "y": 20}},
        {"name": "computer", "args": {"type": "click", "button": "left"}},
    ]


def test_tool_call_event_mapper_uses_canonical_args_for_single_call() -> None:
    call = ToolCall(id="call-1", name="mcp__local__echo", args={"text": "hi"})

    events = ToolCallEventMapper().map_started(call)

    assert events == [{"name": "mcp__local__echo", "args": {"text": "hi"}}]
