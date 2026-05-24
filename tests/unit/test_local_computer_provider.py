from __future__ import annotations

from os_ai_llm.types import ToolCall
from os_ai_core.adapters.tools.local_computer_provider import LocalComputerToolProvider
from os_ai_core.application.ports.tools import ToolExecutionContext
from os_ai_core.tools.registry import ToolRegistry


def test_local_computer_provider_lists_default_computer_tool() -> None:
    registry = ToolRegistry()
    registry.register("computer", lambda args: [{"type": "text", "text": "ok"}])
    provider = LocalComputerToolProvider(registry)

    tools = provider.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "computer"
    assert tools[0].kind == "computer_use"


def test_local_computer_provider_executes_legacy_computer_tool() -> None:
    registry = ToolRegistry()
    registry.register("computer", lambda args: [{"type": "text", "text": args["action"]}])
    provider = LocalComputerToolProvider(registry)

    result = provider.execute(
        ToolCall(id="1", name="computer", args={"action": "screenshot"}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is False
    assert result.content[0].text == "screenshot"
    assert result.metadata["provider_id"] == "local.computer"
