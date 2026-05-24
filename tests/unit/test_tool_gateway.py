from __future__ import annotations

from os_ai_llm.types import TextPart, ToolCall, ToolDescriptor, ToolResult
from os_ai_core.adapters.tools.composite_tool_gateway import CompositeToolGateway
from os_ai_core.application.ports.tools import ToolExecutionContext


class EchoProvider:
    provider_id = "test.echo"

    def __init__(self):
        self.list_count = 0

    def list_tools(self):
        self.list_count += 1
        return [ToolDescriptor(name="echo", kind="function", params={"description": "Echo"})]

    def can_execute(self, call):
        return call.name == "echo"

    def execute(self, call, context):
        return ToolResult(call.id, [TextPart(text=call.args["text"])])


class FailingListProvider:
    provider_id = "test.failing"

    def list_tools(self):
        raise RuntimeError("server unavailable")

    def can_execute(self, call):
        return False

    def execute(self, call, context):
        raise AssertionError("should not execute")


class DynamicProvider(EchoProvider):
    provider_id = "test.dynamic"

    def __init__(self):
        super().__init__()
        self.changed = False

    def list_tools(self):
        self.list_count += 1
        if self.changed:
            return [ToolDescriptor(name="search", kind="function", params={})]
        return [ToolDescriptor(name="echo", kind="function", params={})]

    def mark_tools_changed(self):
        self.changed = True


def test_composite_tool_gateway_routes_to_provider():
    gateway = CompositeToolGateway([EchoProvider()])

    result = gateway.execute(
        ToolCall(id="1", name="echo", args={"text": "hello"}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is False
    assert result.content[0].text == "hello"
    assert result.metadata["provider_id"] == "test.echo"
    assert result.metadata["provider_tool_type"] == "function"


def test_composite_tool_gateway_unknown_tool_returns_typed_error():
    gateway = CompositeToolGateway([])

    result = gateway.execute(
        ToolCall(id="1", name="missing", args={}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is True
    assert result.metadata["error_code"] == "unknown_tool"


def test_composite_tool_gateway_duplicate_names_fail_fast():
    gateway = CompositeToolGateway([EchoProvider(), EchoProvider()])

    try:
        gateway.list_tools()
    except ValueError as exc:
        assert "duplicate tool name" in str(exc)
    else:
        raise AssertionError("expected duplicate tool name failure")


def test_composite_tool_gateway_uses_stable_snapshot_until_marked_dirty():
    provider = EchoProvider()
    gateway = CompositeToolGateway([provider])

    first = gateway.snapshot()
    second = gateway.snapshot()

    assert first is second
    assert provider.list_count == 1

    gateway.mark_dirty()
    third = gateway.snapshot()

    assert third.version == first.version + 1
    assert provider.list_count == 2


def test_composite_tool_gateway_keeps_healthy_tools_when_provider_discovery_fails():
    gateway = CompositeToolGateway([FailingListProvider(), EchoProvider()])

    tools = gateway.list_tools()

    assert [tool.name for tool in tools] == ["echo"]


def test_composite_tool_gateway_invalidates_snapshot_on_tools_list_changed():
    provider = DynamicProvider()
    gateway = CompositeToolGateway([provider])

    assert [tool.name for tool in gateway.list_tools()] == ["echo"]

    gateway.handle_tools_list_changed("test.dynamic")

    assert [tool.name for tool in gateway.list_tools()] == ["search"]
