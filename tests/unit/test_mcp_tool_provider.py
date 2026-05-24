from __future__ import annotations

from dataclasses import dataclass

from os_ai_llm.types import TextPart, ToolCall
from os_ai_core.application.ports.approval import ApprovalDecision
from os_ai_core.application.ports.tools import ToolExecutionContext
from os_ai_core.domain.tools.policies import ToolTrustLevel
from os_ai_mcp.client.config import McpServerPolicyConfig
from os_ai_mcp.client.tool_provider import McpToolProvider


@dataclass
class FakeMcpTool:
    name: str
    description: str
    inputSchema: dict


@dataclass
class FakeMcpResult:
    content: list
    isError: bool = False


@dataclass
class FakeMcpText:
    text: str
    type: str = "text"


class FakeMcpSession:
    def __init__(self):
        self.called = []
        self.list_calls = 0
        self.close_calls = 0

    def initialize(self):
        pass

    def list_tools(self):
        self.list_calls += 1
        return [FakeMcpTool("echo.text", "Echo text", {"type": "object", "properties": {"text": {"type": "string"}}})]

    def call_tool(self, name, arguments, timeout_seconds, cancel_token):
        self.called.append((name, arguments))
        return FakeMcpResult([FakeMcpText(arguments["text"])])

    def close(self):
        self.close_calls += 1


class FakeApproval:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests = []

    def request_approval(self, request):
        self.requests.append(request)
        return self.decision


def test_mcp_provider_namespaces_lists_and_executes_tools():
    session = FakeMcpSession()
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.TRUSTED_LOCAL)

    tools = provider.list_tools()

    assert provider.health_state == "ready"
    assert tools[0].name == "mcp__local__echo_text"
    result = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "hello"}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is False
    assert result.content == [TextPart(text="hello")]
    assert session.called == [("echo.text", {"text": "hello"})]


def test_mcp_provider_reports_starting_while_discovering_tools():
    states = []
    provider_ref = {}
    session = FakeMcpSession()

    def observe_list_tools():
        states.append(provider_ref["provider"].health_state)
        return [FakeMcpTool("echo.text", "Echo text", {"type": "object", "properties": {}})]

    session.list_tools = observe_list_tools
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.TRUSTED_LOCAL)
    provider_ref["provider"] = provider

    provider.list_tools()

    assert states == ["starting"]
    assert provider.health_state == "ready"


def test_mcp_provider_close_transitions_to_disabled_and_closes_session():
    states = []
    provider_ref = {}
    session = FakeMcpSession()

    def observe_close():
        states.append(provider_ref["provider"].health_state)
        session.close_calls += 1

    session.close = observe_close
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.TRUSTED_LOCAL)
    provider_ref["provider"] = provider
    provider.list_tools()

    provider.close()

    assert states == ["stopping"]
    assert session.close_calls == 1
    assert provider.health_state == "disabled"


def test_mcp_provider_denies_local_untrusted_mutation_without_approval_flow():
    provider = McpToolProvider("local", FakeMcpSession(), trust=ToolTrustLevel.LOCAL_UNTRUSTED)
    tools = provider.list_tools()

    result = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "hello"}, metadata={"risk": "local_mutation"}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is True
    assert result.metadata["error_code"] == "mcp_tool_approval_denied"
    assert result.metadata["approval_decision"] == "unavailable"


def test_mcp_provider_executes_untrusted_mutation_after_explicit_approval():
    session = FakeMcpSession()
    approval = FakeApproval(ApprovalDecision.APPROVED)
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.LOCAL_UNTRUSTED)
    tools = provider.list_tools()

    result = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "hello"}, metadata={"risk": "local_mutation"}),
        ToolExecutionContext(job_id="job", approval=approval),
    )

    assert result.is_error is False
    assert result.content == [TextPart(text="hello")]
    assert len(approval.requests) == 1
    assert approval.requests[0].job_id == "job"


def test_mcp_provider_allows_inferred_read_only_tool_for_local_untrusted_server():
    session = FakeMcpSession()
    session.list_tools = lambda: [FakeMcpTool("read_file", "Read file", {"type": "object", "properties": {}})]
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.LOCAL_UNTRUSTED)
    tools = provider.list_tools()

    result = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "hello"}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is False


def test_mcp_provider_returns_tool_error_for_stale_tool_call():
    provider = McpToolProvider("local", FakeMcpSession(), trust=ToolTrustLevel.TRUSTED_LOCAL)

    result = provider.execute(
        ToolCall(id="1", name="mcp__local__missing", args={}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is True
    assert result.metadata["error_code"] == "stale_mcp_tool"


def test_mcp_provider_can_execute_sanitized_server_id_names():
    provider = McpToolProvider("local dev", FakeMcpSession(), trust=ToolTrustLevel.TRUSTED_LOCAL)
    tools = provider.list_tools()

    assert tools[0].name == "mcp__local_dev__echo_text"
    assert provider.can_execute(ToolCall(id="1", name=tools[0].name, args={})) is True


def test_mcp_provider_rejects_duplicate_names_after_normalization():
    session = FakeMcpSession()
    session.list_tools = lambda: [
        FakeMcpTool("echo.text", "Echo text", {"type": "object", "properties": {}}),
        FakeMcpTool("echo_text", "Echo text", {"type": "object", "properties": {}}),
    ]
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.TRUSTED_LOCAL)

    try:
        provider.list_tools()
    except ValueError as exc:
        assert "duplicate MCP tool name" in str(exc)
    else:
        raise AssertionError("expected duplicate MCP tool name failure")


def test_mcp_provider_turns_timeout_into_tool_error():
    session = FakeMcpSession()

    def raise_timeout(name, arguments, timeout_seconds, cancel_token):
        raise TimeoutError("call timed out")

    session.call_tool = raise_timeout
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.TRUSTED_LOCAL)
    tools = provider.list_tools()

    result = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "hello"}),
        ToolExecutionContext(job_id="job", timeout_seconds=0.01),
    )

    assert result.is_error is True
    assert result.metadata["error_code"] == "mcp_tool_error"


def test_mcp_provider_truncates_large_text_outputs_and_preserves_metadata():
    session = FakeMcpSession()
    session.call_tool = lambda name, arguments, timeout_seconds, cancel_token: FakeMcpResult([FakeMcpText("abcdef")])
    provider = McpToolProvider(
        "local",
        session,
        trust=ToolTrustLevel.TRUSTED_LOCAL,
        policy=McpServerPolicyConfig(max_output_chars=3),
    )
    tools = provider.list_tools()

    result = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "ignored"}),
        ToolExecutionContext(job_id="job"),
    )

    assert result.is_error is False
    assert result.content == [TextPart(text="abc...[truncated]")]
    assert result.metadata["truncated"] is True
    assert result.metadata["max_output_chars"] == 3


def test_mcp_provider_disabled_server_lists_no_tools():
    session = FakeMcpSession()
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.DISABLED)

    assert provider.list_tools() == []
    assert provider.health_state == "disabled"
    assert session.list_calls == 0


def test_mcp_provider_backs_off_after_list_tools_failure():
    session = FakeMcpSession()
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.TRUSTED_LOCAL)
    tools = provider.list_tools()

    def fail_list_tools():
        session.list_calls += 1
        raise RuntimeError("boom")

    session.list_tools = fail_list_tools

    assert provider.list_tools() == []
    assert provider.list_tools() == []
    assert provider.health_state == "failed"
    assert session.list_calls == 2

    stale = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "hello"}),
        ToolExecutionContext(job_id="job"),
    )
    assert stale.is_error is True
    assert stale.metadata["error_code"] == "stale_mcp_tool"


def test_mcp_provider_tools_changed_invalidates_current_name_map():
    provider = McpToolProvider("local", FakeMcpSession(), trust=ToolTrustLevel.TRUSTED_LOCAL)
    tools = provider.list_tools()

    provider.mark_tools_changed()

    result = provider.execute(
        ToolCall(id="1", name=tools[0].name, args={"text": "hello"}),
        ToolExecutionContext(job_id="job"),
    )
    assert result.is_error is True
    assert result.metadata["error_code"] == "stale_mcp_tool"
