from __future__ import annotations

from dataclasses import dataclass

from os_ai_llm.types import LLMResponse, Message, TextPart, ToolCall, Usage
from os_ai_core.adapters.events.recording_event_sink import RecordingEventSink
from os_ai_core.adapters.tools.composite_tool_gateway import CompositeToolGateway
from os_ai_core.application.ports.llm import LLMRequest
from os_ai_core.application.use_cases.run_agent import RunAgentCommand, RunAgentUseCase
from os_ai_core.domain.tools.policies import ToolTrustLevel
from os_ai_mcp.client.tool_provider import McpToolProvider


@dataclass
class FakeMcpTool:
    name: str
    description: str
    inputSchema: dict


@dataclass
class FakeMcpText:
    text: str
    type: str = "text"


@dataclass
class FakeMcpResult:
    content: list
    isError: bool = False


class FakeMcpSession:
    def __init__(self) -> None:
        self.calls = []

    def initialize(self) -> None:
        return

    def list_tools(self):
        return [FakeMcpTool("echo", "Echo text", {"type": "object", "properties": {"text": {"type": "string"}}})]

    def call_tool(self, name, arguments, timeout_seconds, cancel_token):
        self.calls.append((name, arguments))
        return FakeMcpResult([FakeMcpText(arguments["text"])])

    def close(self) -> None:
        return


class FakeGatewayLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            tool = next(item for item in request.tools if item.name.startswith("mcp__local__echo"))
            return LLMResponse(
                messages=[Message(role="assistant", content=[TextPart(text="calling echo")])],
                tool_calls=[ToolCall(id="tool-1", name=tool.name, args={"text": "hello"})],
                usage=Usage(input_tokens=1, output_tokens=1),
            )
        return LLMResponse(
            messages=[Message(role="assistant", content=[TextPart(text="done")])],
            tool_calls=[],
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    def append_tool_result(self, messages, tool_call, result):
        return [*messages, Message(role="user", content=[TextPart(text=result.content[0].text)])]

    def provider_name(self):
        return "fake"

    def model_name(self):
        return "fake-model"


def test_agent_can_call_mcp_tool_through_canonical_tool_flow() -> None:
    session = FakeMcpSession()
    provider = McpToolProvider("local", session, trust=ToolTrustLevel.TRUSTED_LOCAL)
    llm = FakeGatewayLLM()
    events = RecordingEventSink()

    result = RunAgentUseCase(
        llm=llm,
        tools=CompositeToolGateway([provider]),
        events=events,
    ).execute(
        RunAgentCommand(
            job_id="job",
            task="echo hello",
            tool_descriptors=[],
            system_prompt=None,
            max_iterations=3,
        )
    )

    assert session.calls == [("echo", {"text": "hello"})]
    assert result.input_tokens == 2
    assert result.output_tokens == 2
    assert any(event.kind == "tool_finished" for event in events.events)
