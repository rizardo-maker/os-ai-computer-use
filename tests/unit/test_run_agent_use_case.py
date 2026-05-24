from __future__ import annotations

from os_ai_llm.types import LLMResponse, Message, TextPart, ToolCall, ToolDescriptor, ToolResult, Usage
from os_ai_core.application.ports.llm import LLMRequest
from os_ai_core.application.ports.tools import ToolExecutionContext
from os_ai_core.application.use_cases.run_agent import RunAgentCommand, RunAgentUseCase


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            return LLMResponse(
                messages=[Message(role="assistant", content=[TextPart(text="using tool")])],
                tool_calls=[ToolCall(id="tool-1", name="echo", args={"text": "hi"})],
                usage=Usage(input_tokens=1, output_tokens=2),
                provider_context={"state": "1"},
            )
        return LLMResponse(
            messages=[Message(role="assistant", content=[TextPart(text="done")])],
            tool_calls=[],
            usage=Usage(input_tokens=3, output_tokens=4),
            provider_context={"state": "2"},
        )

    def append_tool_result(self, messages, tool_call, result):
        return [*messages, Message(role="user", content=[TextPart(text=result.content[0].text)])]

    def provider_name(self):
        return "fake"

    def model_name(self):
        return "fake-model"


class FakeTools:
    def list_tools(self):
        return [ToolDescriptor(name="echo", kind="function", params={})]

    def execute(self, call, context: ToolExecutionContext):
        return ToolResult(tool_call_id=call.id, content=[TextPart(text=call.args["text"])])


class DuplicateReadOnlyLLM(FakeLLM):
    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            return LLMResponse(
                messages=[Message(role="assistant", content=[TextPart(text="reading")])],
                tool_calls=[
                    ToolCall(id="read-1", name="read_file", args={"path": "a.txt"}),
                    ToolCall(id="read-1", name="read_file", args={"path": "a.txt"}),
                ],
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        return LLMResponse(messages=[], tool_calls=[], usage=Usage())


class CountingReadOnlyTools:
    def __init__(self) -> None:
        self.executions = 0

    def list_tools(self):
        return [ToolDescriptor(name="read_file", kind="function", params={"risk": "read_only"})]

    def execute(self, call, context: ToolExecutionContext):
        self.executions += 1
        return ToolResult(tool_call_id=call.id, content=[TextPart(text="content")])


class RecordingEvents:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_run_agent_use_case_executes_tool_and_preserves_provider_state():
    events = RecordingEvents()
    llm = FakeLLM()
    use_case = RunAgentUseCase(llm=llm, tools=FakeTools(), events=events)

    result = use_case.execute(
        RunAgentCommand(
            job_id="job",
            task="do it",
            tool_descriptors=[],
            system_prompt=None,
            max_iterations=3,
        )
    )

    assert llm.calls == 2
    assert result.provider_context == {"state": "2"}
    assert result.input_tokens == 4
    assert result.output_tokens == 6
    assert any(event.kind == "tool_finished" for event in events.events)


def test_run_agent_use_case_uses_catalog_risk_for_replay_decisions():
    events = RecordingEvents()
    llm = DuplicateReadOnlyLLM()
    tools = CountingReadOnlyTools()
    use_case = RunAgentUseCase(llm=llm, tools=tools, events=events)

    use_case.execute(
        RunAgentCommand(
            job_id="job",
            task="read it",
            tool_descriptors=[],
            system_prompt=None,
            max_iterations=2,
        )
    )

    assert tools.executions == 1
    assert [event.kind for event in events.events].count("tool_finished") == 2
