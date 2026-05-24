from __future__ import annotations

from typing import List

import pytest

from os_ai_llm.interfaces import LLMClient
from os_ai_llm.types import Message, ToolDescriptor, LLMResponse, ToolCall, Usage, TextPart
from os_ai_core.tools.registry import ToolRegistry
from os_ai_core.orchestrator import Orchestrator


class DummyLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: List[Message], tools: List[ToolDescriptor], system=None, tool_choice="auto", max_tokens=1024, allow_parallel_tools=True, provider_context=None) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            # Request a single tool call on first turn
            return LLMResponse(
                messages=[Message(role="assistant", content=[TextPart(text="executing tool...")])],
                tool_calls=[ToolCall(id="1", name="computer", args={"action": "wait", "seconds": 0.01})],
                usage=Usage(input_tokens=1, output_tokens=1),
            )
        # No more tool calls on second turn
        return LLMResponse(
            messages=[Message(role="assistant", content=[TextPart(text="done")])],
            tool_calls=[],
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    def format_tool_result(self, result):
        return Message(role="user", content=[TextPart(text="tool_result")])


def test_orchestrator_runs_and_handles_tool_call(monkeypatch):
    # Use a stub tool registry where 'computer' does nothing
    reg = ToolRegistry()
    reg.register("computer", lambda args: [{"type": "text", "text": "ok"}])

    client = DummyLLM()
    orch = Orchestrator(client, reg)
    msgs = orch.run("task", [ToolDescriptor(name="computer", kind="computer_use")], system=None, max_iterations=3)

    assert any(m.role == "assistant" for m in msgs)
    assert client.calls >= 2


def test_orchestrator_can_use_legacy_runner_rollback_path(monkeypatch):
    captured_args = {}
    reg = ToolRegistry()

    def handler(args):
        captured_args.update(args)
        return [{"type": "text", "text": "ok"}]

    reg.register("computer", handler)

    class BatchLLM(DummyLLM):
        def generate(self, messages, tools, system=None, tool_choice="auto", max_tokens=1024, allow_parallel_tools=True, provider_context=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    messages=[],
                    tool_calls=[
                        ToolCall(
                            id="1",
                            name="computer",
                            args={"action": "screenshot"},
                            metadata={
                                "_openai_batch": True,
                                "_openai_actions": [{"action": "screenshot"}],
                            },
                        )
                    ],
                    usage=Usage(input_tokens=1, output_tokens=1),
                )
            return LLMResponse(messages=[], tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1))

    monkeypatch.setenv("OS_AI_USE_APPLICATION_RUNNER", "0")
    orch = Orchestrator(BatchLLM(), reg)

    orch.run("task", [ToolDescriptor(name="computer", kind="computer_use")], system=None, max_iterations=2)

    assert captured_args["_openai_batch"] is True
    assert captured_args["_openai_actions"] == [{"action": "screenshot"}]
