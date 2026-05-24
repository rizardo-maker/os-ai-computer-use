from __future__ import annotations

from os_ai_llm.types import ToolDescriptor
from os_ai_core.adapters.llm.legacy_gateway import LegacyLLMGateway
from os_ai_core.adapters.llm.legacy_mapper import LegacyLLMMapper
from os_ai_core.application.ports.llm import LLMRequest


class RecordingClient:
    def __init__(self) -> None:
        self.tools = None

    def generate(self, messages, tools, system, tool_choice, max_tokens, allow_parallel_tools, provider_context):
        self.tools = tools
        return "response"

    def format_tool_result(self, result):
        raise AssertionError("not used")

    def get_provider_name(self):
        return "fake"

    def get_model_name(self):
        return "fake-model"


def test_legacy_llm_gateway_sanitizes_function_tool_schema_at_adapter_boundary() -> None:
    client = RecordingClient()
    gateway = LegacyLLMGateway(client)
    schema = {
        "$id": "tool",
        "type": "object",
        "properties": {"value": {"type": "string", "default": "x"}},
    }

    response = gateway.generate(
        LLMRequest(
            messages=[],
            tools=[ToolDescriptor(name="echo", kind="function", params={"input_schema": schema})],
        )
    )

    assert response == "response"
    assert schema["$id"] == "tool"
    tool = client.tools[0]
    assert tool.params["schema_sanitized"] is True
    assert "$id" not in tool.params["input_schema"]
    assert "default" not in tool.params["input_schema"]["properties"]["value"]


def test_legacy_llm_mapper_preserves_canonical_objects_by_default() -> None:
    mapper = LegacyLLMMapper()
    tool = ToolDescriptor(name="echo", kind="function", params={})

    assert mapper.map_messages([]) == []
    assert mapper.map_tools([tool]) == [tool]
