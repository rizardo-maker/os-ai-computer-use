from __future__ import annotations

from dataclasses import replace

from os_ai_llm.interfaces import LLMClient
from os_ai_llm.types import Message, ToolCall, ToolDescriptor, ToolResult
from os_ai_core.adapters.llm.legacy_mapper import LegacyLLMMapper
from os_ai_core.adapters.llm.tool_schema_sanitizer import ToolSchemaSanitizer
from os_ai_core.application.ports.llm import LLMGateway, LLMRequest


class LegacyLLMGateway(LLMGateway):
    def __init__(
        self,
        client: LLMClient,
        schema_sanitizer: ToolSchemaSanitizer | None = None,
        mapper: LegacyLLMMapper | None = None,
    ) -> None:
        self._client = client
        self._schema_sanitizer = schema_sanitizer or ToolSchemaSanitizer()
        self._mapper = mapper or LegacyLLMMapper()

    def generate(self, request: LLMRequest):
        return self._client.generate(
            messages=self._mapper.map_messages(request.messages),
            tools=self._sanitize_tools(request.tools),
            system=request.system,
            tool_choice=request.tool_choice,
            max_tokens=request.max_tokens,
            allow_parallel_tools=request.allow_parallel_tools,
            provider_context=request.provider_context,
        )

    def append_tool_result(
        self,
        messages: list[Message],
        tool_call: ToolCall,
        result: ToolResult,
    ) -> list[Message]:
        return [*messages, self._client.format_tool_result(self._mapper.map_tool_result(result))]

    def provider_name(self) -> str:
        return self._client.get_provider_name()

    def model_name(self) -> str:
        return self._client.get_model_name()

    def _sanitize_tools(self, tools: list[ToolDescriptor]) -> list[ToolDescriptor]:
        sanitized: list[ToolDescriptor] = []
        for tool in self._mapper.map_tools(tools):
            if tool.kind != "function":
                sanitized.append(tool)
                continue
            raw_schema = tool.params.get("input_schema") or tool.params.get("parameters")
            clean_schema = self._schema_sanitizer.sanitize(raw_schema)
            params = dict(tool.params)
            if raw_schema != clean_schema:
                params["schema_sanitized"] = True
            if "input_schema" in params:
                params["input_schema"] = clean_schema
            else:
                params["parameters"] = clean_schema
            sanitized.append(replace(tool, params=params))
        return sanitized
