from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from os_ai_llm.types import ImagePart, TextPart, ToolCall, ToolDescriptor, ToolResult
from os_ai_core.application.ports.approval import ApprovalDecision, ApprovalRequest
from os_ai_core.application.ports.tools import ToolExecutionContext, ToolProvider
from os_ai_core.domain.tools.models import ToolRisk
from os_ai_core.domain.tools.policies import ToolTrustLevel
from os_ai_mcp.client.connection import McpClientSession
from os_ai_mcp.client.config import McpServerPolicyConfig
from os_ai_mcp.client.trust_policy import McpTrustPolicy


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
McpProviderHealthState = Literal["disabled", "starting", "ready", "failed", "stopping"]


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value).strip("_")
    return cleaned or "tool"


@dataclass(frozen=True)
class McpToolName:
    server_id: str
    raw_name: str

    @property
    def namespaced(self) -> str:
        return f"mcp__{self.server_id}__{self.raw_name}"

    @staticmethod
    def parse(value: str) -> "McpToolName | None":
        parts = value.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            return None
        return McpToolName(server_id=parts[1], raw_name=parts[2])


class McpToolProvider(ToolProvider):
    def __init__(
        self,
        server_id: str,
        session: McpClientSession,
        trust: ToolTrustLevel = ToolTrustLevel.LOCAL_UNTRUSTED,
        policy: McpServerPolicyConfig | None = None,
    ) -> None:
        self._server_id = server_id
        self._session = session
        self._trust = trust
        self._policy = policy or McpServerPolicyConfig()
        self._trust_policy = McpTrustPolicy(
            trust=trust,
            allow_tools=self._policy.allow_tools,
            deny_tools=self._policy.deny_tools,
        )
        self._provider_server_id = _safe_name(self._server_id)
        self._name_map: dict[str, str] = {}
        self._risk_map: dict[str, ToolRisk] = {}
        self._health_state: McpProviderHealthState = "disabled" if trust is ToolTrustLevel.DISABLED else "ready"
        self._failure_count = 0
        self._retry_after_monotonic = 0.0

    @property
    def provider_id(self) -> str:
        return f"mcp.{self._server_id}"

    @property
    def health_state(self) -> McpProviderHealthState:
        return self._health_state

    def mark_tools_changed(self) -> None:
        self._name_map = {}
        self._risk_map = {}
        if self._trust is not ToolTrustLevel.DISABLED:
            self._health_state = "ready"
            self._retry_after_monotonic = 0.0

    def close(self) -> None:
        if self._health_state == "disabled":
            return
        self._health_state = "stopping"
        try:
            self._session.close()
        finally:
            self._name_map = {}
            self._risk_map = {}
            self._health_state = "disabled"

    def list_tools(self) -> list[ToolDescriptor]:
        if self._trust is ToolTrustLevel.DISABLED:
            self._health_state = "disabled"
            self._name_map = {}
            self._risk_map = {}
            return []
        if self._health_state == "failed" and time.monotonic() < self._retry_after_monotonic:
            return []

        self._health_state = "starting"
        descriptors: list[ToolDescriptor] = []
        name_map: dict[str, str] = {}
        risk_map: dict[str, ToolRisk] = {}
        try:
            tools = self._session.list_tools()
        except Exception:
            self._mark_failed()
            return []

        for tool in tools:
            raw_name = str(getattr(tool, "name", ""))
            if not raw_name:
                continue
            schema = (
                getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
                or {"type": "object", "properties": {}}
            )
            description = str(getattr(tool, "description", "") or "")
            name = McpToolName(self._provider_server_id, _safe_name(raw_name)).namespaced
            if name in name_map:
                self._mark_failed()
                raise ValueError(f"duplicate MCP tool name after normalization: {name}")
            risk = self._infer_risk(raw_name)
            name_map[name] = raw_name
            risk_map[name] = risk
            descriptors.append(
                ToolDescriptor(
                    name=name,
                    kind="function",
                    params={
                        "description": description,
                        "input_schema": schema,
                        "mcp_server_id": self._server_id,
                        "mcp_raw_name": raw_name,
                        "risk": risk.value,
                    },
                )
            )
        self._name_map = name_map
        self._risk_map = risk_map
        self._health_state = "ready"
        self._failure_count = 0
        self._retry_after_monotonic = 0.0
        return descriptors

    def can_execute(self, call: ToolCall) -> bool:
        parsed = McpToolName.parse(call.name)
        return parsed is not None and parsed.server_id == self._provider_server_id

    def execute(self, call: ToolCall, context: ToolExecutionContext) -> ToolResult:
        parsed = McpToolName.parse(call.name)
        if parsed is None:
            return self._error(call.id, "invalid_mcp_tool_name", "Invalid MCP tool name")
        raw_name = self._name_map.get(call.name)
        if raw_name is None:
            return self._error(call.id, "stale_mcp_tool", "MCP tool is not in the current catalog")

        risk = self._risk_from_metadata(call)
        decision = self._trust_policy.decide(raw_name, risk)
        if not decision.allowed:
            return self._error(
                call.id,
                "mcp_tool_denied",
                decision.reason or "MCP tool denied by policy",
            )
        if decision.requires_approval:
            approval_result = self._request_approval(call, context, raw_name, risk)
            if approval_result is not ApprovalDecision.APPROVED:
                return self._error(
                    call.id,
                    "mcp_tool_approval_denied",
                    f"MCP tool approval result: {approval_result.value}",
                    {"approval_decision": approval_result.value},
                )

        try:
            raw = self._session.call_tool(
                raw_name,
                call.args,
                timeout_seconds=context.timeout_seconds,
                cancel_token=context.cancel_token,
            )
        except Exception as exc:
            return self._error(call.id, "mcp_tool_error", str(exc))

        content, truncated = self._normalize_content(raw)
        metadata = {
            "provider_tool_type": "function",
            "provider_id": self.provider_id,
            "mcp_server_id": self._server_id,
            "mcp_raw_name": raw_name,
        }
        if truncated:
            metadata["truncated"] = True
            metadata["max_output_chars"] = self._policy.max_output_chars
        return ToolResult(
            tool_call_id=call.id,
            content=content,
            is_error=bool(getattr(raw, "isError", False) or getattr(raw, "is_error", False)),
            metadata=metadata,
        )

    def _request_approval(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        raw_name: str,
        risk: ToolRisk,
    ) -> ApprovalDecision:
        if context.approval is None:
            return ApprovalDecision.UNAVAILABLE
        return context.approval.request_approval(
            ApprovalRequest(
                job_id=context.job_id,
                tool_call=call,
                risk=risk,
                summary=f"MCP tool {self._server_id}.{raw_name} requires approval ({risk.value})",
            )
        )

    def _risk_from_metadata(self, call: ToolCall) -> ToolRisk:
        try:
            raw = call.metadata.get("risk")
            if raw:
                return ToolRisk(str(raw))
        except Exception:
            pass
        return self._risk_map.get(call.name, ToolRisk.LOCAL_MUTATION)

    def _infer_risk(self, raw_name: str) -> ToolRisk:
        name = raw_name.lower()
        if any(token in name for token in ("read", "list", "search", "get", "find")):
            return ToolRisk.READ_ONLY
        if any(token in name for token in ("shell", "exec", "run_command", "command")):
            return ToolRisk.CODE_EXECUTION
        return ToolRisk.LOCAL_MUTATION

    def _mark_failed(self) -> None:
        self._health_state = "failed"
        self._name_map = {}
        self._risk_map = {}
        self._failure_count += 1
        backoff_seconds = min(30.0, float(2 ** min(self._failure_count, 5)))
        self._retry_after_monotonic = time.monotonic() + backoff_seconds

    def _normalize_content(self, raw: Any) -> tuple[list[Any], bool]:
        content = getattr(raw, "content", raw)
        if not isinstance(content, list):
            structured = getattr(raw, "structuredContent", None) or getattr(raw, "structured_content", None)
            if structured is not None:
                text = json.dumps(structured, ensure_ascii=False)
            else:
                text = str(content)
            text, truncated = self._truncate_text(text)
            return [TextPart(text=text)], truncated

        parts: list[Any] = []
        truncated = False
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text, was_truncated = self._truncate_text(str(getattr(item, "text", "")))
                truncated = truncated or was_truncated
                parts.append(TextPart(text=text))
            elif item_type == "image":
                data = getattr(item, "data", "") or getattr(item, "data_base64", "")
                mime = getattr(item, "mimeType", None) or getattr(item, "mime_type", None) or "image/png"
                parts.append(ImagePart(media_type=str(mime), data_base64=str(data)))
            else:
                text, was_truncated = self._truncate_text(str(item))
                truncated = truncated or was_truncated
                parts.append(TextPart(text=text))
        return parts or [TextPart(text="")], truncated

    def _truncate_text(self, text: str) -> tuple[str, bool]:
        max_chars = self._policy.max_output_chars
        if max_chars < 0 or len(text) <= max_chars:
            return text, False
        return f"{text[:max_chars]}...[truncated]", True

    def _error(
        self,
        tool_call_id: str,
        code: str,
        message: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        metadata = {
            "error_code": code,
            "provider_tool_type": "function",
            "provider_id": self.provider_id,
            "mcp_server_id": self._server_id,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return ToolResult(
            tool_call_id=tool_call_id,
            content=[TextPart(text=f"error: {message}")],
            is_error=True,
            metadata=metadata,
        )
