from __future__ import annotations

from dataclasses import dataclass

from os_ai_core.domain.tools.models import ToolRisk
from os_ai_core.domain.tools.policies import ToolPolicyDecision, ToolTrustLevel


@dataclass(frozen=True)
class McpTrustPolicy:
    trust: ToolTrustLevel
    allow_tools: frozenset[str]
    deny_tools: frozenset[str]

    def decide(self, raw_tool_name: str, risk: ToolRisk) -> ToolPolicyDecision:
        if self.trust is ToolTrustLevel.DISABLED:
            return ToolPolicyDecision(False, reason="MCP server is disabled")
        if raw_tool_name in self.deny_tools:
            return ToolPolicyDecision(False, reason="Tool is denied by config")
        if self.allow_tools and raw_tool_name not in self.allow_tools:
            return ToolPolicyDecision(False, reason="Tool is not allowlisted")
        if self.trust is ToolTrustLevel.REMOTE_UNTRUSTED:
            return ToolPolicyDecision(False, reason="Remote MCP is not enabled")
        if risk in {ToolRisk.CODE_EXECUTION, ToolRisk.PRIVILEGED_OS}:
            return ToolPolicyDecision(False, reason=f"Blocked high-risk tool: {risk.value}")
        if self.trust is ToolTrustLevel.LOCAL_UNTRUSTED and risk is not ToolRisk.READ_ONLY:
            return ToolPolicyDecision(True, requires_approval=True)
        return ToolPolicyDecision(True)
