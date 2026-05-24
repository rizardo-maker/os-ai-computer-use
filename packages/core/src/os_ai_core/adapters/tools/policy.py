from __future__ import annotations

from os_ai_llm.types import ToolCall
from os_ai_core.domain.tools.models import ToolRisk
from os_ai_core.domain.tools.policies import ToolPolicyDecision


class DefaultToolPolicy:
    def decide(self, call: ToolCall, risk: ToolRisk) -> ToolPolicyDecision:
        if call.name == "computer":
            return ToolPolicyDecision(True)
        if risk is ToolRisk.READ_ONLY:
            return ToolPolicyDecision(True)
        return ToolPolicyDecision(True, requires_approval=True)
