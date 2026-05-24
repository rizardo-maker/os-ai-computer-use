from __future__ import annotations

from os_ai_core.application.ports.approval import ApprovalDecision, ApprovalPort, ApprovalRequest


class DenyAllApprovalAdapter(ApprovalPort):
    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.UNAVAILABLE
