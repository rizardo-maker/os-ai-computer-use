from __future__ import annotations

from os_ai_llm.types import TextPart, ToolCall, ToolResult
from os_ai_core.application.ports.approval import ApprovalDecision, ApprovalPort, ApprovalRequest
from os_ai_core.domain.tools.models import ToolRisk


PROVIDER_SAFETY_DENIAL_CODE = "provider_safety_approval_denied"
STOP_AGENT_LOOP_META = "stop_agent_loop"
SUPPRESS_PROVIDER_TOOL_RESULT_META = "suppress_provider_tool_result"


def approve_provider_safety_checks(
    *,
    job_id: str,
    tool_call: ToolCall,
    risk: ToolRisk,
    approval: ApprovalPort | None,
) -> ToolResult | None:
    pending_checks = tool_call.metadata.get("provider_safety_checks") or []
    if not pending_checks:
        return None

    summary_bits = []
    for check in pending_checks:
        if not isinstance(check, dict):
            continue
        code = str(check.get("code") or "safety_check")
        message = str(check.get("message") or "").strip()
        summary_bits.append(f"{code}: {message}" if message else code)
    summary = "Provider computer-use safety check requires approval"
    if summary_bits:
        summary = f"{summary}: {'; '.join(summary_bits)}"

    if approval is None:
        decision = ApprovalDecision.UNAVAILABLE
    else:
        decision = approval.request_approval(
            ApprovalRequest(
                job_id=job_id,
                tool_call=tool_call,
                risk=risk if risk is not ToolRisk.READ_ONLY else ToolRisk.PRIVILEGED_OS,
                summary=summary,
            )
        )

    if decision is ApprovalDecision.APPROVED:
        tool_call.metadata["provider_safety_checks_approved"] = True
        return None

    metadata = {
        "error_code": PROVIDER_SAFETY_DENIAL_CODE,
        "approval_decision": decision.value,
        "provider_tool_type": "computer_use",
        "provider_safety_checks": pending_checks,
        STOP_AGENT_LOOP_META: True,
        SUPPRESS_PROVIDER_TOOL_RESULT_META: True,
    }
    return ToolResult(
        tool_call_id=tool_call.id,
        content=[TextPart(text=f"error: provider safety approval result: {decision.value}")],
        is_error=True,
        metadata=metadata,
    )
