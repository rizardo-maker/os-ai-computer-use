from __future__ import annotations

from os_ai_llm.types import TextPart, ToolCall, ToolResult
from os_ai_core.application.services.tool_execution_ledger import (
    ReplayDecision,
    ToolExecutionLedger,
    stable_json_hash,
)
from os_ai_core.domain.tools.models import ToolRisk


def test_ledger_returns_cached_read_only_result_for_duplicate_call():
    ledger = ToolExecutionLedger()
    call = ToolCall(id="call-1", name="search", args={"q": "x"})
    args_hash = stable_json_hash(call.args)

    assert ledger.decide("job", call, ToolRisk.READ_ONLY, args_hash) is ReplayDecision.EXECUTE
    ledger.record_result("job", call, ToolResult("call-1", [TextPart(text="ok")]))

    assert ledger.decide("job", call, ToolRisk.READ_ONLY, args_hash) is ReplayDecision.RETURN_CACHED_RESULT
    assert ledger.cached_result("job", "call-1").content[0].text == "ok"


def test_ledger_rejects_duplicate_side_effecting_call():
    ledger = ToolExecutionLedger()
    call = ToolCall(id="call-1", name="click", args={"x": 1})
    args_hash = stable_json_hash(call.args)

    assert ledger.decide("job", call, ToolRisk.LOCAL_MUTATION, args_hash) is ReplayDecision.EXECUTE
    ledger.record_result("job", call, ToolResult("call-1", [TextPart(text="clicked")]))

    assert ledger.decide("job", call, ToolRisk.LOCAL_MUTATION, args_hash) is ReplayDecision.REJECT_DUPLICATE_SIDE_EFFECT


def test_ledger_rejects_same_call_id_with_different_args():
    ledger = ToolExecutionLedger()
    call = ToolCall(id="call-1", name="search", args={"q": "x"})
    changed = ToolCall(id="call-1", name="search", args={"q": "y"})

    assert ledger.decide("job", call, ToolRisk.READ_ONLY, stable_json_hash(call.args)) is ReplayDecision.EXECUTE

    assert (
        ledger.decide("job", changed, ToolRisk.READ_ONLY, stable_json_hash(changed.args))
        is ReplayDecision.REJECT_DUPLICATE_SIDE_EFFECT
    )
