from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from os_ai_llm.types import ToolCall, ToolResult
from os_ai_core.domain.tools.models import ToolRisk


class ReplayDecision(str, Enum):
    EXECUTE = "execute"
    RETURN_CACHED_RESULT = "return_cached_result"
    REJECT_DUPLICATE_SIDE_EFFECT = "reject_duplicate_side_effect"


@dataclass(frozen=True)
class ToolExecutionRecord:
    job_id: str
    tool_call_id: str
    tool_name: str
    args_hash: str
    risk: ToolRisk
    result: ToolResult | None = None


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ToolExecutionLedger:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ToolExecutionRecord] = {}

    def decide(
        self,
        job_id: str,
        call: ToolCall,
        risk: ToolRisk,
        args_hash: str,
    ) -> ReplayDecision:
        key = (job_id, call.id)
        existing = self._records.get(key)
        if existing is None:
            self._records[key] = ToolExecutionRecord(job_id, call.id, call.name, args_hash, risk)
            return ReplayDecision.EXECUTE
        if existing.args_hash != args_hash:
            return ReplayDecision.REJECT_DUPLICATE_SIDE_EFFECT
        if risk is ToolRisk.READ_ONLY and existing.result is not None:
            return ReplayDecision.RETURN_CACHED_RESULT
        return ReplayDecision.REJECT_DUPLICATE_SIDE_EFFECT

    def record_result(self, job_id: str, call: ToolCall, result: ToolResult) -> None:
        key = (job_id, call.id)
        existing = self._records[key]
        self._records[key] = ToolExecutionRecord(
            job_id=existing.job_id,
            tool_call_id=existing.tool_call_id,
            tool_name=existing.tool_name,
            args_hash=existing.args_hash,
            risk=existing.risk,
            result=result,
        )

    def cached_result(self, job_id: str, tool_call_id: str) -> ToolResult:
        result = self._records[(job_id, tool_call_id)].result
        if result is None:
            raise ValueError("cached result is not available")
        return result
