from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


AgentEventKind = Literal[
    "progress",
    "assistant_text",
    "tool_started",
    "tool_finished",
    "tool_failed",
    "usage",
    "final",
    "tool_catalog_changed",
]


@dataclass(frozen=True)
class AgentEvent:
    kind: AgentEventKind
    job_id: str
    payload: dict[str, Any]

    @staticmethod
    def progress(job_id: str, stage: str, iteration: int) -> "AgentEvent":
        return AgentEvent("progress", job_id, {"stage": stage, "iteration": iteration})

    @staticmethod
    def assistant_text(job_id: str, text: str) -> "AgentEvent":
        return AgentEvent("assistant_text", job_id, {"text": text})

    @staticmethod
    def tool_started(job_id: str, call: Any) -> "AgentEvent":
        return AgentEvent("tool_started", job_id, {"call": call})

    @staticmethod
    def tool_finished(job_id: str, result: Any) -> "AgentEvent":
        return AgentEvent("tool_finished", job_id, {"result": result})

    @staticmethod
    def tool_failed(job_id: str, call: Any, result: Any) -> "AgentEvent":
        return AgentEvent("tool_failed", job_id, {"call": call, "result": result})

    @staticmethod
    def usage(job_id: str, payload: dict[str, Any]) -> "AgentEvent":
        return AgentEvent("usage", job_id, payload)

    @staticmethod
    def tool_catalog_changed(job_id: str, version: int | None = None) -> "AgentEvent":
        return AgentEvent("tool_catalog_changed", job_id, {"version": version})
