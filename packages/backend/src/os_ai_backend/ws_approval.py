from __future__ import annotations

import asyncio
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import WebSocket

from os_ai_core.application.ports.approval import ApprovalDecision, ApprovalPort, ApprovalRequest


SendEvent = Callable[[WebSocket, str, dict[str, Any]], Awaitable[None]]


@dataclass
class _PendingApproval:
    job_id: str
    request_id: str
    event: threading.Event
    decision: ApprovalDecision = ApprovalDecision.UNAVAILABLE


class WebSocketApprovalAdapter(ApprovalPort):
    def __init__(
        self,
        websocket: WebSocket,
        loop: asyncio.AbstractEventLoop,
        job_id: str,
        send_event: SendEvent,
    ) -> None:
        self._websocket = websocket
        self._loop = loop
        self._job_id = job_id
        self._send_event = send_event
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = threading.Lock()

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        approval_id = str(uuid.uuid4())
        pending = _PendingApproval(job_id=request.job_id, request_id=approval_id, event=threading.Event())
        with self._lock:
            self._pending[approval_id] = pending

        payload = {
            "jobId": self._job_id,
            "approvalId": approval_id,
            "summary": request.summary,
            "risk": request.risk.value,
            "tool": {
                "name": request.tool_call.name,
                "args": request.tool_call.args,
                "metadata": request.tool_call.metadata,
            },
            "expiresInSeconds": request.expires_in_seconds,
        }

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._send_event(self._websocket, "event.approval", payload),
                self._loop,
            )
            future.result(timeout=2.0)
        except Exception:
            self._drop(approval_id)
            return ApprovalDecision.UNAVAILABLE

        timeout_seconds = min(
            int(request.expires_in_seconds),
            _approval_timeout_from_env(request.expires_in_seconds),
        )
        if timeout_seconds <= 0:
            self._drop(approval_id)
            return ApprovalDecision.EXPIRED

        if not pending.event.wait(timeout_seconds):
            self._drop(approval_id)
            return ApprovalDecision.EXPIRED
        self._drop(approval_id)
        return pending.decision

    def respond(self, approval_id: str, approved: bool) -> bool:
        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None:
                return False
            pending.decision = ApprovalDecision.APPROVED if approved else ApprovalDecision.DENIED
            pending.event.set()
            return True

    def cancel_all(self) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.decision = ApprovalDecision.EXPIRED
            item.event.set()

    def _drop(self, approval_id: str) -> None:
        with self._lock:
            self._pending.pop(approval_id, None)


def _approval_timeout_from_env(default: int) -> int:
    raw = os.environ.get("OS_AI_APPROVAL_TIMEOUT_SECONDS")
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)
