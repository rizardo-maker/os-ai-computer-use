from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from os_ai_llm.types import LLMResponse, Message, TextPart, ToolCall, ToolDescriptor, ToolResult
from os_ai_core.application.ports.approval import ApprovalPort
from os_ai_core.application.ports.events import EventSink
from os_ai_core.application.ports.llm import LLMGateway, LLMRequest
from os_ai_core.application.ports.tools import ToolExecutionContext, ToolGateway
from os_ai_core.application.services.tool_execution_ledger import (
    ReplayDecision,
    ToolExecutionLedger,
    stable_json_hash,
)
from os_ai_core.application.services.provider_safety import (
    STOP_AGENT_LOOP_META,
    approve_provider_safety_checks,
)
from os_ai_core.config import LOGGER_NAME, USAGE_LOG_EACH_ITERATION
from os_ai_core.domain.agent.events import AgentEvent
from os_ai_core.domain.tools.models import ToolRisk
from os_ai_core.utils.costs import estimate_cost


@dataclass(frozen=True)
class RunAgentCommand:
    job_id: str
    task: str
    tool_descriptors: list[ToolDescriptor]
    system_prompt: str | None
    max_iterations: int = 30
    conversation_id: str | None = None
    cancel_token: Any | None = None
    initial_messages: list[Message] | None = None
    initial_provider_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunAgentResult:
    messages: list[Message]
    provider_context: dict[str, Any] | None
    input_tokens: int
    output_tokens: int


class RunAgentUseCase:
    def __init__(
        self,
        llm: LLMGateway,
        tools: ToolGateway,
        events: EventSink,
        ledger: ToolExecutionLedger | None = None,
        approval: ApprovalPort | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._events = events
        self._ledger = ledger or ToolExecutionLedger()
        self._approval = approval
        self._logger = logging.getLogger(LOGGER_NAME)

    def execute(self, command: RunAgentCommand) -> RunAgentResult:
        messages = self._initial_messages(command)
        provider_context = command.initial_provider_context
        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(command.max_iterations):
            if self._is_cancelled(command.cancel_token):
                self._events.emit(AgentEvent.progress(command.job_id, "cancelled", iteration))
                break

            self._events.emit(AgentEvent.progress(command.job_id, "iteration_start", iteration))
            tools = self._merge_tool_descriptors(command.tool_descriptors)
            tool_risks = self._tool_risks_by_name(tools)

            try:
                response = self._llm.generate(
                    LLMRequest(
                        messages=messages,
                        tools=tools,
                        system=command.system_prompt,
                        provider_context=provider_context,
                    )
                )
            except httpx.HTTPStatusError as exc:
                self._log_http_error(exc)
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as exc:
                self._logger.error("HTTP timeout from provider: %s", exc)
                break
            except Exception as exc:
                self._logger.error("Provider error: %s", exc)
                break

            provider_context = response.provider_context
            usage_in, usage_out = self._handle_usage(
                command.job_id,
                iteration,
                response,
                total_input_tokens,
                total_output_tokens,
            )
            total_input_tokens += usage_in
            total_output_tokens += usage_out

            self._emit_assistant_text(command.job_id, response)

            if response.messages:
                messages.extend(response.messages)

            if not response.tool_calls:
                break

            stop_requested = False
            for tool_call in response.tool_calls:
                if self._is_cancelled(command.cancel_token):
                    self._events.emit(AgentEvent.progress(command.job_id, "cancelled", iteration))
                    stop_requested = True
                    break

                result = self._execute_tool(command, tool_call, tool_risks.get(tool_call.name))
                if result.is_error:
                    self._events.emit(AgentEvent.tool_failed(command.job_id, tool_call, result))
                else:
                    self._events.emit(AgentEvent.tool_finished(command.job_id, result))

                if self._should_stop_after_tool_result(result):
                    stop_requested = True
                    break

                messages = self._llm.append_tool_result(messages, tool_call, result)

            if stop_requested:
                break

        return RunAgentResult(
            messages=messages,
            provider_context=provider_context,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def _initial_messages(self, command: RunAgentCommand) -> list[Message]:
        messages = list(command.initial_messages or [])
        messages.append(Message(role="user", content=[TextPart(text=command.task)]))
        return messages

    def _merge_tool_descriptors(self, base: list[ToolDescriptor]) -> list[ToolDescriptor]:
        merged: list[ToolDescriptor] = []
        seen: set[str] = set()
        for descriptor in [*base, *self._tools.list_tools()]:
            if descriptor.name in seen:
                continue
            seen.add(descriptor.name)
            merged.append(descriptor)
        return merged

    def _tool_risks_by_name(self, tools: list[ToolDescriptor]) -> dict[str, ToolRisk]:
        risks: dict[str, ToolRisk] = {}
        for descriptor in tools:
            raw = descriptor.params.get("risk")
            if not raw:
                continue
            try:
                risks[descriptor.name] = ToolRisk(str(raw))
            except Exception:
                continue
        return risks

    def _execute_tool(
        self,
        command: RunAgentCommand,
        tool_call: ToolCall,
        catalog_risk: ToolRisk | None,
    ) -> ToolResult:
        self._events.emit(AgentEvent.tool_started(command.job_id, tool_call))
        args_hash = stable_json_hash(tool_call.args)
        risk = self._risk_for(tool_call, catalog_risk)
        decision = self._ledger.decide(command.job_id, tool_call, risk, args_hash)

        if decision is ReplayDecision.RETURN_CACHED_RESULT:
            return self._ledger.cached_result(command.job_id, tool_call.id)
        if decision is ReplayDecision.REJECT_DUPLICATE_SIDE_EFFECT:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=[TextPart(text="error: duplicate side-effecting tool call rejected")],
                is_error=True,
                metadata={"error_code": "duplicate_tool_call", "provider_tool_type": "function"},
            )

        safety_denial = approve_provider_safety_checks(
            job_id=command.job_id,
            tool_call=tool_call,
            risk=risk,
            approval=self._approval,
        )
        if safety_denial is not None:
            self._ledger.record_result(command.job_id, tool_call, safety_denial)
            return safety_denial

        context = ToolExecutionContext(
            job_id=command.job_id,
            conversation_id=command.conversation_id,
            cancel_token=command.cancel_token,
            approval=self._approval,
        )
        result = self._tools.execute(tool_call, context)
        if tool_call.name != "computer":
            result.metadata.setdefault("provider_tool_type", "function")
        self._ledger.record_result(command.job_id, tool_call, result)
        return result

    def _risk_for(self, tool_call: ToolCall, catalog_risk: ToolRisk | None) -> ToolRisk:
        raw = tool_call.metadata.get("risk")
        try:
            return ToolRisk(str(raw)) if raw else (catalog_risk or ToolRisk.LOCAL_MUTATION)
        except Exception:
            return catalog_risk or ToolRisk.LOCAL_MUTATION

    def _should_stop_after_tool_result(self, result: ToolResult) -> bool:
        return result.metadata.get(STOP_AGENT_LOOP_META) is True

    def _emit_assistant_text(self, job_id: str, response: LLMResponse) -> None:
        seen: set[str] = set()
        for message in response.messages or []:
            if getattr(message, "role", None) != "assistant":
                continue
            for part in getattr(message, "content", []) or []:
                if not isinstance(part, TextPart):
                    continue
                text = part.text.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                self._logger.info("🧠 %s", text)
                self._events.emit(AgentEvent.assistant_text(job_id, text))

    def _handle_usage(
        self,
        job_id: str,
        iteration: int,
        response: LLMResponse,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> tuple[int, int]:
        try:
            input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        except Exception:
            return 0, 0

        model_name = self._safe_model_name()
        input_cost, output_cost, total_cost, _tier = estimate_cost(model_name, input_tokens, output_tokens)
        self._events.emit(
            AgentEvent.usage(
                job_id,
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "iteration": iteration,
                    "total_input_tokens": total_input_tokens + input_tokens,
                    "total_output_tokens": total_output_tokens + output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                },
            )
        )
        if USAGE_LOG_EACH_ITERATION:
            self._logger.info(
                "📈 Usage iter in=%s out=%s cost=$%.6f (input=$%.6f, output=$%.6f)",
                input_tokens,
                output_tokens,
                input_cost + output_cost,
                input_cost,
                output_cost,
            )
        return input_tokens, output_tokens

    def _safe_model_name(self) -> str:
        try:
            return self._llm.model_name()
        except Exception:
            return "unknown"

    def _log_http_error(self, exc: httpx.HTTPStatusError) -> None:
        status = getattr(exc.response, "status_code", None)
        provider = "provider"
        try:
            provider = self._llm.provider_name()
        except Exception:
            pass
        try:
            body: Any = exc.response.json()
        except Exception:
            body = exc.response.text
        self._logger.error("HTTP %s from %s: %s", status, provider, body)

    def _is_cancelled(self, cancel_token: Any | None) -> bool:
        return bool(getattr(cancel_token, "is_cancelled", False))
