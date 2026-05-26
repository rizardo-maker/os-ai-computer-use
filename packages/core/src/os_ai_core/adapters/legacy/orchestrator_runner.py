from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from os_ai_llm.interfaces import LLMClient
from os_ai_llm.types import ImagePart, Message, TextPart, ToolCall, ToolDescriptor
from os_ai_core.application.ports.approval import ApprovalPort
from os_ai_core.application.ports.tools import ToolExecutionContext, ToolGateway
from os_ai_core.application.services.provider_safety import STOP_AGENT_LOOP_META, approve_provider_safety_checks
from os_ai_core.config import LOGGER_NAME, USAGE_LOG_EACH_ITERATION
from os_ai_core.domain.tools.models import ToolRisk
from os_ai_core.utils.costs import estimate_cost


@dataclass(frozen=True)
class LegacyRunResult:
    messages: list[Message]
    provider_context: dict[str, Any] | None
    input_tokens: int
    output_tokens: int


class LegacyOrchestratorRunner:
    def __init__(
        self,
        client: LLMClient,
        tools: ToolGateway,
        approval: ApprovalPort | None = None,
        job_id: str = "legacy-orchestrator-run",
    ) -> None:
        self._client = client
        self._tools = tools
        self._approval = approval
        self._job_id = job_id
        self._logger = logging.getLogger(LOGGER_NAME)

    def run(
        self,
        task: str,
        tool_descriptors: list[ToolDescriptor],
        system: str | None,
        max_iterations: int,
        cancel_token: Any | None,
        on_event: Callable[[str, dict[str, Any]], None] | None,
        initial_messages: list[Message] | None,
        initial_provider_context: dict[str, Any] | None,
    ) -> LegacyRunResult:
        messages = list(initial_messages or [])
        messages.append(Message(role="user", content=[TextPart(text=task)]))
        provider_context = initial_provider_context
        total_input_tokens = 0
        total_output_tokens = 0
        tool_risks = self._tool_risks_by_name(tool_descriptors)

        for iteration in range(max_iterations):
            if self._is_cancelled(cancel_token):
                self._safe_emit(on_event, "progress", {"stage": "cancelled", "iteration": iteration})
                break

            self._safe_emit(on_event, "progress", {"stage": "iteration_start", "iteration": iteration})
            try:
                response = self._client.generate(
                    messages=messages,
                    tools=tool_descriptors,
                    system=system,
                    provider_context=provider_context,
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
            usage_in, usage_out = self._handle_usage(on_event, iteration, response, total_input_tokens, total_output_tokens)
            total_input_tokens += usage_in
            total_output_tokens += usage_out

            self._emit_assistant_text(on_event, response)
            if response.messages:
                messages.extend(response.messages)
            if not response.tool_calls:
                break

            stop_requested = False
            for call in response.tool_calls:
                if self._is_cancelled(cancel_token):
                    stop_requested = True
                    break
                self._emit_tool_call(on_event, call)
                safety_denial = approve_provider_safety_checks(
                    job_id=self._job_id,
                    tool_call=call,
                    risk=self._risk_for(call, tool_risks.get(call.name)),
                    approval=self._approval,
                )
                if safety_denial is not None:
                    result = safety_denial
                else:
                    result = self._tools.execute(
                        call,
                        ToolExecutionContext(
                            job_id=self._job_id,
                            cancel_token=cancel_token,
                            approval=self._approval,
                        ),
                    )
                self._emit_tool_result(on_event, result)
                if result.metadata.get(STOP_AGENT_LOOP_META) is True:
                    stop_requested = True
                    break
                messages.append(self._client.format_tool_result(result))

            if stop_requested:
                break

        return LegacyRunResult(
            messages=messages,
            provider_context=provider_context,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def _emit_tool_call(self, on_event: Callable[[str, dict[str, Any]], None] | None, call: ToolCall) -> None:
        actions = call.metadata.get("_openai_actions")
        if isinstance(actions, list) and len(actions) > 1:
            for action in actions:
                self._safe_emit(on_event, "tool_call", {"name": call.name, "args": action})
            return
        self._safe_emit(on_event, "tool_call", {"name": call.name, "args": call.args})

    def _emit_tool_result(self, on_event: Callable[[str, dict[str, Any]], None] | None, result: Any) -> None:
        has_image = any(isinstance(part, ImagePart) for part in result.content)
        if has_image:
            for part in result.content:
                if isinstance(part, ImagePart):
                    self._safe_emit(
                        on_event,
                        "tool_result_image",
                        {"media_type": part.media_type, "data": part.data_base64},
                    )
            return
        for part in result.content:
            if isinstance(part, TextPart):
                self._safe_emit(on_event, "tool_result_text", {"text": part.text})
                return

    def _emit_assistant_text(self, on_event: Callable[[str, dict[str, Any]], None] | None, response: Any) -> None:
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
                self._safe_emit(on_event, "assistant_text", {"text": text})

    def _handle_usage(
        self,
        on_event: Callable[[str, dict[str, Any]], None] | None,
        iteration: int,
        response: Any,
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
        self._safe_emit(
            on_event,
            "usage",
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

    def _log_http_error(self, exc: httpx.HTTPStatusError) -> None:
        status = getattr(exc.response, "status_code", None)
        provider = "provider"
        try:
            provider = self._client.get_provider_name()
        except Exception:
            pass
        try:
            body: Any = exc.response.json()
        except Exception:
            body = exc.response.text
        self._logger.error("HTTP %s from %s: %s", status, provider, body)

    def _safe_model_name(self) -> str:
        try:
            return self._client.get_model_name()
        except Exception:
            return "unknown"

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

    def _risk_for(self, call: ToolCall, catalog_risk: ToolRisk | None) -> ToolRisk:
        raw = call.metadata.get("risk")
        try:
            return ToolRisk(str(raw)) if raw else (catalog_risk or ToolRisk.LOCAL_MUTATION)
        except Exception:
            return catalog_risk or ToolRisk.LOCAL_MUTATION

    def _safe_emit(
        self,
        on_event: Callable[[str, dict[str, Any]], None] | None,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        if on_event is None:
            return
        try:
            on_event(kind, payload)
        except Exception:
            return

    def _is_cancelled(self, cancel_token: Any | None) -> bool:
        return bool(getattr(cancel_token, "is_cancelled", False))
