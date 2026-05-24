from __future__ import annotations

import os
from typing import List, Optional, Callable, Dict, Any

from os_ai_llm.interfaces import LLMClient
from os_ai_llm.types import Message, ToolDescriptor
from os_ai_core.adapters.approval import DenyAllApprovalAdapter
from os_ai_core.adapters.events.callback_event_sink import CallbackEventSink
from os_ai_core.adapters.legacy.orchestrator_runner import LegacyOrchestratorRunner
from os_ai_core.adapters.llm.legacy_gateway import LegacyLLMGateway
from os_ai_core.adapters.tools.composite_tool_gateway import CompositeToolGateway
from os_ai_core.adapters.tools.local_computer_provider import LocalComputerToolProvider
from os_ai_core.application.ports.approval import ApprovalPort
from os_ai_core.application.ports.tools import ToolGateway
from os_ai_core.application.use_cases.run_agent import RunAgentCommand, RunAgentUseCase
from os_ai_core.tools.registry import ToolRegistry


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class Orchestrator:
    def __init__(
        self,
        client: LLMClient,
        tool_registry: ToolRegistry,
        tool_gateway: ToolGateway | None = None,
        approval: ApprovalPort | None = None,
        use_application_runner: bool | None = None,
    ) -> None:
        self._client = client
        self._tools = tool_registry
        self._tool_gateway = tool_gateway or CompositeToolGateway([LocalComputerToolProvider(tool_registry)])
        self._approval = approval or DenyAllApprovalAdapter()
        self._use_application_runner = (
            use_application_runner if use_application_runner is not None else _application_runner_enabled()
        )
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.last_provider_context: Optional[Dict[str, Any]] = None

    def run(
        self,
        task: str,
        tool_descriptors: List[ToolDescriptor],
        system: Optional[str],
        max_iterations: int = 30,
        *,
        cancel_token: Optional[CancelToken] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        initial_messages: Optional[List[Message]] = None,
        initial_provider_context: Optional[Dict[str, Any]] = None,
    ) -> List[Message]:
        if not self._use_application_runner:
            return self._run_legacy(
                task=task,
                tool_descriptors=tool_descriptors,
                system=system,
                max_iterations=max_iterations,
                cancel_token=cancel_token,
                on_event=on_event,
                initial_messages=initial_messages,
                initial_provider_context=initial_provider_context,
            )

        result = RunAgentUseCase(
            llm=LegacyLLMGateway(self._client),
            tools=self._tool_gateway,
            events=CallbackEventSink(on_event),
            approval=self._approval,
        ).execute(
            RunAgentCommand(
                job_id="legacy-orchestrator-run",
                task=task,
                tool_descriptors=tool_descriptors,
                system_prompt=system,
                max_iterations=max_iterations,
                cancel_token=cancel_token,
                initial_messages=initial_messages,
                initial_provider_context=initial_provider_context,
            )
        )
        self.total_input_tokens = result.input_tokens
        self.total_output_tokens = result.output_tokens
        self.last_provider_context = result.provider_context
        return result.messages

    def _run_legacy(
        self,
        task: str,
        tool_descriptors: List[ToolDescriptor],
        system: Optional[str],
        max_iterations: int,
        cancel_token: Optional[CancelToken],
        on_event: Optional[Callable[[str, Dict[str, Any]], None]],
        initial_messages: Optional[List[Message]],
        initial_provider_context: Optional[Dict[str, Any]],
    ) -> List[Message]:
        result = LegacyOrchestratorRunner(self._client, self._tools).run(
            task=task,
            tool_descriptors=tool_descriptors,
            system=system,
            max_iterations=max_iterations,
            cancel_token=cancel_token,
            on_event=on_event,
            initial_messages=initial_messages,
            initial_provider_context=initial_provider_context,
        )
        self.total_input_tokens = result.input_tokens
        self.total_output_tokens = result.output_tokens
        self.last_provider_context = result.provider_context
        return result.messages


def _application_runner_enabled() -> bool:
    return os.environ.get("OS_AI_USE_APPLICATION_RUNNER", "1").lower() not in {"0", "false", "no", "off"}
