from __future__ import annotations

from pathlib import Path

from os_ai_llm.types import TextPart, ToolCall
from os_ai_core.adapters.tools.local_computer_provider import LocalComputerToolProvider
from os_ai_core.application.ports.tools import ToolExecutionContext
from os_ai_core.tools.registry import ToolRegistry


def test_run_agent_use_case_does_not_read_openai_private_metadata():
    path = Path("packages/core/src/os_ai_core/application/use_cases/run_agent.py")
    text = path.read_text(encoding="utf-8")

    assert "_openai_" not in text


def test_local_computer_provider_preserves_provider_metadata_for_llm_adapter():
    captured_args = {}
    registry = ToolRegistry()

    def handler(args):
        captured_args.update(args)
        return [{"type": "text", "text": "ok"}]

    registry.register("computer", handler)
    provider = LocalComputerToolProvider(registry)

    result = provider.execute(
        ToolCall(
            id="1",
            name="computer",
            args={"action": "screenshot"},
            metadata={"_openai_pending_safety_checks": [{"id": "safe"}]},
        ),
        ToolExecutionContext(job_id="job"),
    )

    assert isinstance(result.content[0], TextPart)
    assert result.metadata["_openai_pending_safety_checks"] == [{"id": "safe"}]
    assert "_openai_pending_safety_checks" not in captured_args


def test_local_computer_provider_keeps_openai_batch_bridge_in_adapter_only():
    captured_args = {}
    registry = ToolRegistry()

    def handler(args):
        captured_args.update(args)
        return [{"type": "text", "text": "ok"}]

    registry.register("computer", handler)
    provider = LocalComputerToolProvider(registry)

    provider.execute(
        ToolCall(
            id="1",
            name="computer",
            args={"action": "screenshot"},
            metadata={
                "_openai_batch": True,
                "_openai_actions": [{"action": "screenshot"}],
                "_openai_pending_safety_checks": [{"id": "safe"}],
            },
        ),
        ToolExecutionContext(job_id="job"),
    )

    assert captured_args["_openai_batch"] is True
    assert captured_args["_openai_actions"] == [{"action": "screenshot"}]
    assert "_openai_pending_safety_checks" not in captured_args


def test_local_computer_provider_can_restore_legacy_metadata_merge_with_flag():
    captured_args = {}
    registry = ToolRegistry()

    def handler(args):
        captured_args.update(args)
        return [{"type": "text", "text": "ok"}]

    registry.register("computer", handler)
    provider = LocalComputerToolProvider(registry, strict_provider_metadata=False)

    provider.execute(
        ToolCall(
            id="1",
            name="computer",
            args={"action": "screenshot"},
            metadata={"_openai_pending_safety_checks": [{"id": "safe"}]},
        ),
        ToolExecutionContext(job_id="job"),
    )

    assert captured_args["_openai_pending_safety_checks"] == [{"id": "safe"}]
