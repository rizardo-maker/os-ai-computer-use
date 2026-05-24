from __future__ import annotations

from os_ai_core.domain.tools.models import ToolRisk
from os_ai_core.domain.tools.policies import ToolTrustLevel
from os_ai_core.adapters.tools.policy import DefaultToolPolicy
from os_ai_llm.types import ToolCall
from os_ai_mcp.client.trust_policy import McpTrustPolicy


def test_mcp_trust_policy_blocks_disabled_server() -> None:
    policy = McpTrustPolicy(
        trust=ToolTrustLevel.DISABLED,
        allow_tools=frozenset(),
        deny_tools=frozenset(),
    )

    decision = policy.decide("read_file", ToolRisk.READ_ONLY)

    assert decision.allowed is False


def test_mcp_trust_policy_requires_approval_for_untrusted_mutation() -> None:
    policy = McpTrustPolicy(
        trust=ToolTrustLevel.LOCAL_UNTRUSTED,
        allow_tools=frozenset(),
        deny_tools=frozenset(),
    )

    decision = policy.decide("write_file", ToolRisk.LOCAL_MUTATION)

    assert decision.allowed is True
    assert decision.requires_approval is True


def test_mcp_trust_policy_honors_allowlist_and_blocks_code_execution() -> None:
    policy = McpTrustPolicy(
        trust=ToolTrustLevel.TRUSTED_LOCAL,
        allow_tools=frozenset({"read_file", "run_shell"}),
        deny_tools=frozenset(),
    )

    allowed = policy.decide("read_file", ToolRisk.READ_ONLY)
    missing = policy.decide("list_files", ToolRisk.READ_ONLY)
    blocked = policy.decide("run_shell", ToolRisk.CODE_EXECUTION)

    assert allowed.allowed is True
    assert missing.allowed is False
    assert blocked.allowed is False


def test_default_tool_policy_keeps_computer_permissive_and_requires_approval_for_mutations() -> None:
    policy = DefaultToolPolicy()

    computer = policy.decide(ToolCall(id="1", name="computer", args={}), ToolRisk.PRIVILEGED_OS)
    read_only = policy.decide(ToolCall(id="2", name="read_file", args={}), ToolRisk.READ_ONLY)
    mutation = policy.decide(ToolCall(id="3", name="write_file", args={}), ToolRisk.LOCAL_MUTATION)

    assert computer.allowed is True
    assert computer.requires_approval is False
    assert read_only.allowed is True
    assert read_only.requires_approval is False
    assert mutation.allowed is True
    assert mutation.requires_approval is True
