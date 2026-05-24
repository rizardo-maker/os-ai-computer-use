from __future__ import annotations

import pytest

from os_ai_mcp.server import os_ai_server


def test_os_ai_mcp_server_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OS_AI_MCP_SERVER_ENABLED", raising=False)

    with pytest.raises(PermissionError):
        os_ai_server.screenshot()


def test_os_ai_mcp_server_allows_screenshot_when_enabled(monkeypatch):
    monkeypatch.setenv("OS_AI_MCP_SERVER_ENABLED", "1")
    monkeypatch.setattr(os_ai_server, "_run_computer_action", lambda args: [{"type": "image", "source": {"data": "abc"}}])

    assert os_ai_server.screenshot() == [{"type": "image", "source": {"data": "abc"}}]


def test_os_ai_mcp_server_control_requires_explicit_policy(monkeypatch):
    monkeypatch.setenv("OS_AI_MCP_SERVER_ENABLED", "1")
    monkeypatch.delenv("OS_AI_MCP_SERVER_ALLOW_CONTROL", raising=False)

    with pytest.raises(PermissionError):
        os_ai_server.left_click()


def test_os_ai_mcp_server_control_action_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setenv("OS_AI_MCP_SERVER_ENABLED", "1")
    monkeypatch.setenv("OS_AI_MCP_SERVER_ALLOW_CONTROL", "1")
    monkeypatch.setattr(os_ai_server, "_run_computer_action", lambda args: calls.append(args) or [{"type": "text", "text": "ok"}])

    assert os_ai_server.type_text("hello") == "ok"
    assert calls == [{"action": "type", "text": "hello"}]
