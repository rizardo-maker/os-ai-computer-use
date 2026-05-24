from __future__ import annotations

import logging

from os_ai_core.config import LOGGER_NAME
from os_ai_mcp.client import factory


class FakeRuntime:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True


class FakeSession:
    def __init__(self, config, runtime) -> None:
        self.config = config
        self.runtime = runtime

    def initialize(self) -> None:
        return

    def list_tools(self):
        return []

    def call_tool(self, name, arguments, timeout_seconds, cancel_token):
        raise AssertionError("not used")

    def close(self) -> None:
        return


def test_mcp_factory_logs_exact_command_without_env(monkeypatch, tmp_path, caplog):
    config_path = tmp_path / "mcp-servers.toml"
    config_path.write_text(
        """
version = 1

[servers.echo]
transport = "stdio"
command = "python"
args = ["server.py", "--stdio"]
env = { SECRET_TOKEN = "should-not-log" }
enabled = true
trust = "trusted_local"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OS_AI_MCP_ENABLED", "1")
    monkeypatch.setenv("OS_AI_MCP_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(factory, "McpClientRuntime", FakeRuntime)
    monkeypatch.setattr(factory, "StdioMcpClientSession", FakeSession)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        providers = factory.create_mcp_tool_providers_from_env()

    assert len(providers) == 1
    assert "server_id=echo" in caplog.text
    assert "command='python'" in caplog.text
    assert "args=['server.py', '--stdio']" in caplog.text
    assert "SECRET_TOKEN" not in caplog.text
    assert "should-not-log" not in caplog.text
