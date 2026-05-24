from __future__ import annotations

from os_ai_mcp.client.config import McpConfigLoader
from os_ai_mcp.client.config_migration import McpConfigMigrator


def test_mcp_config_v0_migrates_disabled_by_default():
    result = McpConfigMigrator().migrate(
        {
            "servers": {
                "fs": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["server"],
                }
            }
        }
    )

    assert result.raw["version"] == 1
    assert result.raw["servers"]["fs"]["enabled"] is False
    assert result.raw["servers"]["fs"]["trust"] == "local_untrusted"
    assert result.warnings == ["mcp_config_legacy_v0_migrated_disabled_by_default"]


def test_mcp_config_future_version_disables_mcp():
    result = McpConfigMigrator().migrate({"version": 999, "servers": {"fs": {}}})

    assert result.raw == {"version": 1, "servers": {}}
    assert result.warnings == ["mcp_config_future_version_disabled"]


def test_mcp_config_loader_applies_optional_command_allowlist(monkeypatch, tmp_path):
    path = tmp_path / "mcp-servers.toml"
    path.write_text(
        """
version = 1

[servers.echo]
transport = "stdio"
command = "python"
args = ["server.py"]
enabled = true
trust = "trusted_local"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OS_AI_MCP_ALLOWED_COMMANDS", "uv,npx")

    result = McpConfigLoader().load(path)

    assert result.servers == []
    assert result.warnings == ["mcp_server_invalid:echo:command_not_allowlisted"]


def test_mcp_config_loader_accepts_allowlisted_command_basename(monkeypatch, tmp_path):
    path = tmp_path / "mcp-servers.toml"
    path.write_text(
        """
version = 1

[servers.echo]
transport = "stdio"
command = "/usr/local/bin/python"
args = ["server.py"]
enabled = true
trust = "trusted_local"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OS_AI_MCP_ALLOWED_COMMANDS", "python")

    result = McpConfigLoader().load(path)

    assert [server.server_id for server in result.servers] == ["echo"]
    assert result.warnings == []
