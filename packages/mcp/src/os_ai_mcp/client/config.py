from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from os_ai_core.domain.tools.policies import ToolTrustLevel
from os_ai_mcp.client.config_migration import McpConfigMigrator


class McpTransportKind(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


@dataclass(frozen=True)
class McpServerPolicyConfig:
    allow_tools: frozenset[str] = frozenset()
    deny_tools: frozenset[str] = frozenset()
    max_output_chars: int = 12000


@dataclass(frozen=True)
class StdioMcpServerConfig:
    server_id: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    trust: ToolTrustLevel = ToolTrustLevel.LOCAL_UNTRUSTED
    startup_timeout_seconds: float = 10.0
    tool_timeout_seconds: float = 30.0
    policy: McpServerPolicyConfig = field(default_factory=McpServerPolicyConfig)


@dataclass(frozen=True)
class McpConfigLoadResult:
    servers: list[StdioMcpServerConfig]
    warnings: list[str] = field(default_factory=list)


def default_config_path() -> Path:
    raw = os.environ.get("OS_AI_MCP_CONFIG_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".config" / "os-ai" / "mcp-servers.toml"


class McpConfigLoader:
    def __init__(self, migrator: McpConfigMigrator | None = None) -> None:
        self._migrator = migrator or McpConfigMigrator()
        self._allowed_commands = self._load_allowed_commands()

    def load(self, path: Path) -> McpConfigLoadResult:
        if not path.exists():
            return McpConfigLoadResult(servers=[], warnings=["mcp_config_missing"])
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return McpConfigLoadResult(servers=[], warnings=["mcp_config_parse_error"])

        migrated = self._migrator.migrate(raw)
        servers: list[StdioMcpServerConfig] = []
        warnings = list(migrated.warnings)
        for server_id, item in (migrated.raw.get("servers") or {}).items():
            if not isinstance(item, dict):
                warnings.append(f"mcp_server_invalid:{server_id}")
                continue
            try:
                server = self._parse_stdio_server(str(server_id), item)
            except ValueError as exc:
                warnings.append(f"mcp_server_invalid:{server_id}:{exc}")
                continue
            if server.enabled:
                servers.append(server)
        return McpConfigLoadResult(servers=servers, warnings=warnings)

    def _parse_stdio_server(self, server_id: str, item: dict[str, Any]) -> StdioMcpServerConfig:
        if not server_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("bad_server_id")
        transport = item.get("transport", "stdio")
        if transport != McpTransportKind.STDIO.value:
            raise ValueError("unsupported_transport")
        command = item.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("missing_command")
        self._validate_command(command)
        args = item.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError("bad_args")
        env = item.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError("bad_env")
        policy = self._parse_policy(item.get("policy") or {})
        return StdioMcpServerConfig(
            server_id=server_id,
            command=command,
            args=list(args),
            env=dict(env),
            cwd=item.get("cwd") if isinstance(item.get("cwd"), str) else None,
            enabled=bool(item.get("enabled", True)),
            trust=ToolTrustLevel(str(item.get("trust", ToolTrustLevel.LOCAL_UNTRUSTED.value))),
            startup_timeout_seconds=float(item.get("startup_timeout_seconds", 10)),
            tool_timeout_seconds=float(item.get("tool_timeout_seconds", 30)),
            policy=policy,
        )

    def _validate_command(self, command: str) -> None:
        if command.strip() != command or any(char in command for char in ("\x00", "\n", "\r")):
            raise ValueError("bad_command")
        if not self._allowed_commands:
            return
        command_name = Path(command).name
        if command not in self._allowed_commands and command_name not in self._allowed_commands:
            raise ValueError("command_not_allowlisted")

    def _load_allowed_commands(self) -> frozenset[str]:
        raw = os.environ.get("OS_AI_MCP_ALLOWED_COMMANDS", "")
        return frozenset(item.strip() for item in raw.split(",") if item.strip())

    def _parse_policy(self, raw: dict[str, Any]) -> McpServerPolicyConfig:
        allow = raw.get("allow_tools", [])
        deny = raw.get("deny_tools", [])
        return McpServerPolicyConfig(
            allow_tools=frozenset(str(x) for x in allow if isinstance(x, str)),
            deny_tools=frozenset(str(x) for x in deny if isinstance(x, str)),
            max_output_chars=int(raw.get("max_output_chars", 12000)),
        )
