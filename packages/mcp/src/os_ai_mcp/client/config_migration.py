from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CURRENT_CONFIG_VERSION = 1


@dataclass(frozen=True)
class ConfigMigrationResult:
    raw: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class McpConfigMigrator:
    def migrate(self, raw: dict[str, Any]) -> ConfigMigrationResult:
        version = int(raw.get("version", 0))
        if version == CURRENT_CONFIG_VERSION:
            return ConfigMigrationResult(raw)
        if version == 0:
            migrated = dict(raw)
            migrated["version"] = CURRENT_CONFIG_VERSION
            servers = dict(migrated.get("servers") or {})
            for server in servers.values():
                if isinstance(server, dict):
                    server.setdefault("trust", "local_untrusted")
                    server.setdefault("enabled", False)
            migrated["servers"] = servers
            return ConfigMigrationResult(
                migrated,
                warnings=["mcp_config_legacy_v0_migrated_disabled_by_default"],
            )
        return ConfigMigrationResult(
            {"version": CURRENT_CONFIG_VERSION, "servers": {}},
            warnings=["mcp_config_future_version_disabled"],
        )
