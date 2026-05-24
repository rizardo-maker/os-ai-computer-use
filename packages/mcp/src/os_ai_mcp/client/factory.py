from __future__ import annotations

import logging
import os

from os_ai_core.application.ports.tools import ToolProvider
from os_ai_core.config import LOGGER_NAME
from os_ai_mcp.client.config import McpConfigLoader, default_config_path
from os_ai_mcp.client.connection import StdioMcpClientSession
from os_ai_mcp.client.runtime import McpClientRuntime
from os_ai_mcp.client.tool_provider import McpToolProvider


def create_mcp_tool_providers_from_env() -> list[ToolProvider]:
    if os.environ.get("OS_AI_MCP_ENABLED", "0").lower() not in {"1", "true", "yes", "on"}:
        return []

    loaded = McpConfigLoader().load(default_config_path())
    if not loaded.servers:
        return []

    logger = logging.getLogger(LOGGER_NAME)
    runtime = McpClientRuntime()
    runtime.start()
    providers: list[ToolProvider] = []
    for server in loaded.servers:
        logger.info(
            "MCP stdio server configured: server_id=%s command=%r args=%r cwd=%r trust=%s",
            server.server_id,
            server.command,
            server.args,
            server.cwd,
            server.trust.value,
        )
        session = StdioMcpClientSession(server, runtime)
        providers.append(
            McpToolProvider(
                server_id=server.server_id,
                session=session,
                trust=server.trust,
                policy=server.policy,
            )
        )
    return providers
