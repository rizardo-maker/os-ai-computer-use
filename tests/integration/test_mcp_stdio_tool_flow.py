from __future__ import annotations

import importlib.util
import sys

import pytest

from os_ai_llm.types import ToolCall
from os_ai_core.application.ports.tools import ToolExecutionContext
from os_ai_core.domain.tools.policies import ToolTrustLevel
from os_ai_mcp.client.config import StdioMcpServerConfig
from os_ai_mcp.client.connection import StdioMcpClientSession
from os_ai_mcp.client.runtime import McpClientRuntime
from os_ai_mcp.client.tool_provider import McpToolProvider


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None,
    reason="mcp package is not installed in this test environment",
)


def test_mcp_stdio_tool_flow_with_fake_echo_server(tmp_path) -> None:
    server = tmp_path / "echo_mcp_server.py"
    server.write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test")

@mcp.tool()
def echo(text: str) -> str:
    return text

if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    runtime = McpClientRuntime()
    runtime.start()
    session = StdioMcpClientSession(
        StdioMcpServerConfig(
            server_id="echo",
            command=sys.executable,
            args=[str(server)],
            startup_timeout_seconds=5,
            tool_timeout_seconds=5,
        ),
        runtime,
    )
    provider = McpToolProvider("echo", session, trust=ToolTrustLevel.TRUSTED_LOCAL)

    try:
        tools = provider.list_tools()
        echo_tool = next(tool for tool in tools if tool.params["mcp_raw_name"] == "echo")
        result = provider.execute(
            ToolCall(id="call-1", name=echo_tool.name, args={"text": "hello"}),
            ToolExecutionContext(job_id="job-1", timeout_seconds=5),
        )
    finally:
        session.close()
        runtime.shutdown()

    assert result.is_error is False
    assert result.content[0].text == "hello"
