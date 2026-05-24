from __future__ import annotations

import os
from typing import Any


def get_capabilities() -> dict[str, Any]:
    return {
        "server": "os-ai",
        "transport": "stdio",
        "tools": [
            "computer.screenshot",
            "computer.mouse_move",
            "computer.left_click",
            "computer.type",
            "system.get_capabilities",
        ],
        "control_enabled": _env_enabled("OS_AI_MCP_SERVER_ALLOW_CONTROL"),
    }


def screenshot() -> list[dict[str, Any]]:
    _require_server_enabled()
    return _run_computer_action({"action": "screenshot"})


def mouse_move(x: int, y: int) -> str:
    _require_control_enabled("computer.mouse_move")
    _run_computer_action({"action": "mouse_move", "x": int(x), "y": int(y), "coordinate_space": "auto"})
    return "ok"


def left_click(x: int | None = None, y: int | None = None) -> str:
    _require_control_enabled("computer.left_click")
    args: dict[str, Any] = {"action": "left_click", "coordinate_space": "auto"}
    if x is not None:
        args["x"] = int(x)
    if y is not None:
        args["y"] = int(y)
    _run_computer_action(args)
    return "ok"


def type_text(text: str) -> str:
    _require_control_enabled("computer.type")
    _run_computer_action({"action": "type", "text": text})
    return "ok"


def build_server() -> Any:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("OS AI")
    mcp.tool(name="system.get_capabilities")(get_capabilities)
    mcp.tool(name="computer.screenshot")(screenshot)
    mcp.tool(name="computer.mouse_move")(mouse_move)
    mcp.tool(name="computer.left_click")(left_click)
    mcp.tool(name="computer.type")(type_text)
    return mcp


def main() -> None:
    _require_server_enabled()
    build_server().run(transport="stdio")


def _run_computer_action(args: dict[str, Any]) -> list[dict[str, Any]]:
    from os_ai_core.tools.computer import computer_tool_handler

    return computer_tool_handler(args)


def _require_server_enabled() -> None:
    if not _env_enabled("OS_AI_MCP_SERVER_ENABLED"):
        raise PermissionError("OS AI MCP server is disabled")


def _require_control_enabled(tool_name: str) -> None:
    _require_server_enabled()
    if not _env_enabled("OS_AI_MCP_SERVER_ALLOW_CONTROL"):
        raise PermissionError(f"{tool_name} is disabled by policy")


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
