from __future__ import annotations

import pytest

from os_ai_mcp.client.runtime import McpClientRuntime


async def _return_value(value: int) -> int:
    return value


def test_mcp_runtime_runs_coroutines_on_dedicated_loop() -> None:
    runtime = McpClientRuntime()
    runtime.start()

    try:
        result = runtime.submit(_return_value(42)).result(timeout=1)
    finally:
        runtime.shutdown()

    assert result == 42


def test_mcp_runtime_rejects_submit_after_shutdown() -> None:
    runtime = McpClientRuntime()
    runtime.start()
    runtime.shutdown()
    coroutine = _return_value(42)

    with pytest.raises(RuntimeError):
        runtime.submit(coroutine)

    coroutine.close()
