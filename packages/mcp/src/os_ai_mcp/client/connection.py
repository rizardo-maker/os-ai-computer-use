from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Protocol

from os_ai_mcp.client.config import StdioMcpServerConfig
from os_ai_mcp.client.runtime import McpClientRuntime


class McpClientSession(Protocol):
    def initialize(self) -> None:
        ...

    def list_tools(self) -> list[Any]:
        ...

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_seconds: float | None,
        cancel_token: object | None,
    ) -> Any:
        ...

    def close(self) -> None:
        ...


@dataclass
class McpConnectionState:
    server_id: str
    initialized: bool = False
    failed: bool = False
    last_error: str | None = None


@dataclass
class _McpOperation:
    kind: str
    future: asyncio.Future[Any]
    name: str | None = None
    arguments: dict[str, Any] | None = None


class StdioMcpClientSession:
    def __init__(self, config: StdioMcpServerConfig, runtime: McpClientRuntime) -> None:
        self._config = config
        self._runtime = runtime
        self._start_lock = threading.Lock()
        self._queue: asyncio.Queue[_McpOperation] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self.state = McpConnectionState(server_id=config.server_id)

    def initialize(self) -> None:
        self._ensure_worker_started()
        future = self._runtime.submit(self._request("initialize"))
        future.result(timeout=self._config.startup_timeout_seconds)

    def list_tools(self) -> list[Any]:
        self._ensure_worker_started()
        future = self._runtime.submit(self._request("list_tools"))
        return future.result(timeout=self._config.startup_timeout_seconds)

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_seconds: float | None,
        cancel_token: object | None,
    ) -> Any:
        if bool(getattr(cancel_token, "is_cancelled", False)):
            raise RuntimeError("tool call cancelled")
        timeout = timeout_seconds or self._config.tool_timeout_seconds
        self._ensure_worker_started()
        future = self._runtime.submit(self._request("call_tool", name=name, arguments=arguments))
        return future.result(timeout=timeout)

    def close(self) -> None:
        if self._queue is None:
            return
        future = self._runtime.submit(self._request("close"))
        future.result(timeout=2.0)

    async def _request(
        self,
        kind: str,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        queue = self._queue
        if queue is None:
            if kind == "close":
                return None
            raise RuntimeError("MCP session worker is not started")
        loop = asyncio.get_running_loop()
        result: asyncio.Future[Any] = loop.create_future()
        await queue.put(_McpOperation(kind=kind, name=name, arguments=arguments, future=result))
        return await result

    def _ensure_worker_started(self) -> None:
        if self._queue is not None:
            return
        with self._start_lock:
            if self._queue is not None:
                return
            future = self._runtime.submit(self._start_worker())
            future.result(timeout=self._config.startup_timeout_seconds)

    async def _start_worker(self) -> None:
        if self._queue is not None:
            return
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._run_worker(self._queue))

    async def _run_worker(self, queue: asyncio.Queue[_McpOperation]) -> None:
        close_future: asyncio.Future[Any] | None = None
        session: Any | None = None

        async def ensure_session(stack: AsyncExitStack) -> Any:
            nonlocal session
            if session is not None:
                return session
            session = await self._open_session(stack)
            self.state.initialized = True
            self.state.failed = False
            self.state.last_error = None
            return session

        try:
            async with AsyncExitStack() as stack:
                while True:
                    operation = await queue.get()
                    if operation.kind == "close":
                        close_future = operation.future
                        break
                    try:
                        if operation.kind == "initialize":
                            await ensure_session(stack)
                            if not operation.future.done():
                                operation.future.set_result(None)
                        elif operation.kind == "list_tools":
                            active = await ensure_session(stack)
                            response = await active.list_tools()
                            if not operation.future.done():
                                operation.future.set_result(list(getattr(response, "tools", []) or []))
                        elif operation.kind == "call_tool":
                            active = await ensure_session(stack)
                            result = await active.call_tool(operation.name or "", operation.arguments or {})
                            if not operation.future.done():
                                operation.future.set_result(result)
                        else:
                            if not operation.future.done():
                                operation.future.set_exception(RuntimeError(f"unknown MCP operation: {operation.kind}"))
                    except Exception as exc:
                        self.state.failed = True
                        self.state.last_error = str(exc)
                        if not operation.future.done():
                            operation.future.set_exception(exc)
            if close_future is not None and not close_future.done():
                close_future.set_result(None)
        except Exception as exc:
            self.state.failed = True
            self.state.last_error = str(exc)
            if close_future is not None and not close_future.done():
                close_future.set_exception(exc)
        finally:
            self.state.initialized = False
            self._queue = None
            self._worker_task = None

    async def _open_session(self, stack: AsyncExitStack) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=self._config.env or None,
            cwd=self._config.cwd,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session
