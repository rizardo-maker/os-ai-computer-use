from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Coroutine, TypeVar


T = TypeVar("T")


class McpClientRuntime:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="os-ai-mcp-loop", daemon=True)
        self._stopped = threading.Event()

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._stopped.set()

    def submit(self, coro: Coroutine[object, object, T]) -> Future[T]:
        if self._stopped.is_set():
            raise RuntimeError("MCP runtime is stopped")
        if not self._thread.is_alive():
            raise RuntimeError("MCP runtime is not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self, timeout_seconds: float = 2.0) -> None:
        if self._stopped.is_set():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout_seconds)
