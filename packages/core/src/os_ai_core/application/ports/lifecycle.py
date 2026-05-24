from __future__ import annotations

from typing import Protocol


class ManagedRuntime(Protocol):
    def start(self) -> None:
        ...

    def shutdown(self, timeout_seconds: float = 2.0) -> None:
        ...
