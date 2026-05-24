from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    LOCAL_MUTATION = "local_mutation"
    EXTERNAL_MUTATION = "external_mutation"
    CODE_EXECUTION = "code_execution"
    PRIVILEGED_OS = "privileged_os"


@dataclass(frozen=True)
class ToolProviderId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("tool provider id must not be empty")


@dataclass(frozen=True)
class ToolName:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("tool name must not be empty")


@dataclass(frozen=True)
class ToolSpecMetadata:
    risk: ToolRisk = ToolRisk.LOCAL_MUTATION
    provider_id: str = ""
    raw_name: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
