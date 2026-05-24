from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolTrustLevel(str, Enum):
    TRUSTED_LOCAL = "trusted_local"
    LOCAL_UNTRUSTED = "local_untrusted"
    REMOTE_UNTRUSTED = "remote_untrusted"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ToolPolicyDecision:
    allowed: bool
    requires_approval: bool = False
    reason: str | None = None
