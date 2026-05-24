from __future__ import annotations


class ToolDomainError(Exception):
    pass


class UnknownToolError(ToolDomainError):
    pass


class DuplicateToolNameError(ToolDomainError):
    pass


class ToolPolicyError(ToolDomainError):
    pass
