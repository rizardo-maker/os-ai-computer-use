from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2] / "packages"


RULES = {
    "core/src/os_ai_core/domain": {"mcp", "openai", "anthropic", "os_ai_backend", "os_ai_cli"},
    "core/src/os_ai_core/application": {"mcp", "openai", "anthropic", "os_ai_backend", "os_ai_cli"},
    "llm_openai/src/os_ai_llm_openai": {"os_ai_mcp", "os_ai_backend", "os_ai_cli"},
    "llm_anthropic/src/os_ai_llm_anthropic": {"os_ai_mcp", "os_ai_backend", "os_ai_cli"},
    "mcp/src/os_ai_mcp": {"openai", "anthropic", "os_ai_backend", "os_ai_cli"},
}


@pytest.mark.parametrize(("relative_dir", "forbidden"), RULES.items())
def test_forbidden_imports_do_not_cross_package_boundaries(relative_dir: str, forbidden: set[str]) -> None:
    root = ROOT / relative_dir
    assert root.exists()

    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for module in forbidden:
            assert f"import {module}" not in text, path
            assert f"from {module}" not in text, path


def test_orchestrator_facade_does_not_read_openai_private_metadata() -> None:
    path = ROOT / "core/src/os_ai_core/orchestrator.py"
    text = path.read_text(encoding="utf-8")

    assert "_openai_" not in text
