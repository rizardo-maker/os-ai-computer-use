from __future__ import annotations

from os_ai_llm.config import COMPUTER_TOOL_TYPES
from os_ai_llm.types import ToolDescriptor
from os_ai_os.api import get_drivers


def build_computer_tool_descriptor(provider: str | None) -> ToolDescriptor:
    actual_provider = provider or "openai"
    size = get_drivers().screen.size()
    return ToolDescriptor(
        name="computer",
        kind="computer_use",
        params={
            "type": COMPUTER_TOOL_TYPES.get(actual_provider, "computer_20250124"),
            "display_width_px": int(size.width),
            "display_height_px": int(size.height),
        },
    )
