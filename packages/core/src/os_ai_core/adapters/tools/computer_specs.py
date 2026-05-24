from __future__ import annotations

import pyautogui

from os_ai_llm.config import COMPUTER_TOOL_TYPES
from os_ai_llm.types import ToolDescriptor


def build_computer_tool_descriptor(provider: str | None) -> ToolDescriptor:
    actual_provider = provider or "openai"
    screen_w, screen_h = pyautogui.size()
    return ToolDescriptor(
        name="computer",
        kind="computer_use",
        params={
            "type": COMPUTER_TOOL_TYPES.get(actual_provider, "computer_20250124"),
            "display_width_px": screen_w,
            "display_height_px": screen_h,
        },
    )
