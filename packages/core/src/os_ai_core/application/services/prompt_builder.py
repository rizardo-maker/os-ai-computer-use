from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class DesktopPromptContext:
    action_first: bool = False


class PromptBuilder:
    def build_desktop_operator_prompt(self, context: DesktopPromptContext | None = None) -> str:
        ctx = context or DesktopPromptContext()
        os_name = platform.system()
        if os_name == "Darwin":
            os_version = platform.mac_ver()[0]
        elif os_name == "Linux":
            os_version = platform.release()
        else:
            os_version = platform.version()
        os_label = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(os_name, os_name)
        mod_key = "cmd" if os_name == "Darwin" else "ctrl"
        shortcut_examples = f"'{mod_key}+space', '{mod_key}+c'"

        action_guidance = ""
        if ctx.action_first:
            action_guidance = (
                "When the user asks you to DO something (draw, click, type, open, navigate, etc.), "
                "use the computer tool immediately - do not describe what you would do, just act. "
                "When the user asks a question or wants information, answer in text normally. "
                "When drawing or dragging, pay attention to the visible canvas/window boundaries in the screenshot - "
                "keep all coordinates within the actual drawing area, not the full screen. "
            )

        return (
            f"You are an expert desktop operator on {os_label} {os_version}. "
            f"{action_guidance}"
            "Always complete the task fully - do NOT stop halfway to ask unnecessary questions. "
            "Only ask the user if you hit a genuine dead-end or need credentials/permissions. "
            "ONLY take a screenshot when needed. Prefer keyboard shortcuts. "
            f"NEVER send empty key combos; always include a valid key or hotkey like {shortcut_examples}. "
            f"When using key/hold_key, provide 'key' or 'keys' as a non-empty string (e.g., {shortcut_examples}). "
            "For any action with coordinates, set coordinate_space='auto' in tool input."
        )
