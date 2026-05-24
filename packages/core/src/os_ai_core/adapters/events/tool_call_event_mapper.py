from __future__ import annotations

from typing import Any

from os_ai_llm.types import ToolCall


class ToolCallEventMapper:
    def map_started(self, call: ToolCall) -> list[dict[str, Any]]:
        batch_actions = call.metadata.get("_openai_actions")
        if isinstance(batch_actions, list) and len(batch_actions) > 1:
            return [{"name": call.name, "args": action} for action in batch_actions]
        return [{"name": call.name, "args": call.args}]
