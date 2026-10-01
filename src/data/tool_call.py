"""Data model representing a tool call executed by an agent."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """Represents a structured tool invocation with its arguments."""

    tool_name: str
    args: dict[str, object]
