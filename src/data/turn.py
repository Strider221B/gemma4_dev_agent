"""Data model representing a single turn in a conversational trajectory."""

from dataclasses import dataclass, field

from src.data.tool_call import ToolCall


@dataclass(frozen=True)
class Turn:
    """Represents an interaction turn for either user or model."""

    role: str
    thought: str | None = None
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: str | None = None
