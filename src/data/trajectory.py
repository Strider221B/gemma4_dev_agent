"""Data model representing a full agent trajectory for a task."""

from dataclasses import dataclass

from src.data.complexity_tier import ComplexityTier
from src.data.turn import Turn


@dataclass
class Trajectory:
    """Represents a complete multi-turn interaction trajectory."""

    instance_id: str
    repo: str
    complexity: ComplexityTier
    turns: list[Turn]
    token_count: int
    num_tool_calls: int
    num_files_changed: int
    patch: str | None = None
    had_truncation: bool = False
    elapsed_seconds: float = 0.0
