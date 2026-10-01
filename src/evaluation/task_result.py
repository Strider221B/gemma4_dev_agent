"""Data model representing evaluation outcome for a single task."""

from dataclasses import dataclass

from src.data.complexity_tier import ComplexityTier


@dataclass(frozen=True)
class TaskResult:
    """Represents the benchmark evaluation result for a task instance."""

    instance_id: str
    repo: str
    complexity: ComplexityTier | None = None
    resolved: bool = False
    patch_size: int = 0
    tool_calls_used: int = 0
    tokens_used: int = 0
    had_truncation: bool = False
    time_seconds: float = 0.0
    error: str | None = None
