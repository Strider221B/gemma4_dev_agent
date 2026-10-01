"""Data model representing trajectory execution results for an agent run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    """Represents the raw execution outcome of an agent attempt."""

    instance_id: str
    patch: str | None = None
    num_tool_calls: int = 0
    total_tokens: int = 0
    had_truncation: bool = False
    elapsed_seconds: float = 0.0
