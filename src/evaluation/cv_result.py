"""Data model representing cross-validation evaluation result for a fold."""

from dataclasses import dataclass

from src.evaluation.task_result import TaskResult


@dataclass(frozen=True)
class CVResult:
    """Represents performance and operational metrics for one CV fold."""

    fold_idx: int
    val_repo: str
    resolution_rate: float
    resolved_count: int
    total_count: int
    per_task_results: list[TaskResult]
    avg_tool_calls: float = 0.0
    avg_tokens: float = 0.0
    truncation_rate: float = 0.0
    per_complexity: dict[str, float] | None = None
