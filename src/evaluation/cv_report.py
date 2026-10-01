"""Data model representing an aggregated cross-validation report across all folds."""

from dataclasses import dataclass

from src.evaluation.cv_result import CVResult


@dataclass(frozen=True)
class CVReport:
    """Represents overall evaluation summary and diagnostic recommendations."""

    version: str
    timestamp: str
    fold_results: list[CVResult]
    aggregate_resolution_rate: float
    per_repo_rates: dict[str, float]
    per_complexity_rates: dict[str, float]
    overfitting_signal: str
    recommendations: list[str]
    total_resolved: int = 0
    total_tasks: int = 0
