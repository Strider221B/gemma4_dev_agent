"""Data model representing cross-validation stress test results for a peer strategy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdversarialResult:
    """Represents comparative CV evaluation metrics against baseline."""

    peer_cv_score: float
    baseline_cv_score: float
    improvement: float
    cv_lb_gap: float
    is_significant: bool
    per_fold_comparison: dict[str, dict[str, float]]
