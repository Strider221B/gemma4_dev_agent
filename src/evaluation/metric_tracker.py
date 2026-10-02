"""Metric tracker monitoring CV vs LB scores to detect overfitting trends."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.evaluation.overfitting_signal import OverfittingSignal
from src.utils.telemetry_logger import TelemetryLogger


class MetricTracker:
    """Monitors cross-validation and leaderboard trajectories for generalization gaps."""

    _DEFAULT_GAP_THRESHOLD: float = 0.15
    _DEFAULT_TREND_WINDOW: int = 5
    _MIN_HISTORY_FOR_TREND: int = 2
    _MODERATE_RATIO_MULTIPLIER: float = 2.0
    _JSON_INDENT: int = 2
    _KEY_VERSION: str = "version"
    _KEY_TIMESTAMP: str = "timestamp"
    _KEY_CV_SCORE: str = "cv_score"
    _KEY_LB_SCORE: str = "lb_score"
    _KEY_GAP: str = "gap"
    _ALERT_PREFIX: str = "OVERFITTING ALERT: Version "
    _ALERT_MSG: str = " exceeded generalization gap threshold: "

    def __init__(
        self,
        gap_threshold: float = _DEFAULT_GAP_THRESHOLD,
        trend_window: int = _DEFAULT_TREND_WINDOW,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        """Initialize MetricTracker with threshold and evaluation window."""
        self._gap_threshold: float = gap_threshold
        self._trend_window: int = trend_window
        self._telemetry: TelemetryLogger | None = telemetry
        self._history: list[dict[str, object]] = []

    def record(self, version: str, cv_score: float, lb_score: float | None = None) -> None:
        """Record evaluation metric entry and trigger alert if gap threshold is breached."""
        entry = self._build_entry(version, cv_score, lb_score)
        self._history.append(entry)
        gap = entry.get(self._KEY_GAP)
        if isinstance(gap, (int, float)) and gap > self._gap_threshold:
            self._alert_overfitting(entry)

    def detect_trend(self) -> OverfittingSignal:
        """Analyze recent history window to detect potential overfitting dynamics."""
        recent = self._history[-self._trend_window:]
        if len(recent) < self._MIN_HISTORY_FOR_TREND:
            return OverfittingSignal.NONE
        cv_scores = [float(str(e[self._KEY_CV_SCORE])) for e in recent]
        lb_entries = [
            float(str(e[self._KEY_LB_SCORE]))
            for e in recent
            if e.get(self._KEY_LB_SCORE) is not None
        ]
        if len(lb_entries) < self._MIN_HISTORY_FOR_TREND:
            return OverfittingSignal.NONE
        cv_slope = self._linear_trend(cv_scores)
        lb_slope = self._linear_trend(lb_entries)
        return self._classify_signal(cv_slope, lb_slope)

    def get_history(self) -> list[dict[str, object]]:
        """Return a copy of all recorded metric history entries."""
        return list(self._history)

    def export_report(self) -> str:
        """Export serialized JSON report of full historical tracking records."""
        return json.dumps(self.get_history(), indent=self._JSON_INDENT)

    def _linear_trend(self, values: list[float]) -> float:
        """Compute ordinary least squares linear regression slope for series."""
        count = len(values)
        if count < self._MIN_HISTORY_FOR_TREND:
            return 0.0
        x_mean = (count - 1) / 2.0
        y_mean = sum(values) / count
        return self._compute_slope(values, x_mean, y_mean)

    def _alert_overfitting(self, entry: dict[str, object]) -> None:
        """Log or dispatch alert notification when generalization gap is excessive."""
        version = entry.get(self._KEY_VERSION, "")
        gap = entry.get(self._KEY_GAP, 0.0)
        msg = f"{self._ALERT_PREFIX}{version}{self._ALERT_MSG}{gap}"
        if self._telemetry:
            self._telemetry.log_info(msg)

    def _build_entry(
        self, version: str, cv_score: float, lb_score: float | None
    ) -> dict[str, object]:
        """Construct structured metric dictionary for history logging."""
        gap = abs(cv_score - lb_score) if lb_score is not None else None
        return {
            self._KEY_VERSION: version,
            self._KEY_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
            self._KEY_CV_SCORE: cv_score,
            self._KEY_LB_SCORE: lb_score,
            self._KEY_GAP: gap,
        }

    def _compute_slope(self, values: list[float], x_mean: float, y_mean: float) -> float:
        """Calculate slope numerator and denominator for index-based least squares."""
        numerator = sum(
            (idx - x_mean) * (val - y_mean) for idx, val in enumerate(values)
        )
        denominator = sum((idx - x_mean) ** 2 for idx in range(len(values)))
        return (numerator / denominator) if denominator != 0.0 else 0.0

    def _classify_signal(self, cv_slope: float, lb_slope: float) -> OverfittingSignal:
        """Classify overfitting signal severity based on slope trends."""
        if cv_slope > 0 and lb_slope <= 0:
            return OverfittingSignal.STRONG
        if (
            cv_slope > 0
            and lb_slope > 0
            and cv_slope > (self._MODERATE_RATIO_MULTIPLIER * lb_slope)
        ):
            return OverfittingSignal.MODERATE
        return OverfittingSignal.NONE
