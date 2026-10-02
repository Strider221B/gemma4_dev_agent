"""Unit tests for MetricTracker."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.evaluation.metric_tracker import MetricTracker
from src.evaluation.overfitting_signal import OverfittingSignal
from src.utils.telemetry_logger import TelemetryLogger


class TestMetricTracker:
    """Test suite for MetricTracker evaluation monitoring and trend detection."""

    _VER_1: str = "v1"
    _VER_2: str = "v2"
    _VER_3: str = "v3"
    _VER_4: str = "v4"

    def test_record_stores_entry(self) -> None:
        """Verify record appends entry to history with computed gap."""
        tracker = MetricTracker(gap_threshold=0.15)
        tracker.record(self._VER_1, 0.35, 0.32)
        history = tracker.get_history()
        assert len(history) == 1
        entry = history[0]
        assert entry["version"] == self._VER_1
        assert entry["cv_score"] == 0.35
        assert entry["lb_score"] == 0.32
        assert abs(float(str(entry["gap"])) - 0.03) < 1e-5

    def test_record_triggers_alert_when_gap_exceeded(self) -> None:
        """Verify alert is emitted when generalization gap exceeds threshold."""
        telemetry = MagicMock(spec=TelemetryLogger)
        tracker = MetricTracker(gap_threshold=0.10, telemetry=telemetry)
        tracker.record(self._VER_1, 0.45, 0.20)
        telemetry.log_info.assert_called_once()
        assert "OVERFITTING ALERT" in telemetry.log_info.call_args[0][0]

    def test_detect_trend_none_when_stable(self) -> None:
        """Verify detect_trend returns NONE when CV and LB move proportionally."""
        tracker = MetricTracker()
        tracker.record("v1", 0.30, 0.30)
        tracker.record("v2", 0.32, 0.32)
        tracker.record("v3", 0.34, 0.34)
        assert tracker.detect_trend() == OverfittingSignal.NONE

    def test_detect_trend_strong_when_cv_up_lb_down(self) -> None:
        """Verify detect_trend returns STRONG when CV increases but LB decreases."""
        tracker = MetricTracker()
        tracker.record("v1", 0.30, 0.35)
        tracker.record("v2", 0.35, 0.32)
        tracker.record("v3", 0.40, 0.28)
        assert tracker.detect_trend() == OverfittingSignal.STRONG

    def test_detect_trend_moderate_when_cv_grows_faster(self) -> None:
        """Verify detect_trend returns MODERATE when CV slope > 2x LB slope."""
        tracker = MetricTracker()
        tracker.record("v1", 0.30, 0.30)
        tracker.record("v2", 0.40, 0.32)
        tracker.record("v3", 0.50, 0.34)
        assert tracker.detect_trend() == OverfittingSignal.MODERATE

    def test_detect_trend_insufficient_history_returns_none(self) -> None:
        """Verify detect_trend returns NONE with fewer than 2 entries."""
        tracker = MetricTracker()
        tracker.record("v1", 0.30, 0.30)
        assert tracker.detect_trend() == OverfittingSignal.NONE

    def test_export_report_returns_json(self) -> None:
        """Verify export_report serializes valid JSON list of history."""
        tracker = MetricTracker()
        tracker.record("v1", 0.30, 0.30)
        tracker.record("v2", 0.35, 0.33)
        report_str = tracker.export_report()
        loaded = json.loads(report_str)
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0]["version"] == "v1"
