"""Unit tests for OverfittingSignal enum."""

from src.evaluation.overfitting_signal import OverfittingSignal


class TestOverfittingSignal:
    """Test suite for OverfittingSignal enumeration."""

    _NONE: str = "NONE"
    _MODERATE: str = "MODERATE"
    _STRONG: str = "STRONG"

    def test_overfitting_signal_values(self) -> None:
        """Verify enum members have expected string values."""
        assert OverfittingSignal.NONE.value == self._NONE
        assert OverfittingSignal.MODERATE.value == self._MODERATE
        assert OverfittingSignal.STRONG.value == self._STRONG

    def test_overfitting_signal_lookup(self) -> None:
        """Verify enum member lookup by string value."""
        assert OverfittingSignal(self._NONE) is OverfittingSignal.NONE
        assert OverfittingSignal(self._MODERATE) is OverfittingSignal.MODERATE
        assert OverfittingSignal(self._STRONG) is OverfittingSignal.STRONG
