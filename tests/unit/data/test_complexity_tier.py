"""Unit tests for ComplexityTier enumeration."""

from src.data.complexity_tier import ComplexityTier


class TestComplexityTier:
    """Test suite for ComplexityTier enum values and properties."""

    _EXPECTED_SIMPLE: str = "SIMPLE"
    _EXPECTED_MODERATE: str = "MODERATE"
    _EXPECTED_COMPLEX: str = "COMPLEX"

    def test_complexity_tier_values(self) -> None:
        """Verify enum members have expected string values."""
        assert ComplexityTier.SIMPLE.value == self._EXPECTED_SIMPLE
        assert ComplexityTier.MODERATE.value == self._EXPECTED_MODERATE
        assert ComplexityTier.COMPLEX.value == self._EXPECTED_COMPLEX

    def test_complexity_tier_lookup(self) -> None:
        """Verify enum members can be resolved by value."""
        assert ComplexityTier(self._EXPECTED_SIMPLE) is ComplexityTier.SIMPLE
        assert ComplexityTier(self._EXPECTED_MODERATE) is ComplexityTier.MODERATE
        assert ComplexityTier(self._EXPECTED_COMPLEX) is ComplexityTier.COMPLEX
