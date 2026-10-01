"""Unit tests for InspectionCheck data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.inspection_check import InspectionCheck


class TestInspectionCheck:
    """Test suite for InspectionCheck frozen dataclass."""

    _NAME: str = "leakage_detector"
    _FINDING: str = "Hardcoded instance_id found"
    _CUSTOM_SEVERITY: str = "critical"
    _DEFAULT_SEVERITY: str = "info"
    _MODIFIED_SEVERITY: str = "warning"

    def test_inspection_check_creation(self) -> None:
        """Verify InspectionCheck fields and default severity."""
        check = InspectionCheck(
            name=self._NAME,
            passed=False,
            findings=[self._FINDING],
        )
        assert check.name == self._NAME
        assert check.passed is False
        assert check.findings == [self._FINDING]
        assert check.severity == self._DEFAULT_SEVERITY

    def test_inspection_check_custom_severity(self) -> None:
        """Verify InspectionCheck accepts custom severity level."""
        check = InspectionCheck(
            name=self._NAME,
            passed=True,
            findings=[],
            severity=self._CUSTOM_SEVERITY,
        )
        assert check.severity == self._CUSTOM_SEVERITY

    def test_inspection_check_is_frozen(self) -> None:
        """Verify modifying field on InspectionCheck raises FrozenInstanceError."""
        check = InspectionCheck(
            name=self._NAME,
            passed=True,
            findings=[],
        )
        with pytest.raises(FrozenInstanceError):
            check.severity = self._MODIFIED_SEVERITY
