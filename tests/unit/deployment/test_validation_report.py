"""Unit tests for ValidationReport data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.deployment.validation_check import ValidationCheck
from src.deployment.validation_report import ValidationReport


class TestValidationReport:
    """Test suite for ValidationReport frozen dataclass."""

    _CHECK_NAME: str = "archive_check"
    _CHECK_DETAIL: str = "All checks passed"
    _MODIFIED_STATUS: bool = False

    def test_validation_report_creation(self) -> None:
        """Verify ValidationReport aggregates checks and pass status."""
        check = ValidationCheck(
            name=self._CHECK_NAME,
            passed=True,
            detail=self._CHECK_DETAIL,
        )
        report = ValidationReport(checks=[check], all_passed=True)
        assert len(report.checks) == 1
        assert report.checks[0] == check
        assert report.all_passed is True

    def test_validation_report_is_frozen(self) -> None:
        """Verify modifying field on ValidationReport raises FrozenInstanceError."""
        check = ValidationCheck(
            name=self._CHECK_NAME,
            passed=True,
            detail=self._CHECK_DETAIL,
        )
        report = ValidationReport(checks=[check], all_passed=True)
        with pytest.raises(FrozenInstanceError):
            report.all_passed = self._MODIFIED_STATUS
