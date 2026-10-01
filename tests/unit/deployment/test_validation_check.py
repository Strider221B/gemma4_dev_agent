"""Unit tests for ValidationCheck data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.deployment.validation_check import ValidationCheck


class TestValidationCheck:
    """Test suite for ValidationCheck frozen dataclass."""

    _NAME: str = "wheel_size_check"
    _DETAIL: str = "Archive size is 12MB (<=500MB limit)"
    _MODIFIED_DETAIL: str = "Archive size is 15MB"

    def test_validation_check_creation(self) -> None:
        """Verify ValidationCheck attributes initialization."""
        check = ValidationCheck(
            name=self._NAME,
            passed=True,
            detail=self._DETAIL,
        )
        assert check.name == self._NAME
        assert check.passed is True
        assert check.detail == self._DETAIL

    def test_validation_check_is_frozen(self) -> None:
        """Verify modifying field on ValidationCheck raises FrozenInstanceError."""
        check = ValidationCheck(
            name=self._NAME,
            passed=True,
            detail=self._DETAIL,
        )
        with pytest.raises(FrozenInstanceError):
            check.detail = self._MODIFIED_DETAIL
