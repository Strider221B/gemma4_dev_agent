"""Unit tests for ValidationResult data model."""

from __future__ import annotations

import pytest

from src.data.validation_result import ValidationResult


class TestValidationResult:
    """Test suite for ValidationResult data model."""

    _ERR_MSG: str = "Test error message"
    _WARN_MSG: str = "Test warning message"

    def test_validation_result_default_fields(self) -> None:
        """Verify default field initialization for ValidationResult."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_custom_fields(self) -> None:
        """Verify explicit field assignment in ValidationResult."""
        result = ValidationResult(
            valid=False,
            errors=[self._ERR_MSG],
            warnings=[self._WARN_MSG],
        )
        assert result.valid is False
        assert result.errors == [self._ERR_MSG]
        assert result.warnings == [self._WARN_MSG]

    def test_validation_result_frozen_immutability(self) -> None:
        """Verify ValidationResult instance is immutable."""
        result = ValidationResult(valid=True)
        with pytest.raises(AttributeError):
            setattr(result, "valid", False)
