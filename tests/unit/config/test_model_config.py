"""Unit tests for ModelConfig schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.model_config import ModelConfig


class TestModelConfig:
    """Test suite for ModelConfig Pydantic model validation."""

    _CUSTOM_NAME: str = "custom-org/custom-model"
    _CUSTOM_SEQ_LEN: int = 16384
    _CUSTOM_DTYPE: str = "float16"

    def test_model_config_defaults(self) -> None:
        """Verify default field values match expected architectural settings."""
        config = ModelConfig()
        assert config.name == "google/gemma-4-31b-it-qat-w4a16-ct"
        assert config.load_in_4bit is True
        assert config.max_seq_length == 32768
        assert config.dtype == "bfloat16"

    def test_model_config_custom_values(self) -> None:
        """Verify custom values correctly override default configurations."""
        config = ModelConfig(
            name=self._CUSTOM_NAME,
            load_in_4bit=False,
            max_seq_length=self._CUSTOM_SEQ_LEN,
            dtype=self._CUSTOM_DTYPE,
        )
        assert config.name == self._CUSTOM_NAME
        assert config.load_in_4bit is False
        assert config.max_seq_length == self._CUSTOM_SEQ_LEN
        assert config.dtype == self._CUSTOM_DTYPE

    def test_model_config_validation_error(self) -> None:
        """Verify passing incompatible types raises validation error."""
        with pytest.raises(ValidationError):
            ModelConfig(**{"max_seq_length": "not-an-integer"})

    def test_get_resolved_path_delegates_to_resolver(self) -> None:
        """Verify get_resolved_path delegates resolution to ModelPathResolver."""
        config = ModelConfig(name=self._CUSTOM_NAME)
        assert config.get_resolved_path() == self._CUSTOM_NAME
