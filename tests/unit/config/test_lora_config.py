"""Unit tests for LoRAConfig schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.lora_config import LoRAConfig


class TestLoRAConfig:
    """Test suite for LoRAConfig Pydantic model validation."""

    _CUSTOM_R: int = 16
    _CUSTOM_ALPHA: int = 32
    _CUSTOM_DROPOUT: float = 0.1
    _CUSTOM_MODULE: str = "custom_proj"

    def test_lora_config_defaults(self) -> None:
        """Verify default hyperparameter values for LoRA adapter."""
        config = LoRAConfig()
        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.target_modules
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"

    def test_lora_config_custom_values(self) -> None:
        """Verify custom adapter parameters are assigned properly."""
        config = LoRAConfig(
            r=self._CUSTOM_R,
            lora_alpha=self._CUSTOM_ALPHA,
            lora_dropout=self._CUSTOM_DROPOUT,
            target_modules=[self._CUSTOM_MODULE],
        )
        assert config.r == self._CUSTOM_R
        assert config.lora_alpha == self._CUSTOM_ALPHA
        assert config.lora_dropout == self._CUSTOM_DROPOUT
        assert config.target_modules == [self._CUSTOM_MODULE]

    def test_lora_config_validation_error(self) -> None:
        """Verify passing incompatible types raises validation error."""
        with pytest.raises(ValidationError):
            LoRAConfig(**{"r": "invalid_rank"})
