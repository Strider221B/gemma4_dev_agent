"""Unit tests for TrainingConfig schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.training_config import TrainingConfig


class TestTrainingConfig:
    """Test suite for TrainingConfig Pydantic model validation."""

    _CUSTOM_EPOCHS: int = 10
    _CUSTOM_LR: float = 1e-4

    def test_training_config_defaults(self) -> None:
        """Verify default hyperparameter values for SFT training."""
        config = TrainingConfig()
        assert config.num_epochs == 5
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 2e-4
        assert config.lr_scheduler_type == "cosine"
        assert config.bf16 is True
        assert config.packing is True
        assert config.curriculum_enabled is True

    def test_training_config_custom_values(self) -> None:
        """Verify custom training parameters are assigned properly."""
        config = TrainingConfig(
            num_epochs=self._CUSTOM_EPOCHS,
            learning_rate=self._CUSTOM_LR,
            packing=False,
        )
        assert config.num_epochs == self._CUSTOM_EPOCHS
        assert config.learning_rate == self._CUSTOM_LR
        assert config.packing is False

    def test_training_config_validation_error(self) -> None:
        """Verify passing invalid type raises validation error."""
        with pytest.raises(ValidationError):
            TrainingConfig(**{"num_epochs": "invalid_epochs"})
