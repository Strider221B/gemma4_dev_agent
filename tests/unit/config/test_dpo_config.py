"""Unit tests for DPOHyperConfig schema."""

from __future__ import annotations

from src.config.dpo_config import DPOHyperConfig


class TestDPOConfig:
    """Test suite for DPOHyperConfig Pydantic model validation."""

    _CUSTOM_BETA: float = 0.2
    _CUSTOM_LR: float = 1e-5

    def test_dpo_config_defaults(self) -> None:
        """Verify default configuration parameters for DPO."""
        config = DPOHyperConfig()
        assert config.beta == 0.1
        assert config.loss_type == "sigmoid"
        assert config.max_length == 28672
        assert config.max_prompt_length == 4096
        assert config.num_epochs == 2
        assert config.learning_rate == 5e-6

    def test_dpo_config_custom_values(self) -> None:
        """Verify custom values override default configurations."""
        config = DPOHyperConfig(
            beta=self._CUSTOM_BETA,
            learning_rate=self._CUSTOM_LR,
        )
        assert config.beta == self._CUSTOM_BETA
        assert config.learning_rate == self._CUSTOM_LR
