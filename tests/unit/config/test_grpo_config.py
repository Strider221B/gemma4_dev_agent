"""Unit tests for GRPOHyperConfig schema."""

from __future__ import annotations

from src.config.grpo_config import GRPOHyperConfig


class TestGRPOConfig:
    """Test suite for GRPOHyperConfig Pydantic model validation."""

    _CUSTOM_GENERATIONS: int = 8
    _CUSTOM_TEMP: float = 0.8

    def test_grpo_config_defaults(self) -> None:
        """Verify default configuration parameters for GRPO."""
        config = GRPOHyperConfig()
        assert config.num_generations == 4
        assert config.max_new_tokens == 16384
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.lora_r == 32
        assert config.beta == 0.1
        assert config.discard_truncated is True

    def test_grpo_config_custom_values(self) -> None:
        """Verify custom values override default configurations."""
        config = GRPOHyperConfig(
            num_generations=self._CUSTOM_GENERATIONS,
            temperature=self._CUSTOM_TEMP,
        )
        assert config.num_generations == self._CUSTOM_GENERATIONS
        assert config.temperature == self._CUSTOM_TEMP
