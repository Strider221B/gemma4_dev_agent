"""Unit tests for RLConfig schema."""

from __future__ import annotations

from src.config.model_config import ModelConfig
from src.config.rl_config import RLConfig


class TestRLConfig:
    """Test suite for RLConfig Pydantic model validation."""

    _CUSTOM_MODE: str = "dpo"
    _CUSTOM_ADAPTER: str = "/custom/checkpoints/sft_adapter"

    def test_rl_config_defaults(self) -> None:
        """Verify default configuration parameters for RL."""
        config = RLConfig()
        assert isinstance(config.model, ModelConfig)
        assert config.mode == "grpo"
        assert config.adapter_path == "/kaggle/working/checkpoints/sft_lora"
        assert config.output_dir == "/kaggle/working/checkpoints"
        assert config.log_path == "/kaggle/working/logs/run_rl.log"

    def test_rl_config_custom_values(self) -> None:
        """Verify custom values override default configurations."""
        config = RLConfig(
            mode="dpo",
            adapter_path=self._CUSTOM_ADAPTER,
        )
        assert config.mode == self._CUSTOM_MODE
        assert config.adapter_path == self._CUSTOM_ADAPTER
