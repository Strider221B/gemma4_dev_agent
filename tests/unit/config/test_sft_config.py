"""Unit tests for SFTConfig schema."""

from __future__ import annotations

from src.config.model_config import ModelConfig
from src.config.sft_config import SFTConfig


class TestSFTConfig:
    """Test suite for SFTConfig Pydantic model validation."""

    _CUSTOM_EVAL_FOLD: int = 3
    _CUSTOM_OUTPUT_DIR: str = "/custom/checkpoints"

    def test_sft_config_defaults(self) -> None:
        """Verify default configuration values for SFT pipeline."""
        config = SFTConfig()
        assert isinstance(config.model, ModelConfig)
        assert config.eval_fold == 0
        assert config.output_dir == "/kaggle/working/checkpoints"
        assert config.log_path == "/kaggle/working/logs/run_sft.log"

    def test_sft_config_custom_values(self) -> None:
        """Verify custom values override default configurations."""
        config = SFTConfig(
            eval_fold=self._CUSTOM_EVAL_FOLD,
            output_dir=self._CUSTOM_OUTPUT_DIR,
        )
        assert config.eval_fold == self._CUSTOM_EVAL_FOLD
        assert config.output_dir == self._CUSTOM_OUTPUT_DIR
