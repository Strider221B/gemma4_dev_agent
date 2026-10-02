"""Unit tests for TrainingConfigBuilder verifying TRL training arguments generation."""

from __future__ import annotations

from src.config.sft_config import SFTConfig
from src.training.training_config import TrainingConfigBuilder


class TestTrainingConfigBuilder:
    """Test suite covering TrainingConfigBuilder dictionary generation."""

    _CUSTOM_OUTPUT_DIR: str = "/tmp/test_output"
    _CUSTOM_NUM_EPOCHS: int = 3
    _CUSTOM_LR: float = 1e-4

    def test_build_training_args_contains_expected_keys(self) -> None:
        """Verify generated arguments dictionary contains all necessary TRL keys."""
        config = SFTConfig()
        builder = TrainingConfigBuilder(config=config)
        args = builder.build_training_args()

        assert "output_dir" in args
        assert "num_train_epochs" in args
        assert "learning_rate" in args
        assert "eval_strategy" in args
        assert "save_strategy" in args
        assert "logging_steps" in args
        assert "seed" in args
        assert args["optim"] == "adamw_8bit"
        assert args["report_to"] == "none"

    def test_build_training_args_output_dir_joined(self) -> None:
        """Verify output_dir appends checkpoints subdirectory."""
        config = SFTConfig(output_dir=self._CUSTOM_OUTPUT_DIR)
        builder = TrainingConfigBuilder(config=config)
        args = builder.build_training_args()

        assert args["output_dir"] == f"{self._CUSTOM_OUTPUT_DIR}/checkpoints"

    def test_build_training_args_custom_values(self) -> None:
        """Verify custom hyperparameter values are reflected accurately."""
        config = SFTConfig()
        config.training.num_epochs = self._CUSTOM_NUM_EPOCHS
        config.training.learning_rate = self._CUSTOM_LR

        builder = TrainingConfigBuilder(config=config)
        args = builder.build_training_args()

        assert args["num_train_epochs"] == self._CUSTOM_NUM_EPOCHS
        assert args["learning_rate"] == self._CUSTOM_LR
