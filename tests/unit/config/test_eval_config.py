"""Unit tests for EvalConfig schema."""

from __future__ import annotations

from src.config.eval_config import EvalConfig


class TestEvalConfig:
    """Test suite for EvalConfig Pydantic model validation."""

    _CUSTOM_NUM_FOLDS: int = 8
    _CUSTOM_MODE: str = "cached"

    def test_eval_config_defaults(self) -> None:
        """Verify default configuration parameters for evaluation."""
        config = EvalConfig()
        assert config.num_folds == 4
        assert config.random_seed == 42
        assert config.max_tool_calls == 100
        assert config.max_time_minutes == 60.0
        assert config.max_turns == 500
        assert config.command_timeout_seconds == 300
        assert config.mode == "full"
        assert config.gap_threshold == 0.15
        assert config.trend_window == 5
        assert config.output_dir == "/kaggle/working/cv_results"

    def test_eval_config_custom_values(self) -> None:
        """Verify custom values override default configurations."""
        config = EvalConfig(
            num_folds=self._CUSTOM_NUM_FOLDS,
            mode="cached",
        )
        assert config.num_folds == self._CUSTOM_NUM_FOLDS
        assert config.mode == self._CUSTOM_MODE

    def test_eval_config_flattens_nested_sections(self) -> None:
        """Verify validator flattens nested cv and tracking sections."""
        nested_data: dict[str, object] = {
            "cv": {"num_folds": 6, "max_turns": 400},
            "tracking": {"gap_threshold": 0.20},
            "output_dir": "/custom/cv_results",
        }
        config = EvalConfig.model_validate(nested_data)
        assert config.num_folds == 6
        assert config.max_turns == 400
        assert config.gap_threshold == 0.20
        assert config.output_dir == "/custom/cv_results"
