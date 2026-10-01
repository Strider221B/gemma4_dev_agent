"""Unit tests for DeployConfig schema."""

from __future__ import annotations

from src.config.deploy_config import DeployConfig


class TestDeployConfig:
    """Test suite for DeployConfig Pydantic model validation."""

    _CUSTOM_OUTPUT: str = "dist/v1"
    _CUSTOM_THRESHOLD: float = 0.40

    def test_deploy_config_defaults(self) -> None:
        """Verify default configuration parameters for deployment."""
        config = DeployConfig()
        assert config.templates_dir == "kaggle_staging/submission_templates"
        assert config.output_dir == "dist"
        assert config.zip_path == "dist/submission.zip"
        assert config.adapters == []
        assert config.min_cv_threshold == 0.30
        assert config.bump_type == "minor"
        assert config.staging_dir == "kaggle_staging"

    def test_deploy_config_custom_values(self) -> None:
        """Verify custom values override default configurations."""
        config = DeployConfig(
            output_dir=self._CUSTOM_OUTPUT,
            min_cv_threshold=self._CUSTOM_THRESHOLD,
        )
        assert config.output_dir == self._CUSTOM_OUTPUT
        assert config.min_cv_threshold == self._CUSTOM_THRESHOLD
