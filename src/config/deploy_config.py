"""Pydantic schema for submission package deployment configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class DeployConfig(BaseModel):
    """Configuration settings for artifact packaging and Kaggle submission."""

    _DEFAULT_TEMPLATES_DIR: ClassVar[str] = "kaggle_staging/submission_templates"
    _DEFAULT_OUTPUT_DIR: ClassVar[str] = "dist"
    _DEFAULT_ZIP_PATH: ClassVar[str] = "dist/submission.zip"
    _DEFAULT_MIN_CV_THRESHOLD: ClassVar[float] = 0.30
    _DEFAULT_BUMP_TYPE: ClassVar[str] = "minor"
    _DEFAULT_MESSAGE: ClassVar[str] = ""
    _DEFAULT_STAGING_DIR: ClassVar[str] = "kaggle_staging"

    model_config = ConfigDict(extra="ignore")

    templates_dir: str = _DEFAULT_TEMPLATES_DIR
    output_dir: str = _DEFAULT_OUTPUT_DIR
    zip_path: str = _DEFAULT_ZIP_PATH
    adapters: list[dict[str, str]] = Field(default_factory=list)
    min_cv_threshold: float = _DEFAULT_MIN_CV_THRESHOLD
    bump_type: str = _DEFAULT_BUMP_TYPE
    message: str = _DEFAULT_MESSAGE
    staging_dir: str = _DEFAULT_STAGING_DIR
