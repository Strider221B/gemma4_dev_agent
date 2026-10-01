"""Pydantic schema for cross-validation evaluation configuration."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.data_paths_config import DataPathsConfig


class EvalConfig(BaseModel):
    """Configuration settings for cross-validation evaluation runs."""

    _DEFAULT_NUM_FOLDS: ClassVar[int] = 4
    _DEFAULT_RANDOM_SEED: ClassVar[int] = 42
    _DEFAULT_MAX_TOOL_CALLS: ClassVar[int] = 100
    _DEFAULT_MAX_TIME_MINUTES: ClassVar[float] = 60.0
    _DEFAULT_MAX_TURNS: ClassVar[int] = 500
    _DEFAULT_COMMAND_TIMEOUT_SECONDS: ClassVar[int] = 300
    _DEFAULT_MODE: ClassVar[str] = "full"
    _DEFAULT_SANDBOX: ClassVar[str] = "docker"
    _DEFAULT_CONCURRENCY: ClassVar[int] = 2
    _DEFAULT_GAP_THRESHOLD: ClassVar[float] = 0.15
    _DEFAULT_TREND_WINDOW: ClassVar[int] = 5
    _DEFAULT_OUTPUT_DIR: ClassVar[str] = "/kaggle/working/cv_results"

    model_config = ConfigDict(extra="ignore")

    num_folds: int = _DEFAULT_NUM_FOLDS
    random_seed: int = _DEFAULT_RANDOM_SEED
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS
    max_time_minutes: float = _DEFAULT_MAX_TIME_MINUTES
    max_turns: int = _DEFAULT_MAX_TURNS
    command_timeout_seconds: int = _DEFAULT_COMMAND_TIMEOUT_SECONDS
    mode: Literal["full", "cached", "dry_run"] = "full"
    sandbox: str = _DEFAULT_SANDBOX
    concurrency: int = _DEFAULT_CONCURRENCY
    gap_threshold: float = _DEFAULT_GAP_THRESHOLD
    trend_window: int = _DEFAULT_TREND_WINDOW
    output_dir: str = _DEFAULT_OUTPUT_DIR
    quick_eval: dict[str, object] = Field(default_factory=dict)
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)

    @model_validator(mode="before")
    @classmethod
    def _extract_nested_sections(cls, data: object) -> object:
        """Flatten nested sections from YAML config if present."""
        if not isinstance(data, dict):
            return data
        flattened: dict[str, object] = dict(data)
        for section in ("cv", "tracking"):
            nested = flattened.get(section)
            if isinstance(nested, dict):
                flattened.update(nested)
        return flattened
