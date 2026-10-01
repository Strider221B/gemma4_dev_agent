"""Pydantic schema for supervised fine-tuning (SFT) configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from src.config.data_paths_config import DataPathsConfig
from src.config.lora_config import LoRAConfig
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig


class SFTConfig(BaseModel):
    """Configuration settings for supervised fine-tuning pipeline execution."""

    _DEFAULT_EVAL_FOLD: ClassVar[int] = 0
    _DEFAULT_OUTPUT_DIR: ClassVar[str] = "/kaggle/working/checkpoints"
    _DEFAULT_LOG_PATH: ClassVar[str] = "/kaggle/working/logs/run_sft.log"

    model_config = ConfigDict(extra="ignore")

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval_fold: int = _DEFAULT_EVAL_FOLD
    output_dir: str = _DEFAULT_OUTPUT_DIR
    log_path: str = _DEFAULT_LOG_PATH
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)
