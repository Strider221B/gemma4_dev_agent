"""Pydantic schema for reinforcement learning (RL) configuration."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config.dpo_config import DPOHyperConfig
from src.config.grpo_config import GRPOHyperConfig
from src.config.model_config import ModelConfig


class RLConfig(BaseModel):
    """Configuration settings for RL training pipeline (GRPO and DPO)."""

    _DEFAULT_ADAPTER_PATH: ClassVar[str] = "/kaggle/working/checkpoints/sft_lora"
    _DEFAULT_MODE: ClassVar[str] = "grpo"
    _DEFAULT_OUTPUT_DIR: ClassVar[str] = "/kaggle/working/checkpoints"
    _DEFAULT_LOG_PATH: ClassVar[str] = "/kaggle/working/logs/run_rl.log"

    model_config = ConfigDict(extra="ignore")

    model: ModelConfig = Field(default_factory=ModelConfig)
    adapter_path: str = _DEFAULT_ADAPTER_PATH
    mode: Literal["grpo", "dpo"] = "grpo"
    grpo: GRPOHyperConfig = Field(default_factory=GRPOHyperConfig)
    dpo: DPOHyperConfig = Field(default_factory=DPOHyperConfig)
    reward: dict[str, float] = Field(default_factory=dict)
    output_dir: str = _DEFAULT_OUTPUT_DIR
    log_path: str = _DEFAULT_LOG_PATH
