"""Pydantic schema for GRPO reinforcement learning hyperparameters."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class GRPOHyperConfig(BaseModel):
    """Configuration settings for Group Relative Policy Optimization hyperparameters."""

    _DEFAULT_NUM_GENERATIONS: ClassVar[int] = 4
    _DEFAULT_MAX_NEW_TOKENS: ClassVar[int] = 16384
    _DEFAULT_TEMPERATURE: ClassVar[float] = 0.7
    _DEFAULT_TOP_P: ClassVar[float] = 0.95
    _DEFAULT_LORA_R: ClassVar[int] = 32
    _DEFAULT_BETA: ClassVar[float] = 0.1
    _DEFAULT_NUM_EPOCHS: ClassVar[int] = 2
    _DEFAULT_BATCH_SIZE: ClassVar[int] = 1
    _DEFAULT_GRAD_ACCUM_STEPS: ClassVar[int] = 4
    _DEFAULT_LEARNING_RATE: ClassVar[float] = 5e-5
    _DEFAULT_LR_SCHEDULER_TYPE: ClassVar[str] = "cosine"
    _DEFAULT_WARMUP_RATIO: ClassVar[float] = 0.1
    _DEFAULT_MAX_GRAD_NORM: ClassVar[float] = 0.5
    _DEFAULT_MAX_TRAJECTORY_TOKENS: ClassVar[int] = 28672
    _DEFAULT_DISCARD_TRUNCATED: ClassVar[bool] = True

    model_config = ConfigDict(extra="ignore")

    num_generations: int = _DEFAULT_NUM_GENERATIONS
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE
    top_p: float = _DEFAULT_TOP_P
    lora_r: int = _DEFAULT_LORA_R
    beta: float = _DEFAULT_BETA
    num_epochs: int = _DEFAULT_NUM_EPOCHS
    per_device_train_batch_size: int = _DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = _DEFAULT_GRAD_ACCUM_STEPS
    learning_rate: float = _DEFAULT_LEARNING_RATE
    lr_scheduler_type: str = _DEFAULT_LR_SCHEDULER_TYPE
    warmup_ratio: float = _DEFAULT_WARMUP_RATIO
    max_grad_norm: float = _DEFAULT_MAX_GRAD_NORM
    max_trajectory_tokens: int = _DEFAULT_MAX_TRAJECTORY_TOKENS
    discard_truncated: bool = _DEFAULT_DISCARD_TRUNCATED
