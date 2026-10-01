"""Pydantic schema for DPO reinforcement learning hyperparameters."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class DPOHyperConfig(BaseModel):
    """Configuration settings for Direct Preference Optimization hyperparameters."""

    _DEFAULT_BETA: ClassVar[float] = 0.1
    _DEFAULT_LOSS_TYPE: ClassVar[str] = "sigmoid"
    _DEFAULT_MAX_LENGTH: ClassVar[int] = 28672
    _DEFAULT_MAX_PROMPT_LENGTH: ClassVar[int] = 4096
    _DEFAULT_NUM_EPOCHS: ClassVar[int] = 2
    _DEFAULT_BATCH_SIZE: ClassVar[int] = 1
    _DEFAULT_GRAD_ACCUM_STEPS: ClassVar[int] = 4
    _DEFAULT_LEARNING_RATE: ClassVar[float] = 5e-6

    model_config = ConfigDict(extra="ignore")

    beta: float = _DEFAULT_BETA
    loss_type: str = _DEFAULT_LOSS_TYPE
    max_length: int = _DEFAULT_MAX_LENGTH
    max_prompt_length: int = _DEFAULT_MAX_PROMPT_LENGTH
    num_epochs: int = _DEFAULT_NUM_EPOCHS
    per_device_train_batch_size: int = _DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = _DEFAULT_GRAD_ACCUM_STEPS
    learning_rate: float = _DEFAULT_LEARNING_RATE
