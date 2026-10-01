"""Pydantic schema for training execution configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class TrainingConfig(BaseModel):
    """Configuration settings for supervised fine-tuning training runs."""

    _DEFAULT_NUM_EPOCHS: ClassVar[int] = 5
    _DEFAULT_BATCH_SIZE: ClassVar[int] = 1
    _DEFAULT_GRAD_ACCUM_STEPS: ClassVar[int] = 8
    _DEFAULT_LEARNING_RATE: ClassVar[float] = 2e-4
    _DEFAULT_LR_SCHEDULER_TYPE: ClassVar[str] = "cosine"
    _DEFAULT_WARMUP_RATIO: ClassVar[float] = 0.05
    _DEFAULT_WEIGHT_DECAY: ClassVar[float] = 0.01
    _DEFAULT_MAX_GRAD_NORM: ClassVar[float] = 1.0
    _DEFAULT_BF16: ClassVar[bool] = True
    _DEFAULT_GRAD_CHECKPOINTING: ClassVar[bool] = True
    _DEFAULT_MAX_SEQ_LENGTH: ClassVar[int] = 28672
    _DEFAULT_PACKING: ClassVar[bool] = True
    _DEFAULT_EVAL_STRATEGY: ClassVar[str] = "steps"
    _DEFAULT_EVAL_STEPS: ClassVar[int] = 50
    _DEFAULT_SAVE_STRATEGY: ClassVar[str] = "steps"
    _DEFAULT_SAVE_STEPS: ClassVar[int] = 50
    _DEFAULT_LOAD_BEST_MODEL: ClassVar[bool] = True
    _DEFAULT_METRIC_FOR_BEST_MODEL: ClassVar[str] = "eval_loss"
    _DEFAULT_SAVE_TOTAL_LIMIT: ClassVar[int] = 3
    _DEFAULT_CURRICULUM_ENABLED: ClassVar[bool] = True

    model_config = ConfigDict(extra="ignore")

    num_epochs: int = _DEFAULT_NUM_EPOCHS
    per_device_train_batch_size: int = _DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = _DEFAULT_GRAD_ACCUM_STEPS
    learning_rate: float = _DEFAULT_LEARNING_RATE
    lr_scheduler_type: str = _DEFAULT_LR_SCHEDULER_TYPE
    warmup_ratio: float = _DEFAULT_WARMUP_RATIO
    weight_decay: float = _DEFAULT_WEIGHT_DECAY
    max_grad_norm: float = _DEFAULT_MAX_GRAD_NORM
    bf16: bool = _DEFAULT_BF16
    gradient_checkpointing: bool = _DEFAULT_GRAD_CHECKPOINTING
    max_seq_length: int = _DEFAULT_MAX_SEQ_LENGTH
    packing: bool = _DEFAULT_PACKING
    eval_strategy: str = _DEFAULT_EVAL_STRATEGY
    eval_steps: int = _DEFAULT_EVAL_STEPS
    save_strategy: str = _DEFAULT_SAVE_STRATEGY
    save_steps: int = _DEFAULT_SAVE_STEPS
    load_best_model_at_end: bool = _DEFAULT_LOAD_BEST_MODEL
    metric_for_best_model: str = _DEFAULT_METRIC_FOR_BEST_MODEL
    save_total_limit: int = _DEFAULT_SAVE_TOTAL_LIMIT
    curriculum_enabled: bool = _DEFAULT_CURRICULUM_ENABLED
