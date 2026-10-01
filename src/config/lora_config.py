"""Pydantic schema for LoRA adapter configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class LoRAConfig(BaseModel):
    """Configuration settings for LoRA fine-tuning adapters."""

    _DEFAULT_R: ClassVar[int] = 32
    _DEFAULT_ALPHA: ClassVar[int] = 64
    _DEFAULT_DROPOUT: ClassVar[float] = 0.05
    _DEFAULT_TARGET_MODULES: ClassVar[tuple[str, ...]] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    _DEFAULT_BIAS: ClassVar[str] = "none"
    _DEFAULT_TASK_TYPE: ClassVar[str] = "CAUSAL_LM"

    model_config = ConfigDict(extra="ignore")

    r: int = _DEFAULT_R
    lora_alpha: int = _DEFAULT_ALPHA
    lora_dropout: float = _DEFAULT_DROPOUT
    target_modules: list[str] = Field(
        default_factory=lambda: list(LoRAConfig._DEFAULT_TARGET_MODULES)
    )
    bias: str = _DEFAULT_BIAS
    task_type: str = _DEFAULT_TASK_TYPE
