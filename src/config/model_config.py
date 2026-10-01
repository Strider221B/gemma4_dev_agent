"""Pydantic schema for model configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    """Configuration settings for base language models."""

    _DEFAULT_NAME: ClassVar[str] = "google/gemma-4-31b-it-qat-w4a16-ct"
    _DEFAULT_LOAD_IN_4BIT: ClassVar[bool] = True
    _DEFAULT_MAX_SEQ_LENGTH: ClassVar[int] = 32768
    _DEFAULT_DTYPE: ClassVar[str] = "bfloat16"

    model_config = ConfigDict(extra="ignore")

    name: str = _DEFAULT_NAME
    load_in_4bit: bool = _DEFAULT_LOAD_IN_4BIT
    max_seq_length: int = _DEFAULT_MAX_SEQ_LENGTH
    dtype: str = _DEFAULT_DTYPE
