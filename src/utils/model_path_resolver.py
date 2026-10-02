"""Resolver for base model filesystem and repository paths."""

from __future__ import annotations

import os
from typing import ClassVar

from src.utils.constants import KAGGLE_MODEL_PATH, MODEL_NAME


class ModelPathResolver:
    """Resolves model identifiers to local filesystem paths or hub identifiers."""

    _COMPETITION_MODEL_SLUG: ClassVar[str] = "gemma-4-31b-it-qat-w4a16-ct"
    _ENV_MODEL_PATH: ClassVar[str] = "MODEL_PATH"
    _ENV_GEMMA_PATH: ClassVar[str] = "GEMMA_MODEL_PATH"
    _CANDIDATE_PATHS: ClassVar[tuple[str, ...]] = (
        KAGGLE_MODEL_PATH,
        "/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct",
    )

    def __init__(self) -> None:
        """Initialize ModelPathResolver."""

    @classmethod
    def resolve(cls, model_name_or_path: str) -> str:
        """Resolve a model name or path to an existing local directory if available.

        Args:
            model_name_or_path: HuggingFace model repo identifier or local directory path.

        Returns:
            Resolved local filesystem path if found, or the input identifier.
        """
        if not model_name_or_path:
            return cls._find_fallback_path()
        if os.path.exists(model_name_or_path):
            return model_name_or_path
        env_resolved = cls._check_env_paths()
        if env_resolved:
            return env_resolved
        if cls._is_competition_model(model_name_or_path):
            candidate = cls._find_existing_candidate()
            if candidate:
                return candidate
        return model_name_or_path

    @classmethod
    def _check_env_paths(cls) -> str | None:
        """Check environment variables for locally mounted model paths."""
        for env_var in (cls._ENV_MODEL_PATH, cls._ENV_GEMMA_PATH):
            env_val = os.environ.get(env_var)
            if env_val and os.path.exists(env_val):
                return env_val
        return None

    @classmethod
    def _find_existing_candidate(cls) -> str | None:
        """Find first existing candidate path from known Kaggle mount locations."""
        for candidate in cls._CANDIDATE_PATHS:
            if os.path.exists(candidate):
                return candidate
        return None

    @classmethod
    def _find_fallback_path(cls) -> str:
        """Return fallback candidate path or default model identifier."""
        candidate = cls._find_existing_candidate()
        return candidate if candidate is not None else MODEL_NAME

    @classmethod
    def _is_competition_model(cls, identifier: str) -> bool:
        """Check whether identifier matches standard competition model variants."""
        return identifier in (MODEL_NAME, cls._COMPETITION_MODEL_SLUG)
