"""Resolver for base model filesystem and repository paths."""

from __future__ import annotations

import os
from typing import ClassVar

from src.utils.constants import (
    KAGGLE_MODEL_PATH,
    LOCAL_TEST_MODEL_NAME,
    LOCAL_TEST_MODEL_SLUG,
    MODEL_NAME,
)


class ModelPathResolver:
    """Resolves model identifiers to local filesystem paths or hub identifiers."""

    _COMPETITION_MODEL_SLUG: ClassVar[str] = "gemma-4-31b-it-qat-w4a16-ct"
    _LOCAL_TEST_MODEL_SLUG: ClassVar[str] = LOCAL_TEST_MODEL_SLUG
    _ENV_MODEL_PATH: ClassVar[str] = "MODEL_PATH"
    _ENV_GEMMA_PATH: ClassVar[str] = "GEMMA_MODEL_PATH"
    _ENV_LOCAL_MODEL_PATH: ClassVar[str] = "LOCAL_MODEL_PATH"
    _ENV_GEMMA_E2B_PATH: ClassVar[str] = "GEMMA_E2B_MODEL_PATH"
    _CONFIG_FILE: ClassVar[str] = "config.json"
    _SUBDIR_SNAPSHOTS: ClassVar[str] = "snapshots"
    _CANDIDATE_PATHS: ClassVar[tuple[str, ...]] = (
        KAGGLE_MODEL_PATH,
        "/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct",
    )
    _LOCAL_CANDIDATE_PATHS: ClassVar[tuple[str, ...]] = (
        "models/gemma-4-e2b-it-qat-w4a16-ct",
        "~/.cache/huggingface/hub/models--google--gemma-4-e2b-it-qat-w4a16-ct",
    )

    def __init__(self) -> None:
        """Initialize ModelPathResolver."""

    @classmethod
    def has_local_model(cls, model_name_or_path: str) -> bool:
        """Check if resolved model path exists locally as a directory."""
        resolved = cls.resolve(model_name_or_path)
        return os.path.isdir(os.path.expanduser(resolved))

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
        expanded = os.path.expanduser(model_name_or_path)
        if os.path.exists(expanded):
            return expanded
        env_resolved = cls._check_env_paths(model_name_or_path)
        if env_resolved:
            return env_resolved
        candidate = cls._find_candidate_for_model(model_name_or_path)
        if candidate:
            return candidate
        return model_name_or_path

    @classmethod
    def _check_env_paths(cls, identifier: str = "") -> str | None:
        """Check environment variables for locally mounted model paths."""
        env_vars = (
            (cls._ENV_LOCAL_MODEL_PATH, cls._ENV_GEMMA_E2B_PATH, cls._ENV_MODEL_PATH)
            if cls._is_local_test_model(identifier)
            else (cls._ENV_MODEL_PATH, cls._ENV_GEMMA_PATH)
        )
        for env_var in env_vars:
            env_val = os.environ.get(env_var)
            if env_val and os.path.exists(os.path.expanduser(env_val)):
                return env_val
        return None

    @classmethod
    def _find_candidate_for_model(cls, identifier: str) -> str | None:
        """Find existing candidate path matching identifier type."""
        if cls._is_local_test_model(identifier):
            return cls._find_existing_local_candidate()
        if cls._is_competition_model(identifier):
            return cls._find_existing_candidate()
        return None

    @classmethod
    def _find_existing_candidate(cls) -> str | None:
        """Find first existing candidate path from known Kaggle mount locations."""
        for candidate in cls._CANDIDATE_PATHS:
            if os.path.exists(os.path.expanduser(candidate)):
                return candidate
        return None

    @classmethod
    def _find_existing_local_candidate(cls) -> str | None:
        """Find first existing candidate path from local test model locations."""
        for candidate in cls._LOCAL_CANDIDATE_PATHS:
            expanded = os.path.expanduser(candidate)
            if not os.path.exists(expanded):
                continue
            if os.path.isdir(os.path.join(expanded, cls._SUBDIR_SNAPSHOTS)):
                snapshot = cls._resolve_hf_snapshot(expanded)
                if snapshot:
                    return snapshot
                continue
            return expanded
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

    @classmethod
    def _is_local_test_model(cls, identifier: str) -> bool:
        """Check whether identifier matches local test model variants."""
        return identifier in (LOCAL_TEST_MODEL_NAME, cls._LOCAL_TEST_MODEL_SLUG)

    @classmethod
    def _is_supported_model(cls, identifier: str) -> bool:
        """Check whether identifier matches either competition or local test model."""
        return cls._is_competition_model(identifier) or cls._is_local_test_model(identifier)

    @classmethod
    def _resolve_hf_snapshot(cls, hub_dir: str) -> str | None:
        """Find latest snapshot directory containing config.json within HF hub directory."""
        snapshots_dir = os.path.join(hub_dir, cls._SUBDIR_SNAPSHOTS)
        if not os.path.isdir(snapshots_dir):
            return None
        try:
            entries = sorted(os.listdir(snapshots_dir), reverse=True)
            for entry in entries:
                snapshot_path = os.path.join(snapshots_dir, entry)
                if os.path.isfile(os.path.join(snapshot_path, cls._CONFIG_FILE)):
                    return snapshot_path
        except OSError:
            return None
        return None
