"""Unit tests for ModelPathResolver."""

from __future__ import annotations

import os
from typing import ClassVar
from unittest.mock import patch

from src.utils.constants import KAGGLE_MODEL_PATH, MODEL_NAME
from src.utils.model_path_resolver import ModelPathResolver


class TestModelPathResolver:
    """Test suite for ModelPathResolver offline path and fallback resolution."""

    _CUSTOM_LOCAL_PATH: ClassVar[str] = "/custom/local/gemma-model"
    _CUSTOM_REPO_ID: ClassVar[str] = "custom-org/custom-model"
    _COMPETITION_SLUG: ClassVar[str] = "gemma-4-31b-it-qat-w4a16-ct"
    _ENV_PATH: ClassVar[str] = "/env/path/to/model"

    def test_instantiation_succeeds(self) -> None:
        """Verify ModelPathResolver can be instantiated without error."""
        resolver = ModelPathResolver()
        assert isinstance(resolver, ModelPathResolver)

    def test_resolve_existing_filesystem_path_returns_same_path(self) -> None:
        """Verify existing filesystem path returns directly without modification."""
        with patch("os.path.exists", return_value=True):
            result = ModelPathResolver.resolve(self._CUSTOM_LOCAL_PATH)
            assert result == self._CUSTOM_LOCAL_PATH

    def test_resolve_env_var_model_path_when_exists(self) -> None:
        """Verify resolution prioritizes MODEL_PATH environment variable if valid."""
        with patch.dict(os.environ, {"MODEL_PATH": self._ENV_PATH}):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == self._ENV_PATH
                result = ModelPathResolver.resolve(MODEL_NAME)
                assert result == self._ENV_PATH

    def test_resolve_env_var_gemma_path_when_exists(self) -> None:
        """Verify resolution checks GEMMA_MODEL_PATH environment variable."""
        with patch.dict(os.environ, {"GEMMA_MODEL_PATH": self._ENV_PATH}, clear=True):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == self._ENV_PATH
                result = ModelPathResolver.resolve(MODEL_NAME)
                assert result == self._ENV_PATH

    def test_resolve_competition_model_resolves_to_kaggle_path(self) -> None:
        """Verify competition model ID resolves to Kaggle offline path when present."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == KAGGLE_MODEL_PATH
                result = ModelPathResolver.resolve(MODEL_NAME)
                assert result == KAGGLE_MODEL_PATH

    def test_resolve_competition_slug_resolves_to_kaggle_path(self) -> None:
        """Verify short competition model slug resolves to Kaggle offline path."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == KAGGLE_MODEL_PATH
                result = ModelPathResolver.resolve(self._COMPETITION_SLUG)
                assert result == KAGGLE_MODEL_PATH

    def test_resolve_nonexistent_custom_repo_returns_input(self) -> None:
        """Verify unresolved non-competition repo identifier returns as-is."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                result = ModelPathResolver.resolve(self._CUSTOM_REPO_ID)
                assert result == self._CUSTOM_REPO_ID

    def test_resolve_empty_string_with_existing_candidate(self) -> None:
        """Verify empty input resolves to existing candidate when available."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.side_effect = lambda p: p == KAGGLE_MODEL_PATH
            result = ModelPathResolver.resolve("")
            assert result == KAGGLE_MODEL_PATH

    def test_resolve_empty_string_without_candidate_returns_default(self) -> None:
        """Verify empty input returns MODEL_NAME when no candidates exist."""
        with patch("os.path.exists", return_value=False):
            result = ModelPathResolver.resolve("")
            assert result == MODEL_NAME
