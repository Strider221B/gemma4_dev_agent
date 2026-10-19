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
    _LOCAL_TEST_NAME: ClassVar[str] = "google/gemma-4-e2b-it-qat-w4a16-ct"
    _LOCAL_TEST_SLUG: ClassVar[str] = "gemma-4-e2b-it-qat-w4a16-ct"
    _LOCAL_DIR_CANDIDATE: ClassVar[str] = "models/gemma-4-e2b-it-qat-w4a16-ct"
    _ENV_PATH: ClassVar[str] = "/env/path/to/model"
    _ENV_LOCAL_PATH: ClassVar[str] = "/env/path/to/e2b_model"

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

    def test_resolve_local_test_model_resolves_to_local_candidate_when_present(self) -> None:
        """Verify local test model resolves to local directory candidate if present."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == self._LOCAL_DIR_CANDIDATE
                result = ModelPathResolver.resolve(self._LOCAL_TEST_NAME)
                assert result == self._LOCAL_DIR_CANDIDATE

    def test_resolve_env_var_local_model_path_when_exists(self) -> None:
        """Verify resolution checks LOCAL_MODEL_PATH environment variable for test model."""
        with patch.dict(os.environ, {"LOCAL_MODEL_PATH": self._ENV_LOCAL_PATH}, clear=True):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == self._ENV_LOCAL_PATH
                result = ModelPathResolver.resolve(self._LOCAL_TEST_SLUG)
                assert result == self._ENV_LOCAL_PATH

    def test_resolve_env_var_gemma_e2b_path_when_exists(self) -> None:
        """Verify resolution checks GEMMA_E2B_MODEL_PATH environment variable."""
        with patch.dict(os.environ, {"GEMMA_E2B_MODEL_PATH": self._ENV_LOCAL_PATH}, clear=True):
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p == self._ENV_LOCAL_PATH
                result = ModelPathResolver.resolve(self._LOCAL_TEST_NAME)
                assert result == self._ENV_LOCAL_PATH

    def test_resolve_local_test_model_fallback_when_not_found(self) -> None:
        """Verify local test model identifier returns as-is when no local candidates exist."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                result = ModelPathResolver.resolve(self._LOCAL_TEST_NAME)
                assert result == self._LOCAL_TEST_NAME

    def test_has_local_model_returns_true_when_directory_exists(self) -> None:
        """Verify has_local_model returns True when resolved path is an existing directory."""
        with patch.object(ModelPathResolver, "resolve", return_value="/mock/local/path"):
            with patch("os.path.isdir", return_value=True):
                assert ModelPathResolver.has_local_model(self._LOCAL_TEST_NAME) is True

    def test_has_local_model_returns_false_when_not_directory(self) -> None:
        """Verify has_local_model returns False when resolved path does not exist as directory."""
        with patch.object(ModelPathResolver, "resolve", return_value=self._LOCAL_TEST_NAME):
            with patch("os.path.isdir", return_value=False):
                assert ModelPathResolver.has_local_model(self._LOCAL_TEST_NAME) is False

    def test_is_supported_model_identifies_competition_and_local_models(self) -> None:
        """Verify _is_supported_model recognizes competition and local test models."""
        assert ModelPathResolver._is_supported_model(MODEL_NAME) is True
        assert ModelPathResolver._is_supported_model(self._COMPETITION_SLUG) is True
        assert ModelPathResolver._is_supported_model(self._LOCAL_TEST_NAME) is True
        assert ModelPathResolver._is_supported_model(self._LOCAL_TEST_SLUG) is True
        assert ModelPathResolver._is_supported_model(self._CUSTOM_REPO_ID) is False
