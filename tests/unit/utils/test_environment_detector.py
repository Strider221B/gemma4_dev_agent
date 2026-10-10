"""Unit tests for EnvironmentDetector."""

from __future__ import annotations

import os
from typing import ClassVar
from unittest.mock import patch

from src.utils.environment_detector import EnvironmentDetector


class TestEnvironmentDetector:
    """Test suite for EnvironmentDetector Kaggle vs local execution detection."""

    _CUSTOM_WORK_DIR: ClassVar[str] = "/custom/work"
    _KAGGLE_WORKING: ClassVar[str] = "/kaggle/working"
    _KAGGLE_INPUT: ClassVar[str] = "/kaggle/input"
    _LOCAL_WORKING: ClassVar[str] = "./kaggle_working"
    _LOCAL_INPUT: ClassVar[str] = "./kaggle_staging"
    _ENV_KERNEL_KEY: ClassVar[str] = "KAGGLE_KERNEL_RUN_TYPE"
    _ENV_KERNEL_VAL: ClassVar[str] = "Interactive"
    _ENV_OVERRIDE_KEY: ClassVar[str] = "SWEGEMMA_WORKING_DIR"
    _LOGS_SUFFIX: ClassVar[str] = "/custom/work/logs"
    _CHECKPOINTS_SUFFIX: ClassVar[str] = "/custom/work/checkpoints"
    _SUBMISSION_SUFFIX: ClassVar[str] = "/custom/work/submission"
    _SUBMISSION_ZIP_SUFFIX: ClassVar[str] = "/custom/work/submission.zip"

    def test_instantiation_succeeds(self) -> None:
        """Verify EnvironmentDetector instantiates successfully."""
        detector = EnvironmentDetector()
        assert isinstance(detector, EnvironmentDetector)

    def test_is_kaggle_when_kaggle_working_dir_exists(self) -> None:
        """Verify is_kaggle returns True when /kaggle/working directory exists."""
        detector = EnvironmentDetector()
        with patch("os.path.exists", return_value=True):
            with patch.dict(os.environ, {}, clear=True):
                assert detector.is_kaggle() is True

    def test_is_kaggle_when_env_var_set(self) -> None:
        """Verify is_kaggle returns True when KAGGLE_KERNEL_RUN_TYPE is set."""
        detector = EnvironmentDetector()
        with patch("os.path.exists", return_value=False):
            with patch.dict(os.environ, {self._ENV_KERNEL_KEY: self._ENV_KERNEL_VAL}):
                assert detector.is_kaggle() is True

    def test_is_kaggle_when_neither_present(self) -> None:
        """Verify is_kaggle returns False when neither directory nor env var is present."""
        detector = EnvironmentDetector()
        with patch("os.path.exists", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                assert detector.is_kaggle() is False

    def test_get_working_dir_returns_override_when_set(self) -> None:
        """Verify get_working_dir honors SWEGEMMA_WORKING_DIR environment override."""
        detector = EnvironmentDetector()
        with patch.dict(os.environ, {self._ENV_OVERRIDE_KEY: self._CUSTOM_WORK_DIR}):
            assert detector.get_working_dir() == self._CUSTOM_WORK_DIR

    def test_get_working_dir_returns_kaggle_when_kaggle_env(self) -> None:
        """Verify get_working_dir defaults to /kaggle/working in Kaggle environment."""
        detector = EnvironmentDetector()
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(detector, "is_kaggle", return_value=True):
                assert detector.get_working_dir() == self._KAGGLE_WORKING

    def test_get_working_dir_returns_local_when_not_kaggle(self) -> None:
        """Verify get_working_dir defaults to ./kaggle_working in local environment."""
        detector = EnvironmentDetector()
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(detector, "is_kaggle", return_value=False):
                assert detector.get_working_dir() == self._LOCAL_WORKING

    def test_get_input_dir_returns_kaggle_when_kaggle_env(self) -> None:
        """Verify get_input_dir returns /kaggle/input in Kaggle environment."""
        detector = EnvironmentDetector()
        with patch.object(detector, "is_kaggle", return_value=True):
            assert detector.get_input_dir() == self._KAGGLE_INPUT

    def test_get_input_dir_returns_local_when_not_kaggle(self) -> None:
        """Verify get_input_dir returns ./kaggle_staging in local environment."""
        detector = EnvironmentDetector()
        with patch.object(detector, "is_kaggle", return_value=False):
            assert detector.get_input_dir() == self._LOCAL_INPUT

    def test_get_logs_dir_returns_path_under_working_dir(self) -> None:
        """Verify get_logs_dir appends logs directory under resolved working dir."""
        detector = EnvironmentDetector()
        with patch.object(detector, "get_working_dir", return_value=self._CUSTOM_WORK_DIR):
            assert detector.get_logs_dir() == self._LOGS_SUFFIX

    def test_get_checkpoints_dir_returns_path_under_working_dir(self) -> None:
        """Verify get_checkpoints_dir appends checkpoints directory under working dir."""
        detector = EnvironmentDetector()
        with patch.object(detector, "get_working_dir", return_value=self._CUSTOM_WORK_DIR):
            assert detector.get_checkpoints_dir() == self._CHECKPOINTS_SUFFIX

    def test_get_submission_dir_returns_path_under_working_dir(self) -> None:
        """Verify get_submission_dir appends submission directory under working dir."""
        detector = EnvironmentDetector()
        with patch.object(detector, "get_working_dir", return_value=self._CUSTOM_WORK_DIR):
            assert detector.get_submission_dir() == self._SUBMISSION_SUFFIX

    def test_get_submission_zip_path_returns_path_under_working_dir(self) -> None:
        """Verify get_submission_zip_path appends submission.zip under working dir."""
        detector = EnvironmentDetector()
        with patch.object(detector, "get_working_dir", return_value=self._CUSTOM_WORK_DIR):
            assert detector.get_submission_zip_path() == self._SUBMISSION_ZIP_SUFFIX
