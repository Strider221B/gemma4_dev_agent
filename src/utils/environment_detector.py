"""Environment detector for differentiating Kaggle and local execution."""

from __future__ import annotations

import os
from pathlib import Path


class EnvironmentDetector:
    """Detects whether execution is within Kaggle or a local environment."""

    _KAGGLE_DIR: str = "/kaggle"
    _KAGGLE_WORKING_DIR: str = "/kaggle/working"
    _KAGGLE_INPUT_DIR: str = "/kaggle/input"
    _LOCAL_WORKING_DIR: str = "./kaggle_working"
    _LOCAL_INPUT_DIR: str = "./kaggle_staging"
    _ENV_KAGGLE_KERNEL: str = "KAGGLE_KERNEL_RUN_TYPE"
    _ENV_WORKING_DIR_OVERRIDE: str = "SWEGEMMA_WORKING_DIR"
    _SUBMISSION_ZIP_NAME: str = "submission.zip"
    _SUBMISSION_DIR_NAME: str = "submission"
    _LOGS_DIR_NAME: str = "logs"
    _CHECKPOINTS_DIR_NAME: str = "checkpoints"

    def __init__(self) -> None:
        """Initialize EnvironmentDetector."""

    def is_kaggle(self) -> bool:
        """Check if execution is running inside a Kaggle kernel environment.

        Returns:
            True if /kaggle/working exists or KAGGLE_KERNEL_RUN_TYPE env var is set.
        """
        return (
            os.path.exists(self._KAGGLE_WORKING_DIR)
            or self._ENV_KAGGLE_KERNEL in os.environ
        )

    def get_working_dir(self) -> str:
        """Return the working directory path, supporting env override.

        Returns:
            Resolved working directory path string.
        """
        override = os.environ.get(self._ENV_WORKING_DIR_OVERRIDE)
        if override:
            return override
        return self._resolve_base_working_dir()

    def get_input_dir(self) -> str:
        """Return the input datasets directory path.

        Returns:
            Resolved input directory path string.
        """
        return self._resolve_base_input_dir()

    def get_logs_dir(self) -> str:
        """Return the directory path for logs under the active working directory.

        Returns:
            Resolved logs directory path string.
        """
        return str(Path(self.get_working_dir()) / self._LOGS_DIR_NAME)

    def get_checkpoints_dir(self) -> str:
        """Return the directory path for checkpoints under the working directory.

        Returns:
            Resolved checkpoints directory path string.
        """
        return str(Path(self.get_working_dir()) / self._CHECKPOINTS_DIR_NAME)

    def get_submission_dir(self) -> str:
        """Return the directory path for submission artifacts under the working directory.

        Returns:
            Resolved submission directory path string.
        """
        return str(Path(self.get_working_dir()) / self._SUBMISSION_DIR_NAME)

    def get_submission_zip_path(self) -> str:
        """Return the file path for the packaged submission zip file.

        Returns:
            Resolved submission.zip file path string.
        """
        return str(Path(self.get_working_dir()) / self._SUBMISSION_ZIP_NAME)

    def _resolve_base_working_dir(self) -> str:
        """Resolve default working directory based on Kaggle detection."""
        if self.is_kaggle():
            return self._KAGGLE_WORKING_DIR
        return self._LOCAL_WORKING_DIR

    def _resolve_base_input_dir(self) -> str:
        """Resolve default input directory based on Kaggle detection."""
        if self.is_kaggle():
            return self._KAGGLE_INPUT_DIR
        return self._LOCAL_INPUT_DIR
