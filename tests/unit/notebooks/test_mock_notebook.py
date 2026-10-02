"""Unit tests for MockNotebook local pipeline runner."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import ClassVar

import pytest

from notebooks.mock_notebook import MockNotebook
from src.training.checkpoint_manager import CheckpointManager


class TestMockNotebook:
    """Test suite verifying MockNotebook execution, checkpointing, and tasks."""

    _MIN_SPLIT_TASKS: ClassVar[int] = 2
    _MIN_UNIQUE_REPOS: ClassVar[int] = 2
    _ADAPTER_DIR_NAME: ClassVar[str] = "sft_lora"
    _ADAPTER_CONFIG_NAME: ClassVar[str] = "adapter_config.json"
    _ADAPTER_WEIGHTS_NAME: ClassVar[str] = "adapter_model.safetensors"

    @pytest.fixture
    def temp_work_dir(self) -> str:
        """Provide a temporary working directory for test execution."""
        temp_dir = tempfile.mkdtemp(prefix="test_mock_nb_")
        yield temp_dir
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_mock_notebook_run_generates_valid_zip(self, temp_work_dir: str) -> None:
        """Verify run() executes end-to-end and outputs non-empty submission.zip."""
        runner = MockNotebook(work_dir=temp_work_dir)
        zip_path = runner.run()
        assert os.path.isfile(zip_path)
        assert os.path.getsize(zip_path) > 0

    def test_mock_notebook_create_split_tasks_contains_multiple_repos(self) -> None:
        """Verify created split tasks contain sufficient unique repositories for CV."""
        runner = MockNotebook()
        tasks = runner._create_split_tasks()
        assert len(tasks) >= self._MIN_SPLIT_TASKS
        unique_repos = {task.repo for task in tasks}
        assert len(unique_repos) >= self._MIN_UNIQUE_REPOS

    def test_mock_notebook_create_mock_checkpoint_passes_verification(
        self, temp_work_dir: str
    ) -> None:
        """Verify mock checkpoint creation produces compliant safetensors artifacts."""
        runner = MockNotebook(work_dir=temp_work_dir)
        adapter_path = os.path.join(temp_work_dir, self._ADAPTER_DIR_NAME)
        runner._create_mock_checkpoint(adapter_path)
        config_file = Path(adapter_path) / self._ADAPTER_CONFIG_NAME
        weights_file = Path(adapter_path) / self._ADAPTER_WEIGHTS_NAME
        assert config_file.is_file()
        assert weights_file.is_file()
        mgr = CheckpointManager()
        assert mgr.validate_size(adapter_path)

    def test_mock_notebook_cleanup_removes_owned_work_dir(self) -> None:
        """Verify runner automatically cleans up temporary directory if self-owned."""
        runner = MockNotebook()
        work_dir = runner._work_dir
        assert os.path.isdir(work_dir)
        runner._cleanup()
        assert not os.path.exists(work_dir)
