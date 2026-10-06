"""Unit tests for LocalPipelineRunner."""

from __future__ import annotations

import os
import tempfile
from typing import ClassVar
from unittest.mock import patch

import pytest

from src.training.local_pipeline_runner import LocalPipelineRunner
from src.utils.environment_detector import EnvironmentDetector


class TestLocalPipelineRunner:
    """Test suite covering LocalPipelineRunner mode dispatching, mock execution, and CLI."""

    _CUSTOM_WORK_DIR: ClassVar[str] = "/custom/pipeline/work"
    _SURROGATE_PATH: ClassVar[str] = "/custom/pipeline/surrogate"
    _MODE_SMOKE: ClassVar[str] = "smoke"
    _MODE_MOCK: ClassVar[str] = "mock"
    _MODE_FULL: ClassVar[str] = "full"
    _MODE_INVALID: ClassVar[str] = "invalid_mode"
    _STATUS_KEY: ClassVar[str] = "status"
    _STATUS_SUCCESS: ClassVar[str] = "success"
    _ZIP_KEY: ClassVar[str] = "submission_zip"
    _ADAPTER_KEY: ClassVar[str] = "adapter_path"
    _ARG_MODE: ClassVar[str] = "--mode"
    _ARG_WORK_DIR: ClassVar[str] = "--work-dir"

    def test_instantiation_with_defaults(self) -> None:
        """Verify LocalPipelineRunner instantiates with default dependencies."""
        runner = LocalPipelineRunner()
        assert isinstance(runner, LocalPipelineRunner)

    def test_instantiation_with_custom_work_dir(self) -> None:
        """Verify LocalPipelineRunner preserves injected work directory."""
        detector = EnvironmentDetector()
        runner = LocalPipelineRunner(env_detector=detector, work_dir=self._CUSTOM_WORK_DIR)
        assert runner._work_dir == self._CUSTOM_WORK_DIR

    def test_run_dispatches_smoke_mode(self) -> None:
        """Verify run dispatches to run_smoke when mode is smoke."""
        runner = LocalPipelineRunner(work_dir=self._CUSTOM_WORK_DIR)
        mock_ret = {self._STATUS_KEY: self._STATUS_SUCCESS}
        with patch.object(runner, "run_smoke", return_value=mock_ret) as mock_smoke:
            result = runner.run(mode=self._MODE_SMOKE)
            assert result[self._STATUS_KEY] == self._STATUS_SUCCESS
            mock_smoke.assert_called_once()

    def test_run_dispatches_mock_mode(self) -> None:
        """Verify run dispatches to run_mock when mode is mock."""
        runner = LocalPipelineRunner(work_dir=self._CUSTOM_WORK_DIR)
        mock_ret = {self._STATUS_KEY: self._STATUS_SUCCESS}
        with patch.object(runner, "run_mock", return_value=mock_ret) as mock_mock:
            result = runner.run(mode=self._MODE_MOCK)
            assert result[self._STATUS_KEY] == self._STATUS_SUCCESS
            mock_mock.assert_called_once()

    def test_run_unknown_mode_raises_value_error(self) -> None:
        """Verify run raises ValueError on unsupported mode identifier."""
        runner = LocalPipelineRunner(work_dir=self._CUSTOM_WORK_DIR)
        with pytest.raises(ValueError, match="Unknown execution mode"):
            runner.run(mode=self._MODE_INVALID)

    def test_run_full_mode_raises_not_implemented(self) -> None:
        """Verify run raises NotImplementedError when mode is full."""
        runner = LocalPipelineRunner(work_dir=self._CUSTOM_WORK_DIR)
        with pytest.raises(NotImplementedError, match="Kaggle GPU cluster"):
            runner.run(mode=self._MODE_FULL)

    def test_run_mock_creates_submission_and_returns_summary(self) -> None:
        """Verify run_mock generates valid mock adapter, zip artifact, and report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LocalPipelineRunner(work_dir=tmpdir)
            summary = runner.run_mock()
            assert summary[self._STATUS_KEY] == self._STATUS_SUCCESS
            zip_path = str(summary[self._ZIP_KEY])
            adapter_path = str(summary[self._ADAPTER_KEY])
            assert os.path.isfile(zip_path)
            assert os.path.isdir(adapter_path)

    def test_build_smoke_config_hyperparameters(self) -> None:
        """Verify _build_smoke_config configures lightweight parameters."""
        runner = LocalPipelineRunner(work_dir=self._CUSTOM_WORK_DIR)
        sft_cfg = runner._build_smoke_config(self._SURROGATE_PATH)
        assert sft_cfg.model.name == self._SURROGATE_PATH
        assert sft_cfg.model.load_in_4bit is False
        assert sft_cfg.training.num_epochs == 1
        assert sft_cfg.training.bf16 is False
        assert sft_cfg.training.curriculum_enabled is False

    def test_init_workspace_creates_subdirectories(self) -> None:
        """Verify _init_workspace creates all required lifecycle subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LocalPipelineRunner(work_dir=tmpdir)
            runner._init_workspace()
            for subdir in ("surrogate", "checkpoints", "logs", "submission"):
                assert os.path.isdir(os.path.join(tmpdir, subdir))

    def test_cli_main_executes_successfully(self) -> None:
        """Verify class main method parses arguments and executes pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [self._ARG_MODE, self._MODE_MOCK, self._ARG_WORK_DIR, tmpdir]
            exit_code = LocalPipelineRunner.main(args)
            assert exit_code == 0
