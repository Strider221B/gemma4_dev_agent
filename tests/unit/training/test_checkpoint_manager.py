"""Unit tests for CheckpointManager verifying adapter save, load, and size validation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.training.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test suite covering CheckpointManager capabilities and error cases."""

    _SIZE_LIMIT_BYTES: int = 1_000
    _SMALL_PAYLOAD: bytes = b"weights_data"
    _LARGE_PAYLOAD: bytes = b"x" * 2_000
    _CONFIG_NAME: str = "adapter_config.json"
    _WEIGHTS_NAME: str = "adapter_model.safetensors"
    _FORBIDDEN_NAME: str = "pytorch_model.bin"
    _NON_EXISTENT_DIR: str = "/non/existent/checkpoint/directory"

    def test_validate_size_under_limit_returns_true(self, tmp_path: Path) -> None:
        """Verify directory under size limit passes validation."""
        manager = CheckpointManager(max_size_bytes=self._SIZE_LIMIT_BYTES)
        (tmp_path / self._WEIGHTS_NAME).write_bytes(self._SMALL_PAYLOAD)
        assert manager.validate_size(str(tmp_path)) is True

    def test_validate_size_over_limit_returns_false(self, tmp_path: Path) -> None:
        """Verify directory exceeding size limit fails validation."""
        manager = CheckpointManager(max_size_bytes=self._SIZE_LIMIT_BYTES)
        (tmp_path / self._WEIGHTS_NAME).write_bytes(self._LARGE_PAYLOAD)
        assert manager.validate_size(str(tmp_path)) is False

    def test_verify_safetensors_with_valid_files(self, tmp_path: Path) -> None:
        """Verify directory with config and safetensors files is accepted."""
        manager = CheckpointManager()
        (tmp_path / self._CONFIG_NAME).write_text("{}")
        (tmp_path / self._WEIGHTS_NAME).write_bytes(self._SMALL_PAYLOAD)
        assert manager._verify_safetensors(str(tmp_path)) is True

    def test_verify_safetensors_with_forbidden_files(self, tmp_path: Path) -> None:
        """Verify directory containing legacy forbidden weight formats is rejected."""
        manager = CheckpointManager()
        (tmp_path / self._CONFIG_NAME).write_text("{}")
        (tmp_path / self._WEIGHTS_NAME).write_bytes(self._SMALL_PAYLOAD)
        (tmp_path / self._FORBIDDEN_NAME).write_bytes(self._SMALL_PAYLOAD)
        assert manager._verify_safetensors(str(tmp_path)) is False

    def test_verify_safetensors_missing_components(self, tmp_path: Path) -> None:
        """Verify verification fails when required files are absent."""
        manager = CheckpointManager()
        assert manager._verify_safetensors(str(tmp_path)) is False

    def test_list_checkpoints_returns_dirs(self, tmp_path: Path) -> None:
        """Verify checkpoint listing discovers directories and ignores files."""
        manager = CheckpointManager()
        step_1 = tmp_path / "checkpoint-100"
        step_2 = tmp_path / "checkpoint-200"
        step_1.mkdir()
        step_2.mkdir()
        (tmp_path / "summary.txt").write_text("done")

        checkpoints = manager.list_checkpoints(str(tmp_path))
        assert len(checkpoints) == 2
        assert str(step_1) in checkpoints
        assert str(step_2) in checkpoints

    def test_list_checkpoints_non_existent_directory(self) -> None:
        """Verify listing on non-existent directory returns empty list."""
        manager = CheckpointManager()
        assert manager.list_checkpoints(self._NON_EXISTENT_DIR) == []

    def test_compute_total_size_correct(self, tmp_path: Path) -> None:
        """Verify accurate recursive byte calculation."""
        manager = CheckpointManager()
        sub_dir = tmp_path / "nested"
        sub_dir.mkdir()
        (tmp_path / "a.bin").write_bytes(b"12345")
        (sub_dir / "b.bin").write_bytes(b"67890")
        assert manager._compute_total_size(str(tmp_path)) == 10

    def test_compute_total_size_single_file(self, tmp_path: Path) -> None:
        """Verify single file size calculation."""
        manager = CheckpointManager()
        file_path = tmp_path / "test.safetensors"
        file_path.write_bytes(b"1234")
        assert manager._compute_total_size(str(file_path)) == 4

    def test_compute_total_size_non_existent(self) -> None:
        """Verify non-existent path yields size zero."""
        manager = CheckpointManager()
        assert manager._compute_total_size(self._NON_EXISTENT_DIR) == 0

    def test_check_forbidden_files_non_existent(self) -> None:
        """Verify non-existent path returns empty forbidden list."""
        manager = CheckpointManager()
        assert manager._check_forbidden_files(self._NON_EXISTENT_DIR) == []

    def test_save_adapter_success(self, tmp_path: Path) -> None:
        """Verify successful adapter save and validation flow."""
        manager = CheckpointManager(max_size_bytes=self._SIZE_LIMIT_BYTES)
        dest = tmp_path / "saved_adapter"

        def _mock_save(path: str, safe_serialization: bool = True) -> None:
            p = Path(path)
            (p / self._CONFIG_NAME).write_text("{}")
            (p / self._WEIGHTS_NAME).write_bytes(self._SMALL_PAYLOAD)

        mock_model = MagicMock()
        mock_model.save_pretrained.side_effect = _mock_save

        saved_path = manager.save_adapter(mock_model, str(dest))
        assert saved_path == str(dest)
        mock_model.save_pretrained.assert_called_once_with(
            str(dest), safe_serialization=True
        )

    def test_save_adapter_verification_failure_raises(self, tmp_path: Path) -> None:
        """Verify ValueError raised when safetensors verification fails."""
        manager = CheckpointManager()
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="Safetensors verification failed"):
            manager.save_adapter(mock_model, str(tmp_path))

    def test_save_adapter_size_exceeded_raises(self, tmp_path: Path) -> None:
        """Verify ValueError raised when saved checkpoint exceeds size threshold."""
        manager = CheckpointManager(max_size_bytes=10)

        def _mock_save(path: str, safe_serialization: bool = True) -> None:
            p = Path(path)
            (p / self._CONFIG_NAME).write_text("{}")
            (p / self._WEIGHTS_NAME).write_bytes(b"large_data_exceeding_10_bytes")

        mock_model = MagicMock()
        mock_model.save_pretrained.side_effect = _mock_save

        with pytest.raises(ValueError, match="size exceeds maximum limit"):
            manager.save_adapter(mock_model, str(tmp_path))

    def test_load_adapter_calls_peft(self) -> None:
        """Verify PEFT from_pretrained is invoked with expected arguments."""
        manager = CheckpointManager()
        mock_peft = MagicMock()
        mock_model = MagicMock()
        mock_loaded = MagicMock()
        mock_peft.PeftModel.from_pretrained.return_value = mock_loaded

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "peft", mock_peft)
            result = manager.load_adapter(mock_model, "/path/to/adapter")
            assert result == mock_loaded
            mock_peft.PeftModel.from_pretrained.assert_called_once_with(
                mock_model, "/path/to/adapter"
            )
