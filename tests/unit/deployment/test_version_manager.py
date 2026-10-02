"""Unit tests for VersionManager."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.deployment.version_manager import VersionManager


class TestVersionManager:
    """Test suite for VersionManager version bumping and release orchestration."""

    _VERSION_FILE: str = "VERSION"
    _CHANGELOG_FILE: str = "CHANGELOG.md"
    _DEPLOYMENT_LOG: str = "deployment.log"
    _INITIAL_VERSION: str = "0.1.0"
    _BUMP_MSG: str = "Add new feature"
    _PATCH_VERSION: str = "0.1.1"
    _MINOR_VERSION: str = "0.2.0"
    _MAJOR_VERSION: str = "1.0.0"
    _INVALID_BUMP: str = "super_major"

    @pytest.fixture
    def staging_dir(self, tmp_path: Path) -> Path:
        """Create a staging directory fixture with an initial VERSION file."""
        staging = tmp_path / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / self._VERSION_FILE).write_text(self._INITIAL_VERSION, encoding="utf-8")
        return staging

    def test_bump_patch_increments(self, staging_dir: Path) -> None:
        """Verify patch bump correctly increments the patch digit."""
        mgr = VersionManager(str(staging_dir))
        new_version = mgr.bump("patch", self._BUMP_MSG)
        assert new_version == self._PATCH_VERSION
        current_file_ver = (staging_dir / self._VERSION_FILE).read_text(encoding="utf-8").strip()
        assert current_file_ver == self._PATCH_VERSION
        changelog = (staging_dir / self._CHANGELOG_FILE).read_text(encoding="utf-8")
        assert self._PATCH_VERSION in changelog
        assert self._BUMP_MSG in changelog

    def test_bump_minor_increments(self, staging_dir: Path) -> None:
        """Verify minor bump increments minor digit and resets patch digit."""
        mgr = VersionManager(str(staging_dir))
        new_version = mgr.bump("minor", self._BUMP_MSG)
        assert new_version == self._MINOR_VERSION
        current_file_ver = (staging_dir / self._VERSION_FILE).read_text(encoding="utf-8").strip()
        assert current_file_ver == self._MINOR_VERSION

    def test_bump_major_increments(self, staging_dir: Path) -> None:
        """Verify major bump increments major digit and resets minor and patch digits."""
        mgr = VersionManager(str(staging_dir))
        new_version = mgr.bump("major", self._BUMP_MSG)
        assert new_version == self._MAJOR_VERSION
        current_file_ver = (staging_dir / self._VERSION_FILE).read_text(encoding="utf-8").strip()
        assert current_file_ver == self._MAJOR_VERSION

    def test_bump_invalid_type_raises_error(self, staging_dir: Path) -> None:
        """Verify invalid bump type raises ValueError."""
        mgr = VersionManager(str(staging_dir))
        with pytest.raises(ValueError, match="Invalid bump_type"):
            mgr.bump(self._INVALID_BUMP, self._BUMP_MSG)

    def test_get_current_version_reads_file(self, staging_dir: Path) -> None:
        """Verify get_current_version accurately reads content from VERSION file."""
        mgr = VersionManager(str(staging_dir))
        assert mgr.get_current_version() == self._INITIAL_VERSION

    def test_push_calls_kaggle_cli(self, staging_dir: Path) -> None:
        """Verify push constructs correct Kaggle CLI invocation and logs deployment."""
        mgr = VersionManager(str(staging_dir))
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Version created"
        with patch.object(subprocess, "run", return_value=mock_proc) as mock_run:
            mgr.push(str(staging_dir), self._INITIAL_VERSION, self._BUMP_MSG)
            mock_run.assert_called_once()
            called_cmd = mock_run.call_args[0][0]
            assert called_cmd[0] == "kaggle"
            assert called_cmd[1] == "datasets"
            assert called_cmd[2] == "version"
            assert str(staging_dir) in called_cmd
            assert f"v{self._INITIAL_VERSION}: {self._BUMP_MSG}" in called_cmd

        log_content = (staging_dir / self._DEPLOYMENT_LOG).read_text(encoding="utf-8")
        assert self._INITIAL_VERSION in log_content
        assert self._BUMP_MSG in log_content

    def test_push_failure_raises_runtime_error(self, staging_dir: Path) -> None:
        """Verify failed Kaggle CLI command raises RuntimeError."""
        mgr = VersionManager(str(staging_dir))
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Authentication failed"
        with patch.object(subprocess, "run", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="Kaggle push failed"):
                mgr.push(str(staging_dir), self._INITIAL_VERSION, self._BUMP_MSG)

    def test_read_version_missing_file_returns_default(self, tmp_path: Path) -> None:
        """Verify missing VERSION file returns default 0.1.0 version."""
        empty_dir = tmp_path / "empty_staging"
        mgr = VersionManager(str(empty_dir))
        assert mgr.get_current_version() == "0.1.0"

    def test_parse_version_invalid_format_raises_error(self, staging_dir: Path) -> None:
        """Verify invalid version string raises ValueError."""
        (staging_dir / self._VERSION_FILE).write_text("invalid_version", encoding="utf-8")
        mgr = VersionManager(str(staging_dir))
        with pytest.raises(ValueError, match="Invalid semantic version format"):
            mgr.bump("patch", "msg")
