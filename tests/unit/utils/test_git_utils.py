"""Unit tests for GitUtils."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.utils.git_utils import GitUtils


class TestGitUtils:
    """Test suite for GitUtils diff parser and repository actions."""

    _SINGLE_DIFF: str = (
        "diff --git a/src/example.py b/src/example.py\n"
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1,3 +1,4 @@\n"
        " existing line\n"
        "-old line\n"
        "+new line 1\n"
        "+new line 2\n"
    )
    _MULTI_DIFF: str = (
        "diff --git a/file1.py b/file1.py\n"
        "--- a/file1.py\n"
        "+++ b/file1.py\n"
        "@@ -1,1 +1,2 @@\n"
        "+added line\n"
        "diff --git a/file2.py b/file2.py\n"
        "--- a/file2.py\n"
        "+++ b/file2.py\n"
        "@@ -1,2 +1,1 @@\n"
        "-deleted line\n"
    )
    _SAMPLE_WORKSPACE: str = "/mock/workspace"
    _SAMPLE_FILE: str = "src/example.py"

    def test_parse_unified_diff_single_file(self) -> None:
        """Verify parsing single file diff correctly extracts path and change counts."""
        git_utils = GitUtils()
        result = git_utils.parse_unified_diff(self._SINGLE_DIFF)
        assert len(result) == 1
        entry = result[0]
        assert entry["old_path"] == "a/src/example.py"
        assert entry["new_path"] == "b/src/example.py"
        assert entry["added_lines"] == 2
        assert entry["deleted_lines"] == 1

    def test_parse_unified_diff_multi_file(self) -> None:
        """Verify parsing multi-file unified diff produces distinct entries."""
        git_utils = GitUtils()
        result = git_utils.parse_unified_diff(self._MULTI_DIFF)
        assert len(result) == 2
        assert result[0]["old_path"] == "a/file1.py"
        assert result[0]["added_lines"] == 1
        assert result[0]["deleted_lines"] == 0
        assert result[1]["old_path"] == "a/file2.py"
        assert result[1]["added_lines"] == 0
        assert result[1]["deleted_lines"] == 1

    def test_parse_empty_diff_returns_empty_list(self) -> None:
        """Verify empty or whitespace-only diff strings yield empty list."""
        git_utils = GitUtils()
        assert git_utils.parse_unified_diff("") == []
        assert git_utils.parse_unified_diff("   \n  ") == []

    @patch("subprocess.run")
    def test_apply_patch_success_and_empty(self, mock_run: MagicMock) -> None:
        """Verify applying patch calls git apply subprocess and handles empty patch."""
        git_utils = GitUtils()
        assert git_utils.apply_patch(self._SAMPLE_WORKSPACE, "") is True

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        success = git_utils.apply_patch(self._SAMPLE_WORKSPACE, self._SINGLE_DIFF)
        assert success is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_list_changed_files(self, mock_run: MagicMock) -> None:
        """Verify list_changed_files parses stdout lines correctly."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="file1.py\nfile2.py\n", stderr=""
        )
        git_utils = GitUtils()
        files = git_utils.list_changed_files(self._SAMPLE_WORKSPACE)
        assert files == ["file1.py", "file2.py"]

    @patch("subprocess.run")
    def test_checkout_file(self, mock_run: MagicMock) -> None:
        """Verify checkout_file executes git checkout command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        git_utils = GitUtils()
        success = git_utils.checkout_file(self._SAMPLE_WORKSPACE, self._SAMPLE_FILE)
        assert success is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_apply_patch_failure(self, mock_run: MagicMock) -> None:
        """Verify apply_patch returns False when git apply returns non-zero code."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        git_utils = GitUtils()
        assert git_utils.apply_patch(self._SAMPLE_WORKSPACE, self._SINGLE_DIFF) is False

    @patch("subprocess.run")
    def test_list_changed_files_failure(self, mock_run: MagicMock) -> None:
        """Verify list_changed_files returns empty list on command failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        git_utils = GitUtils()
        assert git_utils.list_changed_files(self._SAMPLE_WORKSPACE) == []

    @patch("subprocess.run")
    def test_run_git_command_os_error(self, mock_run: MagicMock) -> None:
        """Verify _run_git_command catches OSError and returns failure code."""
        mock_run.side_effect = OSError("git not found")
        git_utils = GitUtils()
        code, stdout, stderr = git_utils._run_git_command(self._SAMPLE_WORKSPACE, ["status"])
        assert code == 1
        assert stdout == ""
        assert "Failed" in stderr
