"""Unit tests for MockSandbox."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from src.data.task import Task
from src.evaluation.mock_sandbox import MockSandbox
from src.utils.git_utils import GitUtils


class TestMockSandbox:
    """Test suite for MockSandbox isolation, git operations, and execution."""

    _MOCK_COMMIT: str = "abcdef1234567890"
    _MOCK_CREATED: str = "2026-01-01T00:00:00Z"
    _SAMPLE_PATCH: str = (
        "--- a/utils.py\n"
        "+++ b/utils.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def add_numbers(a, b):\n"
        "+def add_numbers(x, y):\n"
        "     return a + b\n"
    )
    _SAMPLE_TEST_PATCH: str = (
        "--- /dev/null\n+++ b/tests/test_foo.py\n@@ -0,0 +1,2 @@\n"
        "+def test_pass():\n+    pass\n"
    )

    def test_setup_workspace_creates_directory(self) -> None:
        """Verify setup_workspace initializes a directory with git repository."""
        sandbox = MockSandbox()
        task = self._create_task()
        try:
            workspace_dir = sandbox.setup_workspace(task)
            assert os.path.exists(workspace_dir)
            assert os.path.isdir(workspace_dir)
            assert os.path.exists(os.path.join(workspace_dir, ".git"))
            assert os.path.exists(os.path.join(workspace_dir, "utils.py"))
        finally:
            sandbox.cleanup()

    def test_apply_patch_returns_true_for_valid(self) -> None:
        """Verify apply_patch returns True on clean git apply."""
        sandbox = MockSandbox()
        task = self._create_task()
        try:
            sandbox.setup_workspace(task)
            res = sandbox.apply_patch(self._SAMPLE_PATCH)
            assert res is True
        finally:
            sandbox.cleanup()

    def test_apply_patch_empty_patch_returns_true(self) -> None:
        """Verify applying empty patch string returns True without error."""
        sandbox = MockSandbox()
        task = self._create_task()
        try:
            sandbox.setup_workspace(task)
            assert sandbox.apply_patch("") is True
        finally:
            sandbox.cleanup()

    def test_cleanup_removes_workspace(self) -> None:
        """Verify cleanup deletes temporary workspace from filesystem."""
        sandbox = MockSandbox()
        task = self._create_task()
        workspace_dir = sandbox.setup_workspace(task)
        assert os.path.exists(workspace_dir)
        sandbox.cleanup()
        assert not os.path.exists(workspace_dir)
        assert sandbox._workspace_path is None

    def test_run_command_executes_successfully(self) -> None:
        """Verify run_command executes command and returns exit code and output."""
        sandbox = MockSandbox()
        task = self._create_task()
        try:
            sandbox.setup_workspace(task)
            res = sandbox.run_command("python3 -c 'print(\"hello\")'")
            assert res["exit_code"] == 0
            assert "hello" in str(res["stdout"])
        finally:
            sandbox.cleanup()

    def test_run_command_without_workspace_returns_error(self) -> None:
        """Verify run_command before workspace initialization returns error dict."""
        sandbox = MockSandbox()
        res = sandbox.run_command("echo test")
        assert res["exit_code"] == 1
        assert "No workspace" in str(res["stderr"])

    def test_reset_protected_files(self) -> None:
        """Verify reset_protected_files checks out baseline for test files."""
        mock_git = MagicMock(spec=GitUtils)
        mock_git.list_changed_files.return_value = ["tests/test_one.py", "src/foo.py"]
        sandbox = MockSandbox(git_utils=mock_git)
        sandbox._workspace_path = "/dummy/path"
        task = self._create_task()
        sandbox.reset_protected_files(task)
        mock_git.checkout_file.assert_any_call("/dummy/path", "tests/test_one.py")

    def test_is_protected_file_patterns(self) -> None:
        """Verify _is_protected_file detects test file patterns and config files."""
        sandbox = MockSandbox()
        assert sandbox._is_protected_file("test_something.py") is True
        assert sandbox._is_protected_file("some_test.py") is True
        assert sandbox._is_protected_file("tests/unit/test_app.py") is True
        assert sandbox._is_protected_file("conftest.py") is True
        assert sandbox._is_protected_file("pytest.ini") is True
        assert sandbox._is_protected_file("app/main.py") is False

    def test_setup_workspace_cleans_up_existing(self) -> None:
        """Verify setup_workspace cleans up previously initialized workspace."""
        sandbox = MockSandbox()
        task = self._create_task()
        try:
            ws1 = sandbox.setup_workspace(task)
            assert os.path.exists(ws1)
            ws2 = sandbox.setup_workspace(task)
            assert os.path.exists(ws2)
            assert not os.path.exists(ws1)
        finally:
            sandbox.cleanup()

    def test_apply_patch_and_reset_without_workspace(self) -> None:
        """Verify apply_patch and reset_protected_files when workspace is None."""
        sandbox = MockSandbox()
        task = self._create_task()
        assert sandbox.apply_patch("diff --git") is False
        assert sandbox.reset_protected_files(task) is None
        assert sandbox._apply_test_patch("diff --git") is False
        assert sandbox._apply_test_patch("") is False

    def test_run_command_timeout_and_oserror(self) -> None:
        """Verify run_command error handling for timeout and OS errors."""
        sandbox = MockSandbox()
        task = self._create_task()
        try:
            sandbox.setup_workspace(task)
            res_to = sandbox.run_command("python3 -c 'import time; time.sleep(2)'", timeout=1)
            assert res_to["exit_code"] == -1
            assert "timed out" in str(res_to["stderr"])
            res_os = sandbox.run_command("nonexistent_command_12345")
            assert res_os["exit_code"] != 0
        finally:
            sandbox.cleanup()

    def _create_task(self) -> Task:
        """Helper to create dummy Task instance."""
        return Task(
            instance_id="test_inst_001",
            repo="tiangolo/fastapi",
            base_commit=self._MOCK_COMMIT,
            problem_statement="Problem description",
            hints_text="Hints",
            patch="",
            test_patch=self._SAMPLE_TEST_PATCH,
            created_at=self._MOCK_CREATED,
        )
