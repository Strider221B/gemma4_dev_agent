"""Mock sandbox environment for evaluating task patches and running tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.data.task import Task
from src.utils.git_utils import GitUtils


class MockSandbox:
    """Isolated execution workspace simulating benchmark execution environment."""

    _DEFAULT_TIMEOUT: int = 300
    _TEST_PREFIX: str = "test_"
    _TEST_SUFFIX: str = "_test.py"
    _TESTS_DIR: str = "tests"
    _PROTECTED_FILENAMES: frozenset[str] = frozenset({
        "conftest.py",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
    })
    _GIT_USER_NAME: str = "Evaluator"
    _GIT_USER_EMAIL: str = "evaluator@agent.internal"
    _BASELINE_COMMIT_MSG: str = "baseline"
    _DEFAULT_MOCK_FILE: str = "utils.py"
    _DEFAULT_MOCK_CONTENT: str = "def add_numbers(a, b):\n    return a + b\n"
    _TEMP_PREFIX: str = "mock_workspace_"
    _KEY_EXIT_CODE: str = "exit_code"
    _KEY_STDOUT: str = "stdout"
    _KEY_STDERR: str = "stderr"
    _DIFF_OLD_HEADER: str = "--- "
    _DEV_NULL: str = "/dev/null"
    _SLASH: str = "/"

    def __init__(self, git_utils: GitUtils | None = None) -> None:
        """Initialize MockSandbox with git utility dependency."""
        self._git_utils: GitUtils = git_utils or GitUtils()
        self._workspace_path: str | None = None

    def setup_workspace(self, task: Task) -> str:
        """Create temp workspace directory and baseline repository for task."""
        if self._workspace_path is not None:
            self.cleanup()
        workspace_dir = self._create_workspace_from_snapshot(task)
        self._workspace_path = workspace_dir
        return workspace_dir

    def get_workspace_path(self) -> str | None:
        """Return active workspace filesystem path or None."""
        return self._workspace_path

    def apply_patch(self, patch: str) -> bool:
        """Apply unified diff patch to the active workspace."""
        if not self._workspace_path or not patch.strip():
            return self._workspace_path is not None
        return self._git_utils.apply_patch(self._workspace_path, patch)

    def reset_protected_files(self, task: Task) -> None:
        """Reset test files and protected configurations to baseline commit."""
        if not self._workspace_path:
            return
        protected_files = self._collect_protected_files(task)
        for filepath in protected_files:
            self._git_utils.checkout_file(self._workspace_path, filepath)

    def run_command(self, command: str, timeout: int = _DEFAULT_TIMEOUT) -> dict[str, object]:
        """Execute shell command in workspace and capture outputs."""
        if not self._workspace_path:
            return {self._KEY_EXIT_CODE: 1, self._KEY_STDOUT: "", self._KEY_STDERR: "No workspace"}
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self._workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                self._KEY_EXIT_CODE: res.returncode,
                self._KEY_STDOUT: res.stdout,
                self._KEY_STDERR: res.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                self._KEY_EXIT_CODE: -1,
                self._KEY_STDOUT: "",
                self._KEY_STDERR: f"Command timed out after {timeout} seconds",
            }
        except OSError as err:
            return {self._KEY_EXIT_CODE: 1, self._KEY_STDOUT: "", self._KEY_STDERR: str(err)}

    def cleanup(self) -> None:
        """Remove temporary workspace directory and reset state."""
        if self._workspace_path and os.path.exists(self._workspace_path):
            shutil.rmtree(self._workspace_path, ignore_errors=True)
        self._workspace_path = None

    def _is_protected_file(self, filepath: str) -> bool:
        """Check if filepath matches protected test or configuration file patterns."""
        normalized = filepath.replace("\\", self._SLASH).strip()
        path = Path(normalized)
        name = path.name
        if name.startswith(self._TEST_PREFIX) or name.endswith(self._TEST_SUFFIX):
            return True
        if name in self._PROTECTED_FILENAMES:
            return True
        return self._TESTS_DIR in path.parts

    def _create_workspace_from_snapshot(self, task: Task) -> str:
        """Create temporary workspace directory and initialize baseline git repo."""
        temp_dir = tempfile.mkdtemp(prefix=self._TEMP_PREFIX)
        self._init_git_repo(temp_dir)
        self._write_initial_files(temp_dir, task)
        self._commit_baseline(temp_dir)
        return temp_dir

    def _apply_test_patch(self, test_patch: str) -> bool:
        """Apply verification test patch to workspace."""
        if not self._workspace_path or not test_patch.strip():
            return self._workspace_path is not None
        return self._git_utils.apply_patch(self._workspace_path, test_patch)

    def _init_git_repo(self, path: str) -> None:
        """Initialize empty git repository and configure user identity."""
        subprocess.run(["git", "init"], cwd=path, capture_output=True, check=False)
        subprocess.run(
            ["git", "config", "user.name", self._GIT_USER_NAME],
            cwd=path,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "config", "user.email", self._GIT_USER_EMAIL],
            cwd=path,
            capture_output=True,
            check=False,
        )

    def _write_initial_files(self, path: str, task: Task) -> None:
        """Write baseline files for testing."""
        default_file = Path(path) / self._DEFAULT_MOCK_FILE
        default_file.write_text(self._DEFAULT_MOCK_CONTENT, encoding="utf-8")

    def _commit_baseline(self, path: str) -> None:
        """Add all files and record baseline git commit."""
        subprocess.run(["git", "add", "-A"], cwd=path, capture_output=True, check=False)
        subprocess.run(
            ["git", "commit", "-m", self._BASELINE_COMMIT_MSG, "--allow-empty"],
            cwd=path,
            capture_output=True,
            check=False,
        )

    def _collect_protected_files(self, task: Task) -> set[str]:
        """Collect all protected filepaths from changed files and test patch."""
        assert self._workspace_path is not None
        changed = self._git_utils.list_changed_files(self._workspace_path)
        protected = {f for f in changed if self._is_protected_file(f)}
        if task.test_patch.strip():
            parsed_test_files = self._extract_files_from_patch(task.test_patch)
            protected.update(parsed_test_files)
        return protected

    def _extract_files_from_patch(self, patch: str) -> list[str]:
        """Extract modified filepaths from unified diff header lines."""
        files: list[str] = []
        for line in patch.splitlines():
            if line.startswith(self._DIFF_OLD_HEADER):
                raw = line[len(self._DIFF_OLD_HEADER):].strip()
                if raw and raw != self._DEV_NULL:
                    clean_path = raw[2:] if raw.startswith("a/") or raw.startswith("b/") else raw
                    files.append(clean_path)
        return files
