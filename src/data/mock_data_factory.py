"""Mock data factory for generating test tasks, workspaces, and graphs."""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

from src.data.task import Task


class MockDataFactory:
    """Generates mock datasets, repositories, and embeddings for testing."""

    _MOCK_INSTANCE_ID: str = "mock_utils_001"
    _MOCK_REPO: str = "mock/utils-lib"
    _MOCK_COMMIT: str = "0000000000000000000000000000000000000000"
    _MOCK_CREATED_AT: str = "2026-01-01T00:00:00Z"
    _MOCK_PROBLEM: str = (
        "The add_numbers function has an off-by-one error. "
        "add_numbers(2, 3) returns 6 instead of 5."
    )
    _MOCK_HINTS: str = "Check the return statement."
    _MOCK_SOURCE: str = textwrap.dedent(
        '''\
        """Utility functions for mock workspace."""


        def add_numbers(a: int, b: int) -> int:
            """Add two numbers and return the result."""
            return a + b + 1  # Bug: off-by-one error
        '''
    )
    _MOCK_FIXED: str = textwrap.dedent(
        '''\
        """Utility functions for mock workspace."""


        def add_numbers(a: int, b: int) -> int:
            """Add two numbers and return the result."""
            return a + b
        '''
    )
    _UTILS_FILENAME: str = "utils.py"
    _INIT_FILENAME: str = "__init__.py"
    _TEST_FILENAME: str = "test_utils.py"
    _WORKSPACE_PREFIX: str = "mock_ws_"
    _GIT_CMD: str = "git"
    _GIT_INIT_ARG: str = "init"
    _GIT_ADD_ARG: str = "add"
    _GIT_ALL_ARG: str = "."
    _GIT_COMMIT_ARG: str = "commit"
    _GIT_MSG_FLAG: str = "-m"
    _GIT_COMMIT_MSG: str = "baseline"
    _GIT_ALLOW_EMPTY_FLAG: str = "--allow-empty-message"
    _ENV_GIT_AUTHOR_NAME: str = "GIT_AUTHOR_NAME"
    _ENV_GIT_AUTHOR_EMAIL: str = "GIT_AUTHOR_EMAIL"
    _ENV_GIT_COMMITTER_NAME: str = "GIT_COMMITTER_NAME"
    _ENV_GIT_COMMITTER_EMAIL: str = "GIT_COMMITTER_EMAIL"
    _GIT_USER: str = "MockUser"
    _GIT_EMAIL: str = "mock@example.com"
    _NODE_ID: str = "utils.add_numbers"
    _NODE_NAME: str = "add_numbers"
    _EMBEDDING_DIM: int = 256
    _ENCODING: str = "utf-8"
    _KEY_DIRECTED: str = "directed"
    _KEY_MULTIGRAPH: str = "multigraph"
    _KEY_GRAPH: str = "graph"
    _KEY_NODES: str = "nodes"
    _KEY_LINKS: str = "links"
    _KEY_ID: str = "id"
    _KEY_NAME: str = "name"
    _KEY_FILEPATH: str = "filepath"

    def __init__(self) -> None:
        """Initialize MockDataFactory."""

    def create_mock_task(self) -> Task:
        """Create a mock Task with an off-by-one bug fix scenario."""
        return Task(
            instance_id=self._MOCK_INSTANCE_ID,
            repo=self._MOCK_REPO,
            base_commit=self._MOCK_COMMIT,
            problem_statement=self._MOCK_PROBLEM,
            hints_text=self._MOCK_HINTS,
            patch=self._generate_patch(),
            test_patch=self._generate_test_patch(),
            created_at=self._MOCK_CREATED_AT,
        )

    def create_mock_workspace(self) -> str:
        """Create temporary mock directory with source files and git repository."""
        workspace_dir = tempfile.mkdtemp(prefix=self._WORKSPACE_PREFIX)
        workspace_path = Path(workspace_dir)
        (workspace_path / self._UTILS_FILENAME).write_text(
            self._MOCK_SOURCE, encoding=self._ENCODING
        )
        (workspace_path / self._INIT_FILENAME).write_text(
            "", encoding=self._ENCODING
        )
        self._init_git_repo(str(workspace_path))
        return str(workspace_path)

    def create_mock_graph(self) -> dict[str, object]:
        """Create minimal graph structure in node-link format."""
        return {
            self._KEY_DIRECTED: True,
            self._KEY_MULTIGRAPH: True,
            self._KEY_GRAPH: {},
            self._KEY_NODES: [
                {
                    self._KEY_ID: self._NODE_ID,
                    self._KEY_NAME: self._NODE_NAME,
                    self._KEY_FILEPATH: self._UTILS_FILENAME,
                }
            ],
            self._KEY_LINKS: [],
        }

    def create_mock_embeddings(self) -> dict[str, object]:
        """Create minimal node embedding vector mapping."""
        import numpy as np

        vector = np.zeros(self._EMBEDDING_DIM, dtype=np.float32)
        return {self._NODE_ID: vector}

    def _generate_patch(self) -> str:
        """Generate unified git diff for fixing add_numbers."""
        return textwrap.dedent(
            """\
            diff --git a/utils.py b/utils.py
            --- a/utils.py
            +++ b/utils.py
            @@ -4,3 +4,3 @@
             def add_numbers(a: int, b: int) -> int:
                 \"\"\"Add two numbers and return the result.\"\"\"
            -    return a + b + 1  # Bug: off-by-one error
            +    return a + b
            """
        )

    def _generate_test_patch(self) -> str:
        """Generate unified git diff for adding test_utils.py."""
        return textwrap.dedent(
            """\
            diff --git a/test_utils.py b/test_utils.py
            new file mode 100644
            --- /dev/null
            +++ b/test_utils.py
            @@ -0,0 +1,5 @@
            +from utils import add_numbers
            +
            +
            +def test_add_numbers() -> None:
            +    assert add_numbers(2, 3) == 5
            """
        )

    def _init_git_repo(self, workspace_path: str) -> None:
        """Initialize git repository in the mock workspace directory."""
        env = {
            **os.environ,
            self._ENV_GIT_AUTHOR_NAME: self._GIT_USER,
            self._ENV_GIT_AUTHOR_EMAIL: self._GIT_EMAIL,
            self._ENV_GIT_COMMITTER_NAME: self._GIT_USER,
            self._ENV_GIT_COMMITTER_EMAIL: self._GIT_EMAIL,
        }
        subprocess.run(
            [self._GIT_CMD, self._GIT_INIT_ARG],
            cwd=workspace_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [self._GIT_CMD, self._GIT_ADD_ARG, self._GIT_ALL_ARG],
            cwd=workspace_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            [
                self._GIT_CMD,
                self._GIT_COMMIT_ARG,
                self._GIT_MSG_FLAG,
                self._GIT_COMMIT_MSG,
                self._GIT_ALLOW_EMPTY_FLAG,
            ],
            cwd=workspace_path,
            capture_output=True,
            check=True,
            env=env,
        )
