"""Unit tests for MockDataFactory."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.data.mock_data_factory import MockDataFactory
from src.data.patch_parser import PatchParser


class TestMockDataFactory:
    """Test suite for MockDataFactory mock data generation."""

    _UTILS_FILENAME: str = "utils.py"
    _INIT_FILENAME: str = "__init__.py"
    _GIT_DIRNAME: str = ".git"
    _NODE_KEY: str = "nodes"
    _LINKS_KEY: str = "links"
    _NODE_ID: str = "utils.add_numbers"
    _EXPECTED_DIM: int = 256

    def test_create_mock_task_returns_task(self) -> None:
        """Verify mock task is created with all non-empty fields."""
        factory = MockDataFactory()
        task = factory.create_mock_task()
        assert task.instance_id != ""
        assert task.repo != ""
        assert task.base_commit != ""
        assert task.problem_statement != ""
        assert task.hints_text != ""
        assert task.patch != ""
        assert task.test_patch != ""
        assert task.created_at != ""

    def test_create_mock_workspace_creates_files(self) -> None:
        """Verify mock workspace creates utils.py and __init__.py files."""
        factory = MockDataFactory()
        workspace_str = factory.create_mock_workspace()
        workspace_path = Path(workspace_str)
        try:
            assert (workspace_path / self._UTILS_FILENAME).is_file()
            assert (workspace_path / self._INIT_FILENAME).is_file()
        finally:
            shutil.rmtree(workspace_path, ignore_errors=True)

    def test_create_mock_workspace_has_git_repo(self) -> None:
        """Verify mock workspace initializes a valid git repository with commit."""
        factory = MockDataFactory()
        workspace_str = factory.create_mock_workspace()
        workspace_path = Path(workspace_str)
        try:
            assert (workspace_path / self._GIT_DIRNAME).is_dir()
        finally:
            shutil.rmtree(workspace_path, ignore_errors=True)

    def test_create_mock_graph_returns_dict(self) -> None:
        """Verify mock graph structure contains nodes and links."""
        factory = MockDataFactory()
        graph = factory.create_mock_graph()
        assert isinstance(graph, dict)
        assert self._NODE_KEY in graph
        assert self._LINKS_KEY in graph
        nodes = graph[self._NODE_KEY]
        assert isinstance(nodes, list)
        assert len(nodes) > 0

    def test_create_mock_embeddings_returns_dict(self) -> None:
        """Verify mock embeddings returns dict with 256-dim embedding vector."""
        factory = MockDataFactory()
        embeddings = factory.create_mock_embeddings()
        assert isinstance(embeddings, dict)
        assert self._NODE_ID in embeddings
        vector = embeddings[self._NODE_ID]
        assert len(vector) == self._EXPECTED_DIM

    def test_mock_patch_is_valid_diff(self) -> None:
        """Verify generated mock patch is valid and parseable by PatchParser."""
        factory = MockDataFactory()
        task = factory.create_mock_task()
        parser = PatchParser()
        changes = parser.parse(task.patch)
        assert len(changes) == 1
        assert changes[0].filepath == self._UTILS_FILENAME
        assert len(changes[0].hunks) == 1
        assert len(changes[0].hunks[0].old_lines) > 0
        assert len(changes[0].hunks[0].new_lines) > 0

    def test_mock_test_patch_is_valid_diff(self) -> None:
        """Verify generated mock test patch is valid and parseable by PatchParser."""
        factory = MockDataFactory()
        task = factory.create_mock_task()
        parser = PatchParser()
        changes = parser.parse(task.test_patch)
        assert len(changes) == 1
        assert len(changes[0].hunks) == 1
        assert len(changes[0].hunks[0].new_lines) > 0
