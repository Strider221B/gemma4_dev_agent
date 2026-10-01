"""Unit tests for TrajectorySynthesiser."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from src.config.data_paths_config import DataPathsConfig
from src.data.complexity_tier import ComplexityTier
from src.data.hunk import Hunk
from src.data.ingestor import DataIngestor
from src.data.mock_data_factory import MockDataFactory
from src.data.patch_parser import PatchParser
from src.data.task import Task
from src.data.trajectory_synthesiser import TrajectorySynthesiser
from src.utils.token_counter import TokenCounter


class TestTrajectorySynthesiser:
    """Test suite for TrajectorySynthesiser class."""

    _MOCK_TASK_ID: str = "mock_utils_001"
    _TOOL_SEARCH: str = "search_similar_code"
    _TOOL_NEIGHBORS: str = "get_code_neighbors"
    _TOOL_EDIT: str = "edit_file"
    _TOOL_SUBMIT: str = "submit_patch"
    _ROLE_USER: str = "user"
    _ROLE_MODEL: str = "model"

    def test_synthesise_mock_task_returns_trajectory(self) -> None:
        """Verify synthesise returns a structured Trajectory with non-zero fields."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        assert trajectory.instance_id == task.instance_id
        assert trajectory.repo == task.repo
        assert trajectory.complexity == ComplexityTier.SIMPLE
        assert len(trajectory.turns) > 0
        assert trajectory.token_count > 0
        assert trajectory.num_tool_calls > 0
        assert trajectory.num_files_changed == 1

    def test_synthesise_produces_correct_turn_order(self) -> None:
        """Verify turn roles alternate between user and model properly."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        assert trajectory.turns[0].role == self._ROLE_USER
        assert trajectory.turns[1].role == self._ROLE_MODEL
        for i in range(1, len(trajectory.turns)):
            curr_role = trajectory.turns[i].role
            prev_role = trajectory.turns[i - 1].role
            if curr_role == prev_role:
                assert curr_role == self._ROLE_MODEL

    def test_synthesise_initial_prompt_contains_problem_statement(self) -> None:
        """Verify initial turn incorporates the task problem statement and budget."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        initial_turn = trajectory.turns[0]
        assert initial_turn.role == self._ROLE_USER
        assert task.problem_statement in str(initial_turn.text)
        assert "Time allowance: 60.0 minutes" in str(initial_turn.text)

    def test_synthesise_includes_navigation_tools(self) -> None:
        """Verify navigation tool calls are generated."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        tool_names = [
            tc.tool_name
            for turn in trajectory.turns
            for tc in turn.tool_calls
        ]
        assert self._TOOL_SEARCH in tool_names
        assert self._TOOL_NEIGHBORS in tool_names

    def test_synthesise_includes_edit_file_calls(self) -> None:
        """Verify edit_file tool calls are generated for patch hunks."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        tool_names = [
            tc.tool_name
            for turn in trajectory.turns
            for tc in turn.tool_calls
        ]
        assert self._TOOL_EDIT in tool_names

    def test_synthesise_ends_with_submit_patch(self) -> None:
        """Verify final tool invocation is submit_patch."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        all_calls = [
            tc
            for turn in trajectory.turns
            for tc in turn.tool_calls
        ]
        assert len(all_calls) > 0
        assert all_calls[-1].tool_name == self._TOOL_SUBMIT

    def test_synthesise_all_filters_over_budget(self) -> None:
        """Verify batch synthesis excludes trajectories that exceed token budget."""
        token_counter = MagicMock(spec=TokenCounter)
        token_counter.count_tokens.side_effect = (
            lambda text: 30000 if "OVER_BUDGET" in text else 10
        )
        paths = DataPathsConfig(tasks_path="", graphs_dir="", embeddings_dir="")
        ingestor = DataIngestor(paths)
        synthesiser = TrajectorySynthesiser(
            ingestor=ingestor,
            patch_parser=PatchParser(),
            token_counter=token_counter,
            mock_mode=True,
        )
        task1 = MockDataFactory().create_mock_task()
        task2 = Task(
            instance_id="task_over_002",
            repo="mock/repo",
            base_commit="0" * 40,
            problem_statement="OVER_BUDGET problem statement",
            hints_text="hints",
            patch=task1.patch,
            test_patch=task1.test_patch,
            created_at="2026-01-01T00:00:00Z",
        )
        results = synthesiser.synthesise_all([task1, task2])
        assert len(results) == 1
        assert results[0].instance_id == task1.instance_id

    def test_synthesise_mock_mode_works_without_graph_files(self) -> None:
        """Verify synthesiser functions without existing filesystem graph files."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        assert trajectory.num_tool_calls >= 4

    def test_synthesise_non_mock_mode_loads_graph(self) -> None:
        """Verify non-mock mode delegates graph loading to DataIngestor."""
        paths = DataPathsConfig(
            tasks_path="",
            graphs_dir=tempfile.gettempdir(),
            embeddings_dir="",
        )
        ingestor = DataIngestor(paths)
        synthesiser = TrajectorySynthesiser(
            ingestor=ingestor,
            patch_parser=PatchParser(),
            token_counter=TokenCounter(),
            mock_mode=False,
        )
        task = MockDataFactory().create_mock_task()
        trajectory = synthesiser.synthesise(task)
        assert trajectory.instance_id == task.instance_id

    def test_split_hunk_long_lines(self) -> None:
        """Verify large hunk is truncated to max old string lines."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        old_lines = [f"old_{i}" for i in range(25)]
        new_lines = [f"new_{i}" for i in range(25)]
        hunk = Hunk(1, 25, 1, 25, old_lines=old_lines, new_lines=new_lines)
        old_res, new_res = synthesiser._split_hunk(hunk)
        assert len(old_res.splitlines()) == 15
        assert len(new_res.splitlines()) == 15

    def test_extract_symbols_from_graph_structure(self) -> None:
        """Verify symbols extraction when hunk has no symbols but graph does."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        graph = {"nodes": [{"name": "graph_func"}]}
        symbols = synthesiser._extract_symbols([], graph)
        assert "graph_func" in symbols

    def test_extract_keywords_stopwords_only(self) -> None:
        """Verify keyword extraction falls back when only stopwords present."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        kw = synthesiser._extract_keywords("the and that this")
        assert kw == "issue"

    def test_synthesise_large_hunk_splits_lines(self) -> None:
        """Verify synthesising task with >15 hunk lines triggers hunk splitting."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        hunk_old = "\n".join(f"- line {i}" for i in range(20))
        hunk_new = "\n".join(f"+ line {i}" for i in range(20))
        diff_text = (
            "diff --git a/big.py b/big.py\n"
            "--- a/big.py\n"
            "+++ b/big.py\n"
            f"@@ -1,20 +1,20 @@\n{hunk_old}\n{hunk_new}"
        )
        task = Task(
            instance_id="big_001",
            repo="mock/repo",
            base_commit="0" * 40,
            problem_statement="Fix big hunk issue",
            hints_text="",
            patch=diff_text,
            test_patch="",
            created_at="2026-01-01T00:00:00Z",
        )
        trajectory = synthesiser.synthesise(task)
        assert trajectory.num_tool_calls > 0

    def test_extract_symbols_duplicate_definition(self) -> None:
        """Verify symbol extraction ignores duplicates."""
        from src.data.file_change import FileChange

        synthesiser = self._create_synthesiser(mock_mode=True)
        hunk = Hunk(
            1, 2, 1, 2,
            old_lines=["def foo(): pass"],
            new_lines=["def foo(): pass"],
        )
        symbols = synthesiser._extract_symbols([FileChange("a.py", [hunk])], None)
        assert symbols.count("foo") == 1

    def test_extract_symbols_from_graph_invalid_nodes(self) -> None:
        """Verify graph symbol extraction handles missing or malformed nodes."""
        synthesiser = self._create_synthesiser(mock_mode=True)
        assert synthesiser._extract_symbols([], {}) == ["add_numbers"]
        assert synthesiser._extract_symbols([], {"nodes": ["invalid_str"]}) == ["add_numbers"]

    def _create_synthesiser(self, mock_mode: bool) -> TrajectorySynthesiser:
        """Helper to instantiate TrajectorySynthesiser with mock dependencies."""
        paths = DataPathsConfig(tasks_path="", graphs_dir="", embeddings_dir="")
        ingestor = DataIngestor(paths)
        return TrajectorySynthesiser(
            ingestor=ingestor,
            patch_parser=PatchParser(),
            token_counter=TokenCounter(),
            mock_mode=mock_mode,
        )
