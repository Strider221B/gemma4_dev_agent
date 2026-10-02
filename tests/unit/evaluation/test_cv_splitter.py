"""Unit tests for CVSplitter."""

from __future__ import annotations

import pytest

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.evaluation.cv_fold import CVFold
from src.evaluation.cv_splitter import CVSplitter


class TestCVSplitter:
    """Test suite for CVSplitter Group K-Fold cross-validation partitioning."""

    _MOCK_COMMIT: str = "0000000000000000000000000000000000000000"
    _MOCK_CREATED_AT: str = "2026-01-01T00:00:00Z"
    _REPO_FASTAPI: str = "tiangolo/fastapi"
    _REPO_RICH: str = "Textualize/rich"
    _REPO_REQUESTS: str = "psf/requests"
    _REPO_HTTPX: str = "encode/httpx"
    _SHORT_FASTAPI: str = "fastapi"
    _SHORT_RICH: str = "rich"
    _SHORT_REQUESTS: str = "requests"
    _SHORT_HTTPX: str = "httpx"

    def test_create_splits_returns_correct_fold_count(self) -> None:
        """Verify 4 distinct repositories produce exactly 4 folds."""
        splitter = CVSplitter(num_folds=4)
        tasks = self._create_four_repo_tasks()
        splits = splitter.create_splits(tasks)
        assert len(splits) == 4

    def test_create_splits_each_fold_holds_out_one_repo(self) -> None:
        """Verify each fold's val_repo is unique across folds."""
        splitter = CVSplitter(num_folds=4)
        tasks = self._create_four_repo_tasks()
        splits = splitter.create_splits(tasks)
        val_repos = [fold.val_repo for fold in splits]
        assert len(set(val_repos)) == 4
        expected_repos = {
            self._SHORT_FASTAPI,
            self._SHORT_RICH,
            self._SHORT_REQUESTS,
            self._SHORT_HTTPX,
        }
        assert set(val_repos) == expected_repos

    def test_create_splits_no_train_val_overlap(self) -> None:
        """Verify train_ids and val_ids are mutually disjoint for each fold."""
        splitter = CVSplitter(num_folds=4)
        tasks = self._create_four_repo_tasks()
        splits = splitter.create_splits(tasks)
        for fold in splits:
            overlap = set(fold.train_ids).intersection(set(fold.val_ids))
            assert len(overlap) == 0

    def test_create_splits_all_tasks_covered(self) -> None:
        """Verify the union of all val_ids covers all task instance IDs."""
        splitter = CVSplitter(num_folds=4)
        tasks = self._create_four_repo_tasks()
        splits = splitter.create_splits(tasks)
        all_val_ids: set[str] = set()
        for fold in splits:
            all_val_ids.update(fold.val_ids)
        expected_ids = {task.instance_id for task in tasks}
        assert all_val_ids == expected_ids

    def test_get_fold_by_index(self) -> None:
        """Verify get_fold retrieves the expected fold matching create_splits."""
        splitter = CVSplitter(num_folds=4)
        tasks = self._create_four_repo_tasks()
        fold_2 = splitter.get_fold(tasks, 2)
        assert isinstance(fold_2, CVFold)
        assert fold_2.fold_idx == 2

    def test_create_splits_complexity_distribution_populated(self) -> None:
        """Verify complexity distribution dictionary is populated for each fold."""
        splitter = CVSplitter(num_folds=4)
        tasks = self._create_four_repo_tasks()
        splits = splitter.create_splits(tasks)
        for fold in splits:
            dist = fold.complexity_distribution
            assert ComplexityTier.SIMPLE.value in dist
            assert ComplexityTier.MODERATE.value in dist
            assert ComplexityTier.COMPLEX.value in dist
            assert sum(dist.values()) == len(fold.val_ids)

    def test_create_splits_empty_tasks_returns_empty(self) -> None:
        """Verify empty tasks input yields an empty list of folds."""
        splitter = CVSplitter()
        assert splitter.create_splits([]) == []

    def test_create_splits_insufficient_groups_raises_value_error(self) -> None:
        """Verify tasks with only one group raises ValueError."""
        splitter = CVSplitter()
        single_repo_tasks = [
            self._build_task("task_001", self._REPO_FASTAPI, 5, 1),
            self._build_task("task_002", self._REPO_FASTAPI, 8, 1),
        ]
        with pytest.raises(ValueError, match="At least 2 distinct groups"):
            splitter.create_splits(single_repo_tasks)

    def test_get_fold_out_of_bounds_raises_index_error(self) -> None:
        """Verify invalid fold index raises IndexError."""
        splitter = CVSplitter(num_folds=4)
        tasks = self._create_four_repo_tasks()
        with pytest.raises(IndexError, match="Fold index out of bounds"):
            splitter.get_fold(tasks, 99)
        with pytest.raises(IndexError, match="Fold index out of bounds"):
            splitter.get_fold(tasks, -1)

    def test_extract_repo_name_handles_various_formats(self) -> None:
        """Verify _extract_repo_name strips owner prefix and whitespace."""
        splitter = CVSplitter()
        assert splitter._extract_repo_name("tiangolo/fastapi") == "fastapi"
        assert splitter._extract_repo_name("fastapi") == "fastapi"
        assert splitter._extract_repo_name("  owner/repo  ") == "repo"

    def test_classify_complexity_tiers(self) -> None:
        """Verify classification into SIMPLE, MODERATE, and COMPLEX tiers."""
        splitter = CVSplitter()
        simple_task = self._build_task("t1", self._REPO_FASTAPI, 5, 1)
        mod_task = self._build_task("t2", self._REPO_FASTAPI, 50, 2)
        complex_task = self._build_task("t3", self._REPO_FASTAPI, 150, 5)
        assert splitter._classify_complexity(simple_task) == ComplexityTier.SIMPLE
        assert splitter._classify_complexity(mod_task) == ComplexityTier.MODERATE
        assert splitter._classify_complexity(complex_task) == ComplexityTier.COMPLEX

    def _create_four_repo_tasks(self) -> list[Task]:
        """Helper to create 8 tasks distributed across 4 repositories."""
        repos = [
            self._REPO_FASTAPI,
            self._REPO_RICH,
            self._REPO_REQUESTS,
            self._REPO_HTTPX,
        ]
        tasks: list[Task] = []
        for repo_idx, repo in enumerate(repos):
            tasks.append(
                self._build_task(f"task_{repo_idx}_a", repo, 10, 1)
            )
            tasks.append(
                self._build_task(f"task_{repo_idx}_b", repo, 60, 2)
            )
        return tasks

    def _build_task(
        self, instance_id: str, repo: str, lines: int, files: int
    ) -> Task:
        """Construct a Task instance with synthesized patch metrics."""
        patch_lines: list[str] = []
        for file_idx in range(files):
            patch_lines.append(f"diff --git a/mod{file_idx}.py b/mod{file_idx}.py")
            patch_lines.append(f"--- a/mod{file_idx}.py")
            patch_lines.append(f"+++ b/mod{file_idx}.py")
            lines_per_file = max(1, lines // files)
            for _ in range(lines_per_file):
                patch_lines.append("+new_line")
        return Task(
            instance_id=instance_id,
            repo=repo,
            base_commit=self._MOCK_COMMIT,
            problem_statement="Mock problem statement",
            hints_text="Mock hints",
            patch="\n".join(patch_lines),
            test_patch="",
            created_at=self._MOCK_CREATED_AT,
        )
