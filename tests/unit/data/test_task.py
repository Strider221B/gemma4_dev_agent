"""Unit tests for Task data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.data.task import Task


class TestTask:
    """Test suite for Task frozen dataclass."""

    _INSTANCE_ID: str = "repo__task-1"
    _REPO: str = "owner/repo"
    _BASE_COMMIT: str = "a1b2c3d4e5f6"
    _PROBLEM_STATEMENT: str = "Fix the broken parser"
    _HINTS_TEXT: str = "Check ast parsing logic"
    _PATCH: str = "diff --git a/file.py b/file.py"
    _TEST_PATCH: str = "diff --git a/test_file.py b/test_file.py"
    _CREATED_AT: str = "2026-10-01T00:00:00Z"
    _MODIFIED_INSTANCE_ID: str = "repo__task-2"

    def _create_task(self) -> Task:
        """Helper to create a standard Task instance for testing."""
        return Task(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            base_commit=self._BASE_COMMIT,
            problem_statement=self._PROBLEM_STATEMENT,
            hints_text=self._HINTS_TEXT,
            patch=self._PATCH,
            test_patch=self._TEST_PATCH,
            created_at=self._CREATED_AT,
        )

    def test_task_creation_with_all_fields(self) -> None:
        """Verify Task fields are correctly set upon instantiation."""
        task = self._create_task()
        assert task.instance_id == self._INSTANCE_ID
        assert task.repo == self._REPO
        assert task.base_commit == self._BASE_COMMIT
        assert task.problem_statement == self._PROBLEM_STATEMENT
        assert task.hints_text == self._HINTS_TEXT
        assert task.patch == self._PATCH
        assert task.test_patch == self._TEST_PATCH
        assert task.created_at == self._CREATED_AT

    def test_task_is_frozen(self) -> None:
        """Verify modifying field on frozen Task raises FrozenInstanceError."""
        task = self._create_task()
        with pytest.raises(FrozenInstanceError):
            task.instance_id = self._MODIFIED_INSTANCE_ID

    def test_task_equality(self) -> None:
        """Verify two Task instances with identical attributes compare equal."""
        task1 = self._create_task()
        task2 = self._create_task()
        assert task1 == task2
