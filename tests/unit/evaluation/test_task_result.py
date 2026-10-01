"""Unit tests for TaskResult data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.data.complexity_tier import ComplexityTier
from src.evaluation.task_result import TaskResult


class TestTaskResult:
    """Test suite for TaskResult frozen dataclass."""

    _INSTANCE_ID: str = "repo__task-1"
    _REPO: str = "owner/repo"
    _TIME_SECONDS: float = 45.2
    _ERROR_MSG: str = "Timeout during tool execution"
    _MODIFIED_TIME: float = 60.0

    def test_task_result_creation_and_defaults(self) -> None:
        """Verify default values on TaskResult optional attributes."""
        res = TaskResult(instance_id=self._INSTANCE_ID, repo=self._REPO)
        assert res.instance_id == self._INSTANCE_ID
        assert res.repo == self._REPO
        assert res.complexity is None
        assert res.resolved is False
        assert res.patch_size == 0
        assert res.tool_calls_used == 0
        assert res.tokens_used == 0
        assert res.had_truncation is False
        assert res.time_seconds == 0.0
        assert res.error is None

    def test_task_result_custom_values(self) -> None:
        """Verify TaskResult retains customized attributes."""
        res = TaskResult(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            resolved=True,
            patch_size=15,
            tool_calls_used=3,
            tokens_used=1200,
            had_truncation=True,
            time_seconds=self._TIME_SECONDS,
            error=self._ERROR_MSG,
        )
        assert res.complexity == ComplexityTier.SIMPLE
        assert res.resolved is True
        assert res.patch_size == 15
        assert res.tool_calls_used == 3
        assert res.tokens_used == 1200
        assert res.had_truncation is True
        assert res.time_seconds == self._TIME_SECONDS
        assert res.error == self._ERROR_MSG

    def test_task_result_is_frozen(self) -> None:
        """Verify modifying field on frozen TaskResult raises FrozenInstanceError."""
        res = TaskResult(instance_id=self._INSTANCE_ID, repo=self._REPO)
        with pytest.raises(FrozenInstanceError):
            res.time_seconds = self._MODIFIED_TIME
