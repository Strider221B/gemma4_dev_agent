"""Unit tests for ExecutionBudget."""

from __future__ import annotations

import time

from src.evaluation.execution_budget import ExecutionBudget


class TestExecutionBudget:
    """Test suite for ExecutionBudget tracking and limit enforcement."""

    _TOOL_BASH: str = "bash"
    _TOOL_VIEW: str = "view_file"
    _TOOL_SUBMIT: str = "submit_patch"
    _TOOL_STATUS: str = "get_status"
    _MAX_CALLS_TEST: int = 5
    _KEY_OVERRIDE_MAX_TOOLS: str = "max_tool_calls"
    _KEY_OVERRIDE_MAX_TIME: str = "max_time_minutes"
    _STATUS_KEY_USED: str = "tool_calls_used"
    _STATUS_KEY_REMAINING: str = "tool_calls_remaining"
    _STATUS_KEY_MAX: str = "max_tool_calls"

    def test_consume_tool_call_increments_counter(self) -> None:
        """Verify non-free tool calls increment usage counter."""
        budget = ExecutionBudget()
        assert budget.consume_tool_call(self._TOOL_BASH) is True
        status = budget.get_status()
        assert status[self._STATUS_KEY_USED] == 1
        assert budget.consume_tool_call(self._TOOL_VIEW) is True
        status2 = budget.get_status()
        assert status2[self._STATUS_KEY_USED] == 2

    def test_consume_free_tool_does_not_increment(self) -> None:
        """Verify submit_patch and get_status do not consume budget."""
        budget = ExecutionBudget()
        assert budget.consume_tool_call(self._TOOL_SUBMIT) is True
        assert budget.consume_tool_call(self._TOOL_STATUS) is True
        status = budget.get_status()
        assert status[self._STATUS_KEY_USED] == 0

    def test_consume_over_budget_returns_false(self) -> None:
        """Verify tool calls beyond max_tool_calls return False."""
        budget = ExecutionBudget(overrides={self._KEY_OVERRIDE_MAX_TOOLS: self._MAX_CALLS_TEST})
        for _ in range(self._MAX_CALLS_TEST):
            assert budget.consume_tool_call(self._TOOL_BASH) is True
        assert budget.consume_tool_call(self._TOOL_BASH) is False
        status = budget.get_status()
        assert status[self._STATUS_KEY_USED] == self._MAX_CALLS_TEST + 1

    def test_should_warn_when_low_budget(self) -> None:
        """Verify should_warn is True when tool usage is at least 20 and remaining <= 10."""
        budget = ExecutionBudget(overrides={self._KEY_OVERRIDE_MAX_TOOLS: 25})
        for _ in range(19):
            budget.consume_tool_call(self._TOOL_BASH)
        assert budget.should_warn() is False
        budget.consume_tool_call(self._TOOL_BASH)
        assert budget.should_warn() is True

    def test_get_status_returns_dict(self) -> None:
        """Verify get_status returns expected keys and types."""
        budget = ExecutionBudget()
        status = budget.get_status()
        assert isinstance(status, dict)
        assert status["status"] == "ok"
        assert status[self._STATUS_KEY_USED] == 0
        assert status[self._STATUS_KEY_MAX] == 100
        assert status[self._STATUS_KEY_REMAINING] == 100
        assert "time_seconds_remaining" in status

    def test_check_time_returns_true_within_limit(self) -> None:
        """Verify check_time returns True when within configured runtime limit."""
        budget = ExecutionBudget(overrides={self._KEY_OVERRIDE_MAX_TIME: 10.0})
        assert budget.check_time() is True

    def test_check_time_returns_false_when_expired(self) -> None:
        """Verify check_time returns False when time limit is exceeded."""
        budget = ExecutionBudget(overrides={self._KEY_OVERRIDE_MAX_TIME: 0.0001})
        budget._start_time = time.time() - 100.0
        assert budget.check_time() is False
