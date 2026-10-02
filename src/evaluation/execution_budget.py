"""Execution budget simulator for agent tool calls and runtime."""

from __future__ import annotations

import time


class ExecutionBudget:
    """Enforces tool call, time, and turn budget limits for agent execution."""

    _DEFAULT_MAX_TOOL_CALLS: int = 100
    _DEFAULT_MAX_TIME_MINUTES: float = 60.0
    _DEFAULT_MAX_TURNS: int = 500
    _DEFAULT_COMMAND_TIMEOUT: int = 300
    _MAX_STDOUT_CHARS: int = 5000
    _MAX_FILE_LINES: int = 150
    _MAX_FILE_CHARS: int = 10000
    _MAX_NUDGES: int = 3
    _FREE_TOOLS: frozenset[str] = frozenset({"submit_patch", "get_status"})
    _LOW_BUDGET_THRESHOLD: int = 10
    _LOW_BUDGET_MIN_CALLS: int = 20
    _SECONDS_PER_MINUTE: float = 60.0

    _KEY_MAX_TOOL_CALLS: str = "max_tool_calls"
    _KEY_MAX_TIME_MINUTES: str = "max_time_minutes"
    _STATUS_OK: str = "ok"
    _STATUS_KEY_STATUS: str = "status"
    _STATUS_KEY_TOOL_CALLS_USED: str = "tool_calls_used"
    _STATUS_KEY_TOOL_CALLS_REMAINING: str = "tool_calls_remaining"
    _STATUS_KEY_MAX_TOOL_CALLS: str = "max_tool_calls"
    _STATUS_KEY_TIME_REMAINING: str = "time_seconds_remaining"
    _STATUS_KEY_MAX_TIME_MINUTES: str = "max_time_minutes"
    _STATUS_KEY_ELAPSED: str = "agent_elapsed_seconds"

    def __init__(self, overrides: dict[str, object] | None = None) -> None:
        """Initialize budget with optional parameter overrides."""
        cfg = overrides or {}
        raw_max_tools = cfg.get(self._KEY_MAX_TOOL_CALLS, self._DEFAULT_MAX_TOOL_CALLS)
        self._max_tool_calls: int = (
            int(raw_max_tools)
            if isinstance(raw_max_tools, (int, float))
            else self._DEFAULT_MAX_TOOL_CALLS
        )

        raw_max_time = cfg.get(self._KEY_MAX_TIME_MINUTES, self._DEFAULT_MAX_TIME_MINUTES)
        max_time_mins: float = (
            float(raw_max_time)
            if isinstance(raw_max_time, (int, float))
            else self._DEFAULT_MAX_TIME_MINUTES
        )
        self._max_time_seconds: float = max_time_mins * self._SECONDS_PER_MINUTE

        self._tool_calls_used: int = 0
        self._start_time: float = time.time()
        self._consecutive_nudges: int = 0

    def consume_tool_call(self, tool_name: str) -> bool:
        """Register a tool call; return False if budget exceeded."""
        if tool_name in self._FREE_TOOLS:
            return True
        self._tool_calls_used += 1
        return self._tool_calls_used <= self._max_tool_calls

    def check_time(self) -> bool:
        """Return True if time budget is still available."""
        elapsed = time.time() - self._start_time
        return elapsed < self._max_time_seconds

    def should_warn(self) -> bool:
        """Return True if budget warning should be triggered."""
        remaining = self._max_tool_calls - self._tool_calls_used
        return (
            self._tool_calls_used >= self._LOW_BUDGET_MIN_CALLS
            and remaining <= self._LOW_BUDGET_THRESHOLD
        )

    def get_status(self) -> dict[str, object]:
        """Return harness-compatible budget status dictionary."""
        elapsed = time.time() - self._start_time
        remaining_time = max(0.0, self._max_time_seconds - elapsed)
        remaining_calls = self._max_tool_calls - self._tool_calls_used
        return {
            self._STATUS_KEY_STATUS: self._STATUS_OK,
            self._STATUS_KEY_TOOL_CALLS_USED: self._tool_calls_used,
            self._STATUS_KEY_TOOL_CALLS_REMAINING: remaining_calls,
            self._STATUS_KEY_MAX_TOOL_CALLS: self._max_tool_calls,
            self._STATUS_KEY_TIME_REMAINING: remaining_time,
            self._STATUS_KEY_MAX_TIME_MINUTES: self._max_time_seconds / self._SECONDS_PER_MINUTE,
            self._STATUS_KEY_ELAPSED: elapsed,
        }
