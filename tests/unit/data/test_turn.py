"""Unit tests for Turn data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.data.tool_call import ToolCall
from src.data.turn import Turn


class TestTurn:
    """Test suite for Turn frozen dataclass."""

    _ROLE_MODEL: str = "model"
    _ROLE_USER: str = "user"
    _THOUGHT: str = "Analyzing error trace..."
    _TEXT: str = "Applying bug fix to parser."
    _TOOL_NAME: str = "read_file"
    _TOOL_RESULT: str = '{"status": "ok", "lines": 42}'
    _MODIFIED_ROLE: str = "assistant"

    def test_model_turn_with_tool_calls(self) -> None:
        """Verify model turn containing thought, text, and tool calls."""
        tool_call = ToolCall(tool_name=self._TOOL_NAME, args={})
        turn = Turn(
            role=self._ROLE_MODEL,
            thought=self._THOUGHT,
            text=self._TEXT,
            tool_calls=[tool_call],
        )
        assert turn.role == self._ROLE_MODEL
        assert turn.thought == self._THOUGHT
        assert turn.text == self._TEXT
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0] == tool_call
        assert turn.tool_result is None

    def test_user_turn_with_tool_result(self) -> None:
        """Verify user turn containing environment tool results."""
        turn = Turn(
            role=self._ROLE_USER,
            tool_result=self._TOOL_RESULT,
        )
        assert turn.role == self._ROLE_USER
        assert turn.tool_result == self._TOOL_RESULT
        assert turn.thought is None
        assert turn.text is None
        assert turn.tool_calls == []

    def test_turn_is_frozen(self) -> None:
        """Verify modifying field on frozen Turn raises FrozenInstanceError."""
        turn = Turn(role=self._ROLE_USER)
        with pytest.raises(FrozenInstanceError):
            turn.role = self._MODIFIED_ROLE
