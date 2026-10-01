"""Unit tests for ToolCall data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.data.tool_call import ToolCall


class TestToolCall:
    """Test suite for ToolCall frozen dataclass."""

    _TOOL_NAME: str = "read_file"
    _TOOL_ARG_KEY: str = "filepath"
    _TOOL_ARG_VAL: str = "src/main.py"
    _MODIFIED_TOOL_NAME: str = "edit_file"

    def _create_tool_call(self) -> ToolCall:
        """Helper to create a standard ToolCall instance."""
        return ToolCall(
            tool_name=self._TOOL_NAME,
            args={self._TOOL_ARG_KEY: self._TOOL_ARG_VAL},
        )

    def test_tool_call_creation(self) -> None:
        """Verify ToolCall attributes are correctly set."""
        call = self._create_tool_call()
        assert call.tool_name == self._TOOL_NAME
        assert call.args == {self._TOOL_ARG_KEY: self._TOOL_ARG_VAL}

    def test_tool_call_is_frozen(self) -> None:
        """Verify modifying field on frozen ToolCall raises FrozenInstanceError."""
        call = self._create_tool_call()
        with pytest.raises(FrozenInstanceError):
            call.tool_name = self._MODIFIED_TOOL_NAME
