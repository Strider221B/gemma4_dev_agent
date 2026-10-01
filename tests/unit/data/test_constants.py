"""Unit tests for data layer constants."""

from __future__ import annotations

import src.data.constants as data_constants


class TestDataConstants:
    """Test suite for data layer constants."""

    _EXPECTED_TRAJECTORY_TOKENS: int = 28672
    _EXPECTED_CONTEXT_LINES: int = 30

    def test_data_layer_constants(self) -> None:
        """Verify data layer tokens and line limit constants."""
        assert data_constants.MAX_TRAJECTORY_TOKENS == self._EXPECTED_TRAJECTORY_TOKENS
        assert data_constants.READ_CONTEXT_LINES == self._EXPECTED_CONTEXT_LINES
        assert data_constants.MAX_OLD_STRING_LINES == 15
        assert data_constants.MAX_TOOL_CALLS_PER_TRAJECTORY == 80
