"""Unit tests for Hunk data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.data.hunk import Hunk


class TestHunk:
    """Test suite for Hunk frozen dataclass."""

    _OLD_START: int = 10
    _OLD_COUNT: int = 5
    _NEW_START: int = 10
    _NEW_COUNT: int = 7
    _OLD_LINES: list[str] = ["-def old():", "-    pass"]
    _NEW_LINES: list[str] = ["+def new():", "+    return 1", "+    pass"]
    _MODIFIED_START: int = 99

    def _create_hunk(self) -> Hunk:
        """Helper to create a standard Hunk instance for testing."""
        return Hunk(
            old_start=self._OLD_START,
            old_count=self._OLD_COUNT,
            new_start=self._NEW_START,
            new_count=self._NEW_COUNT,
            old_lines=list(self._OLD_LINES),
            new_lines=list(self._NEW_LINES),
        )

    def test_hunk_creation(self) -> None:
        """Verify Hunk fields are correctly initialized."""
        hunk = self._create_hunk()
        assert hunk.old_start == self._OLD_START
        assert hunk.old_count == self._OLD_COUNT
        assert hunk.new_start == self._NEW_START
        assert hunk.new_count == self._NEW_COUNT
        assert hunk.old_lines == self._OLD_LINES
        assert hunk.new_lines == self._NEW_LINES

    def test_hunk_is_frozen(self) -> None:
        """Verify modifying field on frozen Hunk raises FrozenInstanceError."""
        hunk = self._create_hunk()
        with pytest.raises(FrozenInstanceError):
            hunk.old_start = self._MODIFIED_START
