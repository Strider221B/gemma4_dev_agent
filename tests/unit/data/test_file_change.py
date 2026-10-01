"""Unit tests for FileChange data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.data.file_change import FileChange
from src.data.hunk import Hunk


class TestFileChange:
    """Test suite for FileChange frozen dataclass."""

    _FILEPATH: str = "src/module/service.py"
    _MODIFIED_FILEPATH: str = "src/module/other.py"
    _OLD_START: int = 1
    _OLD_COUNT: int = 2
    _NEW_START: int = 1
    _NEW_COUNT: int = 3

    def _create_hunk(self) -> Hunk:
        """Helper to create a sample hunk."""
        return Hunk(
            old_start=self._OLD_START,
            old_count=self._OLD_COUNT,
            new_start=self._NEW_START,
            new_count=self._NEW_COUNT,
            old_lines=["-old"],
            new_lines=["+new", "+line"],
        )

    def test_file_change_with_hunks(self) -> None:
        """Verify FileChange attributes and nested hunks."""
        hunk = self._create_hunk()
        change = FileChange(filepath=self._FILEPATH, hunks=[hunk])
        assert change.filepath == self._FILEPATH
        assert len(change.hunks) == 1
        assert change.hunks[0] == hunk

    def test_hunk_fields(self) -> None:
        """Verify hunk fields nested within FileChange."""
        hunk = self._create_hunk()
        change = FileChange(filepath=self._FILEPATH, hunks=[hunk])
        hunk_ref = change.hunks[0]
        assert hunk_ref.old_start == self._OLD_START
        assert hunk_ref.new_count == self._NEW_COUNT

    def test_file_change_is_frozen(self) -> None:
        """Verify modifying field on FileChange raises FrozenInstanceError."""
        change = FileChange(filepath=self._FILEPATH, hunks=[self._create_hunk()])
        with pytest.raises(FrozenInstanceError):
            change.filepath = self._MODIFIED_FILEPATH
