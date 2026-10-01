"""Unit tests for PatchParser."""

from __future__ import annotations

import textwrap

from src.data.patch_parser import PatchParser


class TestPatchParser:
    """Test suite for PatchParser unified git diff parsing."""

    _DIFF_SINGLE_HUNK: str = textwrap.dedent(
        """\
        diff --git a/module.py b/module.py
        --- a/module.py
        +++ b/module.py
        @@ -10,3 +10,3 @@ def foo():
         def foo():
        -    return 1
        +    return 2
        """
    )
    _DIFF_MULTI_HUNK: str = textwrap.dedent(
        """\
        diff --git a/module.py b/module.py
        --- a/module.py
        +++ b/module.py
        @@ -1,3 +1,3 @@
         def a():
        -    return 1
        +    return 2
        @@ -10,3 +10,3 @@
         def b():
        -    return 10
        +    return 20
        """
    )
    _DIFF_MULTI_FILE: str = textwrap.dedent(
        """\
        diff --git a/file1.py b/file1.py
        --- a/file1.py
        +++ b/file1.py
        @@ -1,2 +1,2 @@
         -a = 1
         +a = 2
        diff --git a/file2.py b/file2.py
        --- a/file2.py
        +++ b/file2.py
        @@ -1,2 +1,2 @@
         -b = 1
         +b = 2
        diff --git a/file3.py b/file3.py
        --- a/file3.py
        +++ b/file3.py
        @@ -1,2 +1,2 @@
         -c = 1
         +c = 2
        """
    )
    _DIFF_NEW_FILE: str = textwrap.dedent(
        """\
        diff --git a/dev/null b/new_module.py
        --- /dev/null
        +++ b/new_module.py
        @@ -0,0 +1,2 @@
        +new_line_1
        +new_line_2
        """
    )
    _DIFF_DELETED_FILE: str = textwrap.dedent(
        """\
        diff --git a/old_module.py b/dev/null
        --- a/old_module.py
        +++ /dev/null
        @@ -1,2 +0,0 @@
        -deleted_1
        -deleted_2
        """
    )
    _DIFF_BINARY: str = textwrap.dedent(
        """\
        diff --git a/image.png b/image.png
        Binary files a/image.png and b/image.png differ
        """
    )
    _DIFF_NO_NEWLINE: str = textwrap.dedent(
        """\
        diff --git a/single.py b/single.py
        --- a/single.py
        +++ b/single.py
        @@ -1,1 +1,1 @@
        -old
        \\ No newline at end of file
        +new
        \\ No newline at end of file
        """
    )
    _TARGET_FILE_MODULE: str = "module.py"
    _TARGET_FILE_NEW: str = "new_module.py"
    _TARGET_FILE_OLD: str = "old_module.py"
    _EMPTY_PATCH: str = ""
    _WHITESPACE_PATCH: str = "   \n  \t  "
    _INVALID_HEADER_LINE: str = "@@ invalid line @@"
    _INVALID_DIFF_LINE: str = "not a diff git line"

    def test_parse_single_file_single_hunk(self) -> None:
        """Verify parsing a diff with a single file and single hunk."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_SINGLE_HUNK)
        assert len(changes) == 1
        assert changes[0].filepath == self._TARGET_FILE_MODULE
        assert len(changes[0].hunks) == 1
        assert changes[0].hunks[0].old_start == 10
        assert changes[0].hunks[0].new_start == 10

    def test_parse_single_file_multi_hunk(self) -> None:
        """Verify parsing a diff with multiple hunks in one file."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_MULTI_HUNK)
        assert len(changes) == 1
        assert len(changes[0].hunks) == 2
        assert changes[0].hunks[0].old_start == 1
        assert changes[0].hunks[1].old_start == 10

    def test_parse_multi_file(self) -> None:
        """Verify parsing a diff spanning multiple distinct files."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_MULTI_FILE)
        assert len(changes) == 3
        paths = [c.filepath for c in changes]
        assert paths == ["file1.py", "file2.py", "file3.py"]

    def test_parse_new_file(self) -> None:
        """Verify parsing a newly created file diff."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_NEW_FILE)
        assert len(changes) == 1
        assert changes[0].filepath == self._TARGET_FILE_NEW
        assert len(changes[0].hunks[0].new_lines) == 2

    def test_parse_deleted_file(self) -> None:
        """Verify parsing a deleted file diff."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_DELETED_FILE)
        assert len(changes) == 1
        assert changes[0].filepath == self._TARGET_FILE_OLD
        assert len(changes[0].hunks[0].old_lines) == 2

    def test_parse_empty_patch_returns_empty(self) -> None:
        """Verify parsing an empty or whitespace diff returns empty list."""
        parser = PatchParser()
        assert parser.parse(self._EMPTY_PATCH) == []
        assert parser.parse(self._WHITESPACE_PATCH) == []

    def test_parse_hunk_header_extracts_line_numbers(self) -> None:
        """Verify hunk header correctly extracts old and new line numbers and counts."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_SINGLE_HUNK)
        hunk = changes[0].hunks[0]
        assert hunk.old_start == 10
        assert hunk.old_count == 3
        assert hunk.new_start == 10
        assert hunk.new_count == 3

    def test_parse_preserves_old_and_new_lines(self) -> None:
        """Verify hunk properly accumulates deleted lines in old and added lines in new."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_SINGLE_HUNK)
        hunk = changes[0].hunks[0]
        assert "    return 1" in hunk.old_lines
        assert "    return 2" in hunk.new_lines

    def test_parse_binary_file_skipped(self) -> None:
        """Verify binary diff markers cause binary files to be ignored."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_BINARY)
        assert len(changes) == 0

    def test_parse_handles_no_newline_marker(self) -> None:
        """Verify '\\ No newline at end of file' lines are ignored."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_NO_NEWLINE)
        assert len(changes) == 1
        hunk = changes[0].hunks[0]
        assert "old" in hunk.old_lines
        assert "new" in hunk.new_lines

    def test_count_patch_lines(self) -> None:
        """Verify helper method counts total lines across hunks."""
        parser = PatchParser()
        changes = parser.parse(self._DIFF_MULTI_FILE)
        total_lines = parser._count_patch_lines(changes)
        assert total_lines > 0

    def test_extract_filepath_invalid(self) -> None:
        """Verify invalid line header returns empty string."""
        parser = PatchParser()
        assert parser._extract_filepath(self._INVALID_DIFF_LINE) == ""

    def test_parse_hunk_header_invalid(self) -> None:
        """Verify invalid hunk line returns a default empty Hunk."""
        parser = PatchParser()
        hunk = parser._parse_hunk_header(self._INVALID_HEADER_LINE)
        assert hunk.old_start == 0
        assert hunk.new_start == 0
