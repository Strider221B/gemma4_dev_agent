"""Unified git diff parser producing structured FileChange objects."""

from __future__ import annotations

import re

from src.data.file_change import FileChange
from src.data.hunk import Hunk


class PatchParser:
    """Parses unified git diff into structured FileChange objects."""

    _DIFF_HEADER: str = "diff --git"
    _HUNK_HEADER_PATTERN: str = r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
    _FILE_HEADER_PATTERN: str = r"^diff --git a/(.*?) b/(.*)$"
    _OLD_FILE_PREFIX: str = "--- "
    _NEW_FILE_PREFIX: str = "+++ "
    _DEV_NULL: str = "/dev/null"
    _DEV_NULL_REL: str = "dev/null"
    _NO_NEWLINE_MARKER: str = r"\ No newline at end of file"
    _BINARY_DIFF_MARKER: str = "Binary files "
    _GIT_BINARY_MARKER: str = "GIT binary patch"
    _ADD_PREFIX: str = "+"
    _REMOVE_PREFIX: str = "-"
    _CONTEXT_PREFIX: str = " "
    _EMPTY_STR: str = ""
    _DEFAULT_COUNT: int = 1

    def __init__(self) -> None:
        """Initialize PatchParser."""

    def parse(self, patch_text: str) -> list[FileChange]:
        """Parse unified git diff into structured FileChange objects."""
        if not patch_text or not patch_text.strip():
            return []
        files: list[FileChange] = []
        current_file: str | None = None
        current_hunks: list[Hunk] = []
        is_binary: bool = False
        for line in patch_text.splitlines():
            if line.startswith(self._DIFF_HEADER):
                self._flush_file(files, current_file, current_hunks, is_binary)
                current_file = self._extract_filepath(line)
                current_hunks = []
                is_binary = False
            elif self._is_binary_line(line):
                is_binary = True
            elif self._is_hunk_header(line) and not is_binary:
                current_hunks.append(self._parse_hunk_header(line))
            elif current_hunks and not is_binary:
                self._accumulate_hunk_lines(current_hunks[-1], line)
        self._flush_file(files, current_file, current_hunks, is_binary)
        return files

    def _flush_file(
        self,
        files: list[FileChange],
        current_file: str | None,
        hunks: list[Hunk],
        is_binary: bool,
    ) -> None:
        """Append accumulated file change if valid and not binary."""
        if current_file is not None and not is_binary:
            files.append(FileChange(filepath=current_file, hunks=hunks))

    def _extract_filepath(self, line: str) -> str:
        """Extract file path from a diff git header line."""
        match = re.match(self._FILE_HEADER_PATTERN, line)
        if not match:
            return self._EMPTY_STR
        old_path = match.group(1)
        new_path = match.group(2)
        if new_path in (self._DEV_NULL, self._DEV_NULL_REL):
            return old_path
        return new_path

    def _is_binary_line(self, line: str) -> bool:
        """Determine if diff line marks a binary file."""
        return line.startswith(self._BINARY_DIFF_MARKER) or line.startswith(
            self._GIT_BINARY_MARKER
        )

    def _is_hunk_header(self, line: str) -> bool:
        """Determine if diff line is a hunk header."""
        return bool(re.match(self._HUNK_HEADER_PATTERN, line))

    def _parse_hunk_header(self, line: str) -> Hunk:
        """Parse hunk header into a Hunk instance."""
        match = re.match(self._HUNK_HEADER_PATTERN, line)
        if not match:
            return Hunk(0, 0, 0, 0, [], [])
        old_start = int(match.group(1))
        old_count = (
            int(match.group(2))
            if match.group(2) is not None
            else self._DEFAULT_COUNT
        )
        new_start = int(match.group(3))
        new_count = (
            int(match.group(4))
            if match.group(4) is not None
            else self._DEFAULT_COUNT
        )
        return Hunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            old_lines=[],
            new_lines=[],
        )

    def _accumulate_hunk_lines(self, hunk: Hunk, line: str) -> None:
        """Accumulate diff lines into the current hunk."""
        if line.startswith(self._NO_NEWLINE_MARKER):
            return
        if line.startswith(self._ADD_PREFIX):
            hunk.new_lines.append(line[1:])
        elif line.startswith(self._REMOVE_PREFIX):
            hunk.old_lines.append(line[1:])
        elif line.startswith(self._CONTEXT_PREFIX):
            hunk.old_lines.append(line[1:])
            hunk.new_lines.append(line[1:])
        elif line == self._EMPTY_STR:
            hunk.old_lines.append(self._EMPTY_STR)
            hunk.new_lines.append(self._EMPTY_STR)

    def _count_patch_lines(self, file_changes: list[FileChange]) -> int:
        """Calculate total lines changed across all hunks."""
        total = 0
        for change in file_changes:
            for hunk in change.hunks:
                total += len(hunk.old_lines) + len(hunk.new_lines)
        return total
