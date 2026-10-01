"""Data model representing a unified diff hunk."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hunk:
    """Represents a single diff hunk within a file patch."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_lines: list[str]
    new_lines: list[str]
