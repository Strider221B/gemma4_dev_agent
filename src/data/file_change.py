"""Data model representing file modifications in a patch."""

from dataclasses import dataclass

from src.data.hunk import Hunk


@dataclass(frozen=True)
class FileChange:
    """Represents a set of hunk changes applied to a specific file."""

    filepath: str
    hunks: list[Hunk]
