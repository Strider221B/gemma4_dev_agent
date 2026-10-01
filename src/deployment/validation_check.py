"""Data model representing a pre-submission deployment validation check."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationCheck:
    """Represents the outcome of a submission constraint validation check."""

    name: str
    passed: bool
    detail: str
