"""Validation result data model for trajectory quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of validating a trajectory."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
