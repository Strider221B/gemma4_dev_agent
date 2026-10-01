"""Data model representing the aggregated pre-submission validation report."""

from dataclasses import dataclass

from src.deployment.validation_check import ValidationCheck


@dataclass(frozen=True)
class ValidationReport:
    """Represents the complete pre-flight check report before Kaggle submission."""

    checks: list[ValidationCheck]
    all_passed: bool
