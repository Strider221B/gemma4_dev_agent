"""Data model representing a static analysis or inspection check result."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InspectionCheck:
    """Represents the outcome of a single inspection rule or scan."""

    name: str
    passed: bool
    findings: list[str]
    severity: str = "info"
