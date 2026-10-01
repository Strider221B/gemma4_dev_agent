"""Data model representing unit test verification results on a patched repo."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    """Represents verification test suite execution results."""

    resolved: bool
    error: str | None = None
    passed_tests: int = 0
    failed_tests: int = 0
    total_tests: int = 0
