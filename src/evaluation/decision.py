"""Data model representing competitive adoption decision for a peer solution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """Represents an actionable decision regarding a competitor solution."""

    action: str
    reason: str
    confidence: str
    components_to_adopt: list[str] | None = None
