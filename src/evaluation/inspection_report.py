"""Data model representing a full inspection report for a peer solution."""

from dataclasses import dataclass, field

from src.evaluation.inspection_check import InspectionCheck
from src.evaluation.peer_solution import PeerSolution


@dataclass(frozen=True)
class InspectionReport:
    """Represents comprehensive findings from static inspection of a notebook."""

    solution: PeerSolution
    checks: list[InspectionCheck]
    architecture: str | None = None
    prompts: list[str] = field(default_factory=list)
    training: str | None = None
    novel_techniques: list[str] = field(default_factory=list)
    risk_level: str = "low"
