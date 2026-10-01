"""Task data model representing a SWE-bench benchmark instance."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """Represents a software engineering problem instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    patch: str
    test_patch: str
    created_at: str
