"""Protocol defining the contract for model verification smoke testers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelVerificationRunner(Protocol):
    """Protocol defining the contract for model verification smoke testers."""

    def run_verification(self) -> dict[str, object]:
        """Execute verification workflow and return status summary."""
        ...
