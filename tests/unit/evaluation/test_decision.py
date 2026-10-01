"""Unit tests for Decision data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.decision import Decision


class TestDecision:
    """Test suite for Decision frozen dataclass."""

    _ACTION: str = "ADOPT"
    _REASON: str = "Statistically significant CV improvement"
    _CONFIDENCE: str = "high"
    _COMPONENT: str = "system_prompt"
    _MODIFIED_ACTION: str = "REJECT"

    def test_decision_defaults(self) -> None:
        """Verify Decision optional components default to None."""
        dec = Decision(
            action=self._ACTION,
            reason=self._REASON,
            confidence=self._CONFIDENCE,
        )
        assert dec.action == self._ACTION
        assert dec.reason == self._REASON
        assert dec.confidence == self._CONFIDENCE
        assert dec.components_to_adopt is None

    def test_decision_with_components(self) -> None:
        """Verify Decision correctly retains components to adopt."""
        dec = Decision(
            action=self._ACTION,
            reason=self._REASON,
            confidence=self._CONFIDENCE,
            components_to_adopt=[self._COMPONENT],
        )
        assert dec.components_to_adopt == [self._COMPONENT]

    def test_decision_is_frozen(self) -> None:
        """Verify modifying field on Decision raises FrozenInstanceError."""
        dec = Decision(
            action=self._ACTION,
            reason=self._REASON,
            confidence=self._CONFIDENCE,
        )
        with pytest.raises(FrozenInstanceError):
            dec.action = self._MODIFIED_ACTION
