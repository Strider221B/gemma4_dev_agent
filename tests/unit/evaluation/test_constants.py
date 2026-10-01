"""Unit tests for evaluation layer constants."""

from __future__ import annotations

import src.evaluation.constants as eval_constants


class TestEvaluationConstants:
    """Test suite for evaluation layer constants."""

    _EXPECTED_RESOLUTION_RATE: float = 0.10
    _EXPECTED_GAP_THRESHOLD: float = 0.15

    def test_evaluation_layer_constants(self) -> None:
        """Verify evaluation resolution thresholds and nudge limits."""
        assert eval_constants.MIN_RESOLUTION_RATE == self._EXPECTED_RESOLUTION_RATE
        assert eval_constants.GENERALIZATION_GAP_THRESHOLD == self._EXPECTED_GAP_THRESHOLD
        assert eval_constants.MAX_NUDGES == 3
