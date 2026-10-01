"""Unit tests for training layer constants."""

from __future__ import annotations

import src.training.constants as training_constants


class TestTrainingConstants:
    """Test suite for training layer constants."""

    _EXPECTED_SIZE_LIMIT: int = 1_500_000_000
    _EXPECTED_LOSS_IMPROVEMENT: float = 0.001

    def test_training_layer_constants(self) -> None:
        """Verify training size limits and loss improvement thresholds."""
        assert training_constants.ADAPTER_SIZE_LIMIT_BYTES == self._EXPECTED_SIZE_LIMIT
        assert (
            training_constants.MIN_EVAL_LOSS_IMPROVEMENT
            == self._EXPECTED_LOSS_IMPROVEMENT
        )
        assert training_constants.DEFAULT_SEED == 42
