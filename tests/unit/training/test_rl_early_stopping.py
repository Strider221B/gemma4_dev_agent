"""Unit tests for RLEarlyStoppingCallback monitoring trajectory reward progress."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.training.rl_early_stopping import RLEarlyStoppingCallback


class TestRLEarlyStopping:
    """Test suite verifying RLEarlyStoppingCallback stopping conditions and plateau checks."""

    _MIN_IMPROVEMENT: float = 0.02
    _PATIENCE: int = 2
    _METRIC_KEY: str = "reward"

    def test_stops_after_patience_exceeded(self) -> None:
        """Verify callback halts training when evaluations fail to improve past patience."""
        callback = RLEarlyStoppingCallback(
            min_reward_improvement=self._MIN_IMPROVEMENT, patience=self._PATIENCE
        )
        control = MagicMock()
        control.should_training_stop = False

        # Evaluation 1: initial baseline
        callback.on_evaluate(None, None, control, {self._METRIC_KEY: 0.50})
        assert callback.best_reward == 0.50
        assert callback.wait_count == 0
        assert not callback.should_stop

        # Evaluation 2: no improvement (wait=1)
        callback.on_evaluate(None, None, control, {self._METRIC_KEY: 0.505})
        assert callback.wait_count == 1
        assert not callback.should_stop

        # Evaluation 3: no improvement (wait=2 == patience)
        callback.on_evaluate(None, None, control, {self._METRIC_KEY: 0.501})
        assert callback.wait_count == 2
        assert callback.should_stop
        assert control.should_training_stop is True

    def test_continues_when_improving(self) -> None:
        """Verify callback resets wait count and continues when reward improves significantly."""
        callback = RLEarlyStoppingCallback(
            min_reward_improvement=self._MIN_IMPROVEMENT, patience=self._PATIENCE
        )
        control = MagicMock()
        control.should_training_stop = False

        callback.on_evaluate(None, None, control, {self._METRIC_KEY: 0.40})
        assert callback.best_reward == 0.40
        assert callback.wait_count == 0

        # Improves past min_improvement (0.40 -> 0.45 >= 0.40 + 0.02)
        callback.on_evaluate(None, None, control, {self._METRIC_KEY: 0.45})
        assert callback.best_reward == 0.45
        assert callback.wait_count == 0
        assert not callback.should_stop
        assert control.should_training_stop is False

    def test_handles_empty_or_missing_metrics(self) -> None:
        """Verify callback safely handles None or metric dicts without reward keys."""
        callback = RLEarlyStoppingCallback()
        control = MagicMock()
        callback.on_evaluate(None, None, control, None)
        assert callback.best_reward is None
        callback.on_evaluate(None, None, control, {"other_metric": 12.3})
        assert callback.best_reward is None
