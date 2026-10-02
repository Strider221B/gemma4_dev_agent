"""Early stopping callback monitoring trajectory reward improvement for RL training."""

from __future__ import annotations


class RLEarlyStoppingCallback:
    """Trainer callback enforcing early stopping based on reward evaluation plateaus."""

    _REWARD_KEYS: tuple[str, ...] = (
        "reward",
        "eval_reward",
        "mean_reward",
        "eval_mean_reward",
        "reward_mean",
    )
    _CONTROL_STOP_ATTR: str = "should_training_stop"

    def __init__(
        self, min_reward_improvement: float = 0.01, patience: int = 5
    ) -> None:
        """Initialize callback with reward improvement delta and patience steps."""
        self._min_reward_improvement: float = min_reward_improvement
        self._patience: int = patience
        self._best_reward: float | None = None
        self._wait_count: int = 0
        self._should_stop: bool = False

    @property
    def best_reward(self) -> float | None:
        """Return the highest recorded reward metric."""
        return self._best_reward

    @property
    def should_stop(self) -> bool:
        """Return whether the stopping condition has been triggered."""
        return self._should_stop

    @property
    def wait_count(self) -> int:
        """Return current consecutive non-improving evaluation count."""
        return self._wait_count

    def on_evaluate(
        self,
        args: object,
        state: object,
        control: object,
        metrics: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Check reward metric improvement and signal training termination on plateau."""
        if not metrics:
            return
        current_reward = self._extract_reward(metrics)
        if current_reward is None:
            return
        if self._best_reward is None:
            self._best_reward = current_reward
            self._wait_count = 0
            return
        if current_reward >= self._best_reward + self._min_reward_improvement:
            self._best_reward = current_reward
            self._wait_count = 0
        else:
            self._wait_count += 1
            if self._wait_count >= self._patience:
                self._should_stop = True
                if control is not None and hasattr(control, self._CONTROL_STOP_ATTR):
                    setattr(control, self._CONTROL_STOP_ATTR, True)

    def _extract_reward(self, metrics: dict[str, object]) -> float | None:
        """Extract primary reward scalar from evaluation metrics dictionary."""
        for key in self._REWARD_KEYS:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, (int, float)):
                    return float(val)
        return None
