"""Unit tests for TelemetryCallback verifying metric forwarding to TelemetryLogger."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.training.callback_handler import TelemetryCallback


class TestTelemetryCallback:
    """Test suite covering training lifecycle callbacks in TelemetryCallback."""

    _STEP: int = 150
    _EPOCH: float = 3.0
    _TRAIN_LOSS: float = 0.423
    _LEARNING_RATE: float = 1.8e-4
    _GRAD_NORM: float = 0.89
    _EVAL_LOSS: float = 0.512
    _EVAL_RUNTIME: float = 12.5

    def test_on_log_logs_metrics(self) -> None:
        """Verify on_log forwards formatted training telemetry."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        state = SimpleNamespace(global_step=self._STEP, epoch=self._EPOCH)
        logs = {
            "loss": self._TRAIN_LOSS,
            "learning_rate": self._LEARNING_RATE,
            "grad_norm": self._GRAD_NORM,
        }

        callback.on_log(args=None, state=state, control=None, logs=logs)

        logger.log_metrics.assert_called_once_with(
            {
                "phase": "sft",
                "step": self._STEP,
                "epoch": self._EPOCH,
                "train_loss": self._TRAIN_LOSS,
                "learning_rate": self._LEARNING_RATE,
                "grad_norm": self._GRAD_NORM,
            }
        )

    def test_on_log_empty_logs_ignored(self) -> None:
        """Verify empty logs dictionary produces no telemetry record."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        callback.on_log(args=None, state=SimpleNamespace(), control=None, logs=None)
        logger.log_metrics.assert_not_called()

    def test_on_evaluate_logs_eval_metrics(self) -> None:
        """Verify on_evaluate forwards evaluation loss and runtime."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        state = SimpleNamespace(global_step=self._STEP, epoch=self._EPOCH)
        metrics = {
            "eval_loss": self._EVAL_LOSS,
            "eval_runtime": self._EVAL_RUNTIME,
        }

        callback.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        assert logger.log_metrics.call_count == 1
        payload = logger.log_metrics.call_args[0][0]
        assert payload["phase"] == "sft_eval"
        assert payload["step"] == self._STEP
        assert payload["epoch"] == self._EPOCH
        assert payload["eval_loss"] == self._EVAL_LOSS
        assert payload["eval_runtime"] == self._EVAL_RUNTIME

    def test_on_evaluate_empty_metrics_ignored(self) -> None:
        """Verify empty metrics dictionary produces no evaluation record."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        callback.on_evaluate(args=None, state=SimpleNamespace(), control=None, metrics=None)
        logger.log_metrics.assert_not_called()

    def test_on_evaluate_appends_gpu_metrics_when_cuda_available(self) -> None:
        """Verify GPU allocation memory is captured when CUDA is active."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        state = SimpleNamespace(global_step=self._STEP, epoch=self._EPOCH)
        metrics = {"eval_loss": self._EVAL_LOSS}

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.memory_allocated.side_effect = [2_000_000_000, 3_000_000_000]

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "torch", mock_torch)
            callback.on_evaluate(args=None, state=state, control=None, metrics=metrics)

        payload = logger.log_metrics.call_args[0][0]
        assert payload["gpu_0_memory_gb"] == 2.0
        assert payload["gpu_1_memory_gb"] == 3.0

    def test_on_save_logs_info(self) -> None:
        """Verify on_save emits checkpoint creation informational message."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        state = SimpleNamespace(global_step=self._STEP, best_metric=self._EVAL_LOSS)

        callback.on_save(args=None, state=state, control=None)

        logger.log_info.assert_called_once_with(
            f"Checkpoint saved at step {self._STEP}, best metric: {self._EVAL_LOSS}"
        )

    def test_unhandled_trainer_event_returns_noop_callable(self) -> None:
        """Verify unhandled trainer event hook returns no-op callable."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        handler = getattr(callback, "on_step_begin")
        assert callable(handler)
        assert handler(None, None, None) is None

    def test_non_trainer_attribute_raises_attribute_error(self) -> None:
        """Verify accessing non-event attribute raises AttributeError."""
        logger = MagicMock()
        callback = TelemetryCallback(logger=logger)
        with pytest.raises(AttributeError, match="has no attribute 'invalid_attr'"):
            _ = getattr(callback, "invalid_attr")

