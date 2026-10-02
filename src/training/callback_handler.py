"""Telemetry callback integration for HuggingFace / TRL trainers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utils.telemetry_logger import TelemetryLogger


class TelemetryCallback:
    """Custom trainer callback recording loss, runtime, and GPU metrics via TelemetryLogger."""

    _PHASE_SFT: str = "sft"
    _PHASE_SFT_EVAL: str = "sft_eval"
    _KEY_PHASE: str = "phase"
    _KEY_STEP: str = "step"
    _KEY_EPOCH: str = "epoch"
    _KEY_TRAIN_LOSS: str = "train_loss"
    _KEY_LOSS: str = "loss"
    _KEY_LEARNING_RATE: str = "learning_rate"
    _KEY_GRAD_NORM: str = "grad_norm"
    _KEY_EVAL_LOSS: str = "eval_loss"
    _KEY_EVAL_RUNTIME: str = "eval_runtime"
    _BYTES_PER_GIGABYTE: float = 1e9
    _DEFAULT_GLOBAL_STEP: int = 0
    _DEFAULT_EPOCH: float = 0.0
    _EVENT_PREFIX: str = "on_"
    _SAVE_LOG_TEMPLATE: str = "Checkpoint saved at step {step}, best metric: {best_metric}"

    def __init__(self, logger: TelemetryLogger) -> None:
        """Initialize TelemetryCallback with telemetry recording backend."""
        self._logger: TelemetryLogger = logger

    def __getattr__(self, name: str) -> Any:
        """Provide no-op handlers for unhandled trainer callback events."""
        if name.startswith(self._EVENT_PREFIX):
            return self._noop_event
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def on_evaluate(
        self,
        args: object,
        state: object,
        control: object,
        metrics: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Record evaluation loss, runtime, and hardware usage metrics."""
        if not metrics:
            return
        global_step = getattr(state, "global_step", self._DEFAULT_GLOBAL_STEP)
        epoch = getattr(state, "epoch", self._DEFAULT_EPOCH)
        eval_payload: dict[str, object] = {
            self._KEY_PHASE: self._PHASE_SFT_EVAL,
            self._KEY_STEP: global_step,
            self._KEY_EPOCH: epoch,
            self._KEY_EVAL_LOSS: metrics.get(self._KEY_EVAL_LOSS),
            self._KEY_EVAL_RUNTIME: metrics.get(self._KEY_EVAL_RUNTIME),
        }
        self._append_gpu_metrics(eval_payload)
        self._logger.log_metrics(eval_payload)

    def on_log(
        self,
        args: object,
        state: object,
        control: object,
        logs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Record periodic training progress metrics to telemetry logger."""
        if not logs:
            return
        global_step = getattr(state, "global_step", self._DEFAULT_GLOBAL_STEP)
        epoch = getattr(state, "epoch", self._DEFAULT_EPOCH)
        self._logger.log_metrics(
            {
                self._KEY_PHASE: self._PHASE_SFT,
                self._KEY_STEP: global_step,
                self._KEY_EPOCH: epoch,
                self._KEY_TRAIN_LOSS: logs.get(self._KEY_LOSS),
                self._KEY_LEARNING_RATE: logs.get(self._KEY_LEARNING_RATE),
                self._KEY_GRAD_NORM: logs.get(self._KEY_GRAD_NORM),
            }
        )

    def on_save(
        self,
        args: object,
        state: object,
        control: object,
        **kwargs: object,
    ) -> None:
        """Record checkpoint persistence notification."""
        global_step = getattr(state, "global_step", self._DEFAULT_GLOBAL_STEP)
        best_metric = getattr(state, "best_metric", None)
        message = self._SAVE_LOG_TEMPLATE.format(
            step=global_step,
            best_metric=best_metric,
        )
        self._logger.log_info(message)

    def _append_gpu_metrics(self, payload: dict[str, object]) -> None:
        """Attach GPU memory allocation metrics if PyTorch CUDA is available."""
        try:
            import torch

            if torch.cuda.is_available():
                for device_idx in range(torch.cuda.device_count()):
                    allocated_gb = (
                        torch.cuda.memory_allocated(device_idx)
                        / self._BYTES_PER_GIGABYTE
                    )
                    payload[f"gpu_{device_idx}_memory_gb"] = allocated_gb
        except Exception:
            pass

    def _noop_event(self, *args: object, **kwargs: object) -> None:
        """No-op fallback for unused trainer callback events."""
        return None
