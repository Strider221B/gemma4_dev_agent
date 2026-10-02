"""Builder for constructing HuggingFace / TRL training arguments from SFTConfig."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.sft_config import SFTConfig


class TrainingConfigBuilder:
    """Constructs dictionary of training arguments compatible with TRL SFTConfig."""

    _CHECKPOINTS_DIR_NAME: str = "checkpoints"
    _DEFAULT_OPTIM: str = "adamw_8bit"
    _DEFAULT_LOGGING_STEPS: int = 10
    _DEFAULT_DATASET_NUM_PROC: int = 4
    _DEFAULT_SEED: int = 42
    _REPORT_TO_NONE: str = "none"
    _USE_REENTRANT_KEY: str = "use_reentrant"

    def __init__(self, config: SFTConfig) -> None:
        """Initialize builder with application SFT configuration."""
        self._config: SFTConfig = config

    def build_training_args(self) -> dict[str, object]:
        """Assemble complete training argument dictionary for TRL."""
        args: dict[str, object] = {}
        args.update(self._build_batch_args())
        args.update(self._build_optimization_args())
        args.update(self._build_evaluation_args())
        args.update(self._build_runtime_args())
        return args

    def _build_batch_args(self) -> dict[str, object]:
        """Construct batch and sequence configuration arguments."""
        training = self._config.training
        output_path = Path(self._config.output_dir) / self._CHECKPOINTS_DIR_NAME
        return {
            "output_dir": str(output_path),
            "num_train_epochs": training.num_epochs,
            "per_device_train_batch_size": training.per_device_train_batch_size,
            "per_device_eval_batch_size": training.per_device_train_batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "max_seq_length": training.max_seq_length,
            "packing": training.packing,
        }

    def _build_optimization_args(self) -> dict[str, object]:
        """Construct optimizer, learning rate, and precision arguments."""
        training = self._config.training
        return {
            "learning_rate": training.learning_rate,
            "lr_scheduler_type": training.lr_scheduler_type,
            "warmup_ratio": training.warmup_ratio,
            "weight_decay": training.weight_decay,
            "max_grad_norm": training.max_grad_norm,
            "optim": self._DEFAULT_OPTIM,
            "bf16": training.bf16,
            "gradient_checkpointing": training.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {self._USE_REENTRANT_KEY: False},
        }

    def _build_evaluation_args(self) -> dict[str, object]:
        """Construct evaluation, saving, and best model retention arguments."""
        training = self._config.training
        return {
            "eval_strategy": training.eval_strategy,
            "eval_steps": training.eval_steps,
            "save_strategy": training.save_strategy,
            "save_steps": training.save_steps,
            "load_best_model_at_end": training.load_best_model_at_end,
            "metric_for_best_model": training.metric_for_best_model,
            "greater_is_better": False,
            "save_total_limit": training.save_total_limit,
        }

    def _build_runtime_args(self) -> dict[str, object]:
        """Construct telemetry, reproducibility, and multiprocessing arguments."""
        return {
            "logging_steps": self._DEFAULT_LOGGING_STEPS,
            "logging_first_step": True,
            "report_to": self._REPORT_TO_NONE,
            "dataset_num_proc": self._DEFAULT_DATASET_NUM_PROC,
            "seed": self._DEFAULT_SEED,
            "data_seed": self._DEFAULT_SEED,
        }
