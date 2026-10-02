"""Pipeline orchestrating Supervised Fine-Tuning (SFT) on gold trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.lora_config import LoRAConfig
    from src.config.model_config import ModelConfig
    from src.config.sft_config import SFTConfig
    from src.config.training_config import TrainingConfig
    from src.training.checkpoint_manager import CheckpointManager
    from src.training.curriculum_scheduler import CurriculumScheduler
    from src.utils.telemetry_logger import TelemetryLogger


class SFTTrainerPipeline:
    """Orchestrates QLoRA supervised fine-tuning using Unsloth and TRL."""

    _ADAPTER_SIZE_LIMIT: int = 1_500_000_000
    _MIN_EVAL_LOSS_IMPROVEMENT: float = 0.001
    _REQUIRED_SPECIAL_TOKENS: tuple[str, ...] = (
        "<start_of_turn>",
        "<end_of_turn>",
        "<|tool_call|>",
        "<|/tool_call|>",
        "<|thought|>",
        "<|/thought|>",
    )
    _DEFAULT_SFT_LORA_SUBDIR: str = "sft_lora"
    _PHASE_SFT: str = "sft"
    _EARLY_STOPPING_PATIENCE: int = 3
    _DEFAULT_RANDOM_STATE: int = 42
    _FOLD_KEY: str = "fold"

    def __init__(
        self,
        checkpoint_mgr: CheckpointManager,
        telemetry: TelemetryLogger,
        curriculum: CurriculumScheduler,
    ) -> None:
        """Initialize SFT pipeline with dependencies injected via constructor."""
        self._checkpoint_mgr: CheckpointManager = checkpoint_mgr
        self._telemetry: TelemetryLogger = telemetry
        self._curriculum: CurriculumScheduler = curriculum

    def run(self, config: SFTConfig, dataset: object) -> str:
        """Execute complete SFT training pipeline and return path to saved adapter."""
        self._telemetry.log_start(self._PHASE_SFT, config)
        model, tokenizer = self._load_model(config.model)
        self._verify_special_tokens(tokenizer)
        model = self._apply_lora(model, config.lora)
        train_ds, val_ds = self._prepare_splits(dataset, config.eval_fold)
        if config.training.curriculum_enabled:
            train_ds = self._apply_curriculum(train_ds, config.training)
        trainer = self._create_trainer(model, tokenizer, train_ds, val_ds, config)
        train_result = getattr(trainer, "train")()
        adapter_dir = str(Path(config.output_dir) / self._DEFAULT_SFT_LORA_SUBDIR)
        adapter_path = self._save_adapter(model, adapter_dir)
        self._log_training_end(train_result, trainer, adapter_path)
        return adapter_path

    def _apply_curriculum(self, dataset: object, config: TrainingConfig) -> object:
        """Apply curriculum filter for the initial training stage."""
        return self._curriculum.get_epoch_data(dataset, 1, config.num_epochs)

    def _apply_lora(self, model: object, config: LoRAConfig) -> object:
        """Apply QLoRA adapters targeting specified linear projection layers."""
        from unsloth import FastLanguageModel

        lora_model = FastLanguageModel.get_peft_model(
            model,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias=config.bias,
            use_gradient_checkpointing="unsloth",
            random_state=self._DEFAULT_RANDOM_STATE,
            use_rslora=True,
            loftq_config=None,
        )
        if hasattr(lora_model, "get_nb_trainable_parameters"):
            trainable, total = lora_model.get_nb_trainable_parameters()
            self._telemetry.log_info(f"Trainable: {trainable:,} / {total:,}")
        return lora_model

    def _compute_size(self, path: str) -> int:
        """Compute total size of adapter directory via CheckpointManager."""
        return self._checkpoint_mgr._compute_total_size(path)

    def _create_trainer(
        self,
        model: object,
        tokenizer: object,
        train_ds: object,
        val_ds: object,
        config: SFTConfig,
    ) -> object:
        """Build and configure TRL SFTTrainer with callbacks and hyperparameters."""
        from transformers import EarlyStoppingCallback
        from trl import SFTConfig as TRLSFTConfig
        from trl import SFTTrainer

        from src.training.callback_handler import TelemetryCallback
        from src.training.training_config import TrainingConfigBuilder

        args_dict = TrainingConfigBuilder(config).build_training_args()
        training_args = TRLSFTConfig(**args_dict)
        callbacks = [
            TelemetryCallback(self._telemetry),
            EarlyStoppingCallback(
                early_stopping_patience=self._EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=self._MIN_EVAL_LOSS_IMPROVEMENT,
            ),
        ]
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=training_args,
            callbacks=callbacks,
        )

    def _load_model(self, config: ModelConfig) -> tuple[object, object]:
        """Load base language model and tokenizer using Unsloth."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.name,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
            device_map="auto",
            trust_remote_code=True,
        )
        return model, tokenizer

    def _log_training_end(
        self, train_result: object, trainer: object, adapter_path: str
    ) -> None:
        """Record final training metrics upon successful completion."""
        training_loss = getattr(train_result, "training_loss", None)
        trainer_state = getattr(trainer, "state", None)
        best_metric = getattr(trainer_state, "best_metric", None)
        global_step = getattr(trainer_state, "global_step", 0)
        adapter_size = self._compute_size(adapter_path)
        self._telemetry.log_end(
            self._PHASE_SFT,
            {
                "final_train_loss": training_loss,
                "best_eval_loss": best_metric,
                "total_steps": global_step,
                "adapter_size_bytes": adapter_size,
            },
        )

    def _prepare_splits(
        self, dataset: object, eval_fold: int
    ) -> tuple[object, object]:
        """Partition dataset into training and validation splits according to fold."""
        if hasattr(dataset, "filter"):
            train_ds = getattr(dataset, "filter")(
                lambda item: (
                    item.get(self._FOLD_KEY)
                    if isinstance(item, dict)
                    else getattr(item, self._FOLD_KEY, None)
                )
                != eval_fold
            )
            val_ds = getattr(dataset, "filter")(
                lambda item: (
                    item.get(self._FOLD_KEY)
                    if isinstance(item, dict)
                    else getattr(item, self._FOLD_KEY, None)
                )
                == eval_fold
            )
            return train_ds, val_ds
        if isinstance(dataset, list):
            train_items = [
                item
                for item in dataset
                if (
                    item.get(self._FOLD_KEY)
                    if isinstance(item, dict)
                    else getattr(item, self._FOLD_KEY, None)
                )
                != eval_fold
            ]
            val_items = [
                item
                for item in dataset
                if (
                    item.get(self._FOLD_KEY)
                    if isinstance(item, dict)
                    else getattr(item, self._FOLD_KEY, None)
                )
                == eval_fold
            ]
            return train_items, val_items
        return dataset, dataset

    def _save_adapter(self, model: object, path: str) -> str:
        """Save trained adapter checkpoint via CheckpointManager."""
        return self._checkpoint_mgr.save_adapter(model, path)

    def _verify_special_tokens(self, tokenizer: object) -> None:
        """Verify tokenizer vocab contains all mandatory tool and turn delimiters."""
        vocab: dict[str, int] = {}
        if hasattr(tokenizer, "get_vocab"):
            vocab = getattr(tokenizer, "get_vocab")()
        for token in self._REQUIRED_SPECIAL_TOKENS:
            if token not in vocab:
                raise ValueError(f"Missing required special token: {token}")
