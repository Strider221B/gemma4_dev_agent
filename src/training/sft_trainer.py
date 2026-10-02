"""Pipeline orchestrating Supervised Fine-Tuning (SFT) on gold trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.config.lora_config import LoRAConfig
    from src.config.model_config import ModelConfig
    from src.config.sft_config import SFTConfig
    from src.config.training_config import TrainingConfig
    from src.training.checkpoint_manager import CheckpointManager
    from src.training.curriculum_scheduler import CurriculumScheduler
    from src.utils.telemetry_logger import TelemetryLogger


class SFTTrainerPipeline:
    """Orchestrates QLoRA supervised fine-tuning using HuggingFace transformers and TRL."""

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
    _BNB_QUANT_TYPE: str = "nf4"
    _DEVICE_MAP: str = "auto"
    _DTYPE_BFLOAT16: str = "bfloat16"
    _DTYPE_FLOAT16: str = "float16"
    _DTYPE_FLOAT32: str = "float32"
    _KW_USE_REENTRANT: str = "use_reentrant"

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
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        bias_val: Any = config.bias
        lora_config = PeftLoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias=bias_val,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        )
        self._enable_input_grads(model)
        lora_model = get_peft_model(cast(Any, model), lora_config)
        self._enable_gradient_checkpointing(lora_model)
        self._log_trainable_params(lora_model)
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

    def _enable_gradient_checkpointing(self, model: object) -> None:
        """Enable gradient checkpointing with non-reentrant mode for memory efficiency."""
        if hasattr(model, "gradient_checkpointing_enable"):
            getattr(model, "gradient_checkpointing_enable")(
                gradient_checkpointing_kwargs={self._KW_USE_REENTRANT: False}
            )

    def _enable_input_grads(self, model: object) -> None:
        """Enable input gradients required for LoRA training with quantised models."""
        if hasattr(model, "enable_input_require_grads"):
            getattr(model, "enable_input_require_grads")()

    def _load_model(self, config: ModelConfig) -> tuple[object, object]:
        """Load base language model and tokenizer using HuggingFace transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from src.utils.model_path_resolver import ModelPathResolver

        resolved_path = ModelPathResolver.resolve(config.name)
        quant_cls: Any = BitsAndBytesConfig
        quantization_config = quant_cls(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=self._resolve_dtype(config.dtype),
            bnb_4bit_quant_type=self._BNB_QUANT_TYPE,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            quantization_config=quantization_config,
            device_map=self._DEVICE_MAP,
            trust_remote_code=True,
            torch_dtype=self._resolve_dtype(config.dtype),
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
        return model, tokenizer

    def _log_trainable_params(self, model: object) -> None:
        """Log trainable vs total parameter counts from PEFT model."""
        if hasattr(model, "get_nb_trainable_parameters"):
            trainable, total = getattr(model, "get_nb_trainable_parameters")()
            self._telemetry.log_info(f"Trainable: {trainable:,} / {total:,}")

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

    def _resolve_dtype(self, dtype_str: str) -> object:
        """Convert string dtype identifier to torch dtype."""
        try:
            import torch

            dtype_map: dict[str, object] = {
                self._DTYPE_BFLOAT16: torch.bfloat16,
                self._DTYPE_FLOAT16: torch.float16,
                self._DTYPE_FLOAT32: torch.float32,
            }
            return dtype_map.get(dtype_str, torch.bfloat16)
        except ImportError:
            return dtype_str

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
