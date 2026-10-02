"""Reinforcement learning training pipeline orchestrating GRPO and DPO algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.config.rl_config import RLConfig
    from src.data.task import Task
    from src.training.checkpoint_manager import CheckpointManager
    from src.training.reward_model import RewardModel
    from src.utils.telemetry_logger import TelemetryLogger


class RLTrainerPipeline:
    """Orchestrates Reinforcement Learning on agent trajectories using GRPO and DPO."""

    _ADAPTER_SIZE_LIMIT: int = 1_500_000_000
    _MODE_GRPO: str = "grpo"
    _MODE_DPO: str = "dpo"
    _PHASE_RL: str = "rl"
    _SUBDIR_GRPO_CHECKPOINTS: str = "grpo_checkpoints"
    _SUBDIR_DPO_CHECKPOINTS: str = "dpo_checkpoints"
    _SUBDIR_RL_LORA: str = "rl_lora"
    _DEFAULT_LORA_R: int = 32
    _DEFAULT_LORA_ALPHA: int = 64
    _DEFAULT_LORA_DROPOUT: float = 0.05
    _DEFAULT_RANDOM_STATE: int = 42
    _DEFAULT_MIN_REWARD_IMPROVEMENT: float = 0.01
    _DEFAULT_PATIENCE: int = 5
    _REPORT_TO_NONE: str = "none"
    _KEY_MODE: str = "mode"
    _KEY_ADAPTER_SIZE_BYTES: str = "adapter_size_bytes"
    _TARGET_MODULES: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    _BNB_QUANT_TYPE: str = "nf4"
    _DEVICE_MAP: str = "auto"
    _DTYPE_BFLOAT16: str = "bfloat16"
    _DTYPE_FLOAT16: str = "float16"
    _DTYPE_FLOAT32: str = "float32"
    _KW_USE_REENTRANT: str = "use_reentrant"
    _MSG_ADAPTER_MERGED: str = "SFT adapter merged into base model"

    def __init__(
        self,
        reward_model: RewardModel,
        checkpoint_mgr: CheckpointManager,
        telemetry: TelemetryLogger,
    ) -> None:
        """Initialize RLTrainerPipeline with injected dependencies."""
        self._reward_model: RewardModel = reward_model
        self._checkpoint_mgr: CheckpointManager = checkpoint_mgr
        self._telemetry: TelemetryLogger = telemetry

    def run(self, config: RLConfig, tasks: list[Task]) -> str:
        """Execute RL training pipeline (GRPO or DPO) and return path to saved adapter."""
        self._telemetry.log_start(self._PHASE_RL, config)
        model, tokenizer = self._load_sft_model(config)
        model = self._merge_and_reapply_lora(model, config)
        prompt_dataset = self._build_prompts(tasks)
        if config.mode == self._MODE_GRPO:
            adapter_path = self._train_grpo(model, tokenizer, prompt_dataset, config)
        elif config.mode == self._MODE_DPO:
            adapter_path = self._train_dpo(model, tokenizer, prompt_dataset, config)
        else:
            raise ValueError(f"Unknown RL mode: {config.mode}")
        self._telemetry.log_end(
            self._PHASE_RL,
            {
                self._KEY_MODE: config.mode,
                self._KEY_ADAPTER_SIZE_BYTES: self._compute_size(adapter_path),
            },
        )
        return adapter_path

    def _build_prompts(self, tasks: list[Task]) -> object:
        """Format task statements into prompt dictionary dataset for RL training."""
        prompt_records: list[dict[str, str]] = [
            {
                "prompt": f"<start_of_turn>user\nProblem: {t.problem_statement}\n<end_of_turn>\n",
                "instance_id": t.instance_id,
            }
            for t in tasks
        ]
        try:
            from datasets import Dataset

            return Dataset.from_list(prompt_records)
        except ImportError:
            return prompt_records

    def _compute_size(self, path: str) -> int:
        """Compute recursive directory size using CheckpointManager helper."""
        return self._checkpoint_mgr._compute_total_size(path)

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

    def _load_sft_model(self, config: RLConfig) -> tuple[object, object]:
        """Load base pretrained language model and merge SFT adapter weights."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_cls: Any = BitsAndBytesConfig
        quantization_config = quant_cls(
            load_in_4bit=config.model.load_in_4bit,
            bnb_4bit_compute_dtype=self._resolve_dtype(config.model.dtype),
            bnb_4bit_quant_type=self._BNB_QUANT_TYPE,
            bnb_4bit_use_double_quant=True,
        )
        base_model: Any = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=quantization_config,
            device_map=self._DEVICE_MAP,
            torch_dtype=self._resolve_dtype(config.model.dtype),
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        peft_model: Any = PeftModel.from_pretrained(base_model, str(config.adapter_path))
        if hasattr(peft_model, "merge_and_unload"):
            peft_model = peft_model.merge_and_unload()
        self._telemetry.log_info(self._MSG_ADAPTER_MERGED)
        return peft_model, tokenizer

    def _merge_and_reapply_lora(self, model: object, config: RLConfig) -> object:
        """Attach fresh LoRA adapter layers on top of merged SFT base model."""
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        lora_r = getattr(config.grpo, "lora_r", self._DEFAULT_LORA_R)
        lora_config = PeftLoraConfig(
            r=lora_r,
            lora_alpha=self._DEFAULT_LORA_ALPHA,
            lora_dropout=self._DEFAULT_LORA_DROPOUT,
            target_modules=list(self._TARGET_MODULES),
            task_type=TaskType.CAUSAL_LM,
        )
        self._enable_input_grads(model)
        lora_model = get_peft_model(cast(Any, model), lora_config)
        self._enable_gradient_checkpointing(lora_model)
        return lora_model

    def _resolve_dtype(self, dtype_str: str) -> object:
        """Convert string dtype identifier to torch dtype."""
        import torch

        dtype_map: dict[str, object] = {
            self._DTYPE_BFLOAT16: torch.bfloat16,
            self._DTYPE_FLOAT16: torch.float16,
            self._DTYPE_FLOAT32: torch.float32,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def _train_dpo(
        self, model: object, tokenizer: object, prompts: object, config: RLConfig
    ) -> str:
        """Execute DPO preference training on pairwise ranked trajectories."""
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer

        from src.training.callback_handler import TelemetryCallback

        output_dir = str(Path(config.output_dir) / self._SUBDIR_DPO_CHECKPOINTS)
        dpo_args = TRLDPOConfig(
            output_dir=output_dir,
            beta=config.dpo.beta,
            loss_type=config.dpo.loss_type,
            max_length=config.dpo.max_length,
            max_prompt_length=config.dpo.max_prompt_length,
            num_train_epochs=config.dpo.num_epochs,
            per_device_train_batch_size=config.dpo.per_device_train_batch_size,
            gradient_accumulation_steps=config.dpo.gradient_accumulation_steps,
            learning_rate=config.dpo.learning_rate,
            bf16=True,
            logging_steps=5,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=2,
            seed=self._DEFAULT_RANDOM_STATE,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            args=dpo_args,
            train_dataset=prompts,
            callbacks=[TelemetryCallback(self._telemetry)],
        )
        if hasattr(trainer, "train"):
            trainer.train()
        adapter_dir = str(Path(config.output_dir) / self._SUBDIR_RL_LORA)
        return self._checkpoint_mgr.save_adapter(model, adapter_dir)

    def _train_grpo(
        self, model: object, tokenizer: object, prompts: object, config: RLConfig
    ) -> str:
        """Execute GRPO policy optimization with multi-signal reward feedback."""
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer

        from src.training.callback_handler import TelemetryCallback
        from src.training.rl_early_stopping import RLEarlyStoppingCallback

        output_dir = str(Path(config.output_dir) / self._SUBDIR_GRPO_CHECKPOINTS)
        grpo_args = TRLGRPOConfig(
            output_dir=output_dir,
            num_generations=config.grpo.num_generations,
            max_new_tokens=config.grpo.max_new_tokens,
            temperature=config.grpo.temperature,
            top_p=config.grpo.top_p,
            beta=config.grpo.beta,
            num_train_epochs=config.grpo.num_epochs,
            per_device_train_batch_size=config.grpo.per_device_train_batch_size,
            gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
            learning_rate=config.grpo.learning_rate,
            lr_scheduler_type=config.grpo.lr_scheduler_type,
            warmup_ratio=config.grpo.warmup_ratio,
            max_grad_norm=config.grpo.max_grad_norm,
            bf16=True,
            logging_steps=5,
            report_to=self._REPORT_TO_NONE,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=2,
            seed=self._DEFAULT_RANDOM_STATE,
        )
        callbacks = [
            TelemetryCallback(self._telemetry),
            RLEarlyStoppingCallback(
                min_reward_improvement=self._DEFAULT_MIN_REWARD_IMPROVEMENT,
                patience=self._DEFAULT_PATIENCE,
            ),
        ]
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[self._reward_model.compute_reward],
            args=grpo_args,
            train_dataset=prompts,
            callbacks=callbacks,
        )
        if hasattr(trainer, "train"):
            trainer.train()
        adapter_dir = str(Path(config.output_dir) / self._SUBDIR_RL_LORA)
        return self._checkpoint_mgr.save_adapter(model, adapter_dir)
