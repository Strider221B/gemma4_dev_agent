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
    _KEY_DEQUANTIZE: str = "dequantize"
    _KEY_QUANT_CONFIG: str = "quantization_config"
    _KEY_QUANT_METHOD: str = "quant_method"
    _METHOD_COMPRESSED_TENSORS: str = "compressed-tensors"
    _MSG_ADAPTER_MERGED: str = "SFT adapter merged into base model"
    _PROMPT_TEMPLATE: str = "<|turn>user\nProblem: {problem}\n<turn|>\n"

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
                "prompt": self._PROMPT_TEMPLATE.format(problem=t.problem_statement),
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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from src.utils.model_path_resolver import ModelPathResolver

        resolved_path = ModelPathResolver.resolve(config.model.name)
        model_config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
        kwargs: dict[str, Any] = {
            "device_map": self._DEVICE_MAP,
            "trust_remote_code": True,
            "torch_dtype": self._resolve_dtype(config.model.dtype),
        }
        quant_config = self._build_quantization_config(
            model_config, config.model.load_in_4bit, config.model.dtype
        )
        if quant_config is not None:
            kwargs[self._KEY_QUANT_CONFIG] = quant_config

        base_model: Any = AutoModelForCausalLM.from_pretrained(resolved_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
        peft_model: Any = PeftModel.from_pretrained(base_model, str(config.adapter_path))
        if hasattr(peft_model, "merge_and_unload"):
            peft_model = peft_model.merge_and_unload()
        self._telemetry.log_info(self._MSG_ADAPTER_MERGED)
        return peft_model, tokenizer

    def _build_quantization_config(
        self, model_config: object, load_in_4bit: bool, dtype_str: str
    ) -> object | None:
        """Resolve quantization config: dequantize compressed-tensors or apply BitsAndBytes."""
        quant_cfg = getattr(model_config, self._KEY_QUANT_CONFIG, None)
        if quant_cfg is not None:
            if self._is_compressed_tensors(quant_cfg):
                return self._create_dequantize_config(quant_cfg)
            return None
        if load_in_4bit:
            return self._create_bnb_config(dtype_str)
        return None

    def _create_bnb_config(self, dtype_str: str) -> object:
        """Construct BitsAndBytesConfig for 4-bit QLoRA training."""
        from transformers import BitsAndBytesConfig

        quant_cls: Any = BitsAndBytesConfig
        return quant_cls(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self._resolve_dtype(dtype_str),
            bnb_4bit_quant_type=self._BNB_QUANT_TYPE,
            bnb_4bit_use_double_quant=True,
        )

    def _create_dequantize_config(self, quant_cfg: object) -> object:
        """Create CompressedTensorsConfig with dequantize enabled for training."""
        from transformers import CompressedTensorsConfig

        if isinstance(quant_cfg, dict):
            cfg_dict = dict(quant_cfg)
            cfg_dict[self._KEY_DEQUANTIZE] = True
            return CompressedTensorsConfig(**cfg_dict)
        if hasattr(quant_cfg, "to_dict"):
            cfg_dict = getattr(quant_cfg, "to_dict")()
            cfg_dict[self._KEY_DEQUANTIZE] = True
            return CompressedTensorsConfig(**cfg_dict)
        if hasattr(quant_cfg, self._KEY_DEQUANTIZE):
            setattr(quant_cfg, self._KEY_DEQUANTIZE, True)
            return quant_cfg
        return CompressedTensorsConfig(dequantize=True)

    def _is_compressed_tensors(self, quant_cfg: object) -> bool:
        """Check whether quantization configuration uses compressed-tensors method."""
        if isinstance(quant_cfg, dict):
            return quant_cfg.get(self._KEY_QUANT_METHOD) == self._METHOD_COMPRESSED_TENSORS
        method = getattr(quant_cfg, self._KEY_QUANT_METHOD, None)
        if method == self._METHOD_COMPRESSED_TENSORS:
            return True
        return self._METHOD_COMPRESSED_TENSORS in str(type(quant_cfg)).lower()

    def _merge_and_reapply_lora(self, model: object, config: RLConfig) -> object:
        """Attach fresh LoRA adapter layers on top of merged SFT base model."""
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        from src.training.lora_target_resolver import LoRATargetModuleResolver

        lora_r = getattr(config.grpo, "lora_r", self._DEFAULT_LORA_R)
        resolved_targets = LoRATargetModuleResolver().resolve(model, list(self._TARGET_MODULES))
        lora_config = PeftLoraConfig(
            r=lora_r,
            lora_alpha=self._DEFAULT_LORA_ALPHA,
            lora_dropout=self._DEFAULT_LORA_DROPOUT,
            target_modules=resolved_targets,
            task_type=TaskType.CAUSAL_LM,
        )
        self._enable_input_grads(model)
        lora_model = get_peft_model(cast(Any, model), lora_config)
        self._enable_gradient_checkpointing(lora_model)
        return lora_model

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

    def _build_dpo_args(self, config: RLConfig, output_dir: str) -> object:
        """Construct arguments for DPOTrainer."""
        import trl

        dpo_config_cls: Any = getattr(trl, "DPOConfig")
        return dpo_config_cls(
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

    def _train_dpo(
        self, model: object, tokenizer: object, prompts: object, config: RLConfig
    ) -> str:
        """Execute DPO preference training on pairwise ranked trajectories."""
        import trl

        from src.training.callback_handler import TelemetryCallback

        output_dir = str(Path(config.output_dir) / self._SUBDIR_DPO_CHECKPOINTS)
        dpo_args = self._build_dpo_args(config, output_dir)
        dpo_trainer_cls: Any = getattr(trl, "DPOTrainer")
        callbacks: list[Any] = [TelemetryCallback(self._telemetry)]
        trainer = dpo_trainer_cls(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            args=dpo_args,
            train_dataset=prompts,
            callbacks=callbacks,
        )
        if hasattr(trainer, "train"):
            trainer.train()
        adapter_dir = str(Path(config.output_dir) / self._SUBDIR_RL_LORA)
        return self._checkpoint_mgr.save_adapter(model, adapter_dir)

    def _build_grpo_args(self, config: RLConfig, output_dir: str) -> object:
        """Construct arguments for GRPOTrainer."""
        import trl

        grpo_config_cls: Any = getattr(trl, "GRPOConfig")
        return grpo_config_cls(
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

    def _train_grpo(
        self, model: object, tokenizer: object, prompts: object, config: RLConfig
    ) -> str:
        """Execute GRPO policy optimization with multi-signal reward feedback."""
        import trl

        from src.training.callback_handler import TelemetryCallback
        from src.training.rl_early_stopping import RLEarlyStoppingCallback

        output_dir = str(Path(config.output_dir) / self._SUBDIR_GRPO_CHECKPOINTS)
        grpo_args = self._build_grpo_args(config, output_dir)
        callbacks: list[Any] = [
            TelemetryCallback(self._telemetry),
            RLEarlyStoppingCallback(
                min_reward_improvement=self._DEFAULT_MIN_REWARD_IMPROVEMENT,
                patience=self._DEFAULT_PATIENCE,
            ),
        ]
        grpo_trainer_cls: Any = getattr(trl, "GRPOTrainer")
        reward_funcs: list[Any] = [self._reward_model.compute_reward]
        trainer = grpo_trainer_cls(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=prompts,
            callbacks=callbacks,
        )
        if hasattr(trainer, "train"):
            trainer.train()
        adapter_dir = str(Path(config.output_dir) / self._SUBDIR_RL_LORA)
        return self._checkpoint_mgr.save_adapter(model, adapter_dir)
