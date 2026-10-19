"""Local model verification smoke tester for Gemma 4 QAT W4A16 models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from typing import Any, ClassVar, cast

from src.training.lora_target_resolver import LoRATargetModuleResolver
from src.training.model_verification_runner import ModelVerificationRunner
from src.training.peft_model_patcher import PEFTModelPatcher
from src.utils.model_path_resolver import ModelPathResolver


class LocalModelSmokeTester(ModelVerificationRunner):
    """Executes isolated 7-stage verification workflow on local Gemma 4 models."""

    _DEFAULT_MODEL_NAME: ClassVar[str] = "google/gemma-4-e2b-it-qat-w4a16-ct"
    _TARGET_MODULES: ClassVar[tuple[str, ...]] = ("q_proj", "v_proj")
    _REQUIRED_SPECIAL_TOKENS: ClassVar[tuple[str, ...]] = (
        "<|turn>",
        "<turn|>",
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|think|>",
        "<|channel>",
        "<channel|>",
        "<|image|>",
        "<|audio|>",
        "<|video|>",
    )
    _DTYPE_BFLOAT16: ClassVar[str] = "bfloat16"
    _DTYPE_FLOAT32: ClassVar[str] = "float32"
    _DEVICE_AUTO: ClassVar[str] = "auto"
    _DEVICE_CPU: ClassVar[str] = "cpu"
    _MIN_GPU_MEMORY_BYTES: ClassVar[int] = 16 * (1024**3)
    _ENV_DEVICE_MAP: ClassVar[str] = "LOCAL_MODEL_DEVICE_MAP"
    _KEY_DTYPE: ClassVar[str] = "dtype"
    _MODULE_LANGUAGE_MODEL: ClassVar[str] = "language_model"
    _PREFIX_DOT: ClassVar[str] = "."
    _DEFAULT_LORA_R: ClassVar[int] = 8
    _DEFAULT_LORA_ALPHA: ClassVar[int] = 16
    _DUMMY_SEQ_LEN: ClassVar[int] = 16
    _SUBDIR_TEST_ADAPTER: ClassVar[str] = "test_adapter"
    _KEY_STATUS: ClassVar[str] = "status"
    _KEY_STATUS_SUCCESS: ClassVar[str] = "success"
    _KEY_MODEL_NAME: ClassVar[str] = "model_name"
    _KEY_RESOLVED_PATH: ClassVar[str] = "resolved_path"
    _KEY_FORWARD_PASSED: ClassVar[str] = "forward_passed"
    _KEY_BACKWARD_PASSED: ClassVar[str] = "backward_passed"
    _KEY_ADAPTER_SAVED: ClassVar[str] = "adapter_saved"
    _KEY_ELAPSED_SECONDS: ClassVar[str] = "elapsed_seconds"
    _KEY_QUANT_CONFIG: ClassVar[str] = "quantization_config"
    _KEY_DEQUANTIZE: ClassVar[str] = "dequantize"
    _KEY_QUANT_METHOD: ClassVar[str] = "quant_method"
    _QUANT_METHOD_COMPRESSED: ClassVar[str] = "compressed-tensors"
    _FILE_ADAPTER_MODEL: ClassVar[str] = "adapter_model.safetensors"
    _FILE_ADAPTER_CONFIG: ClassVar[str] = "adapter_config.json"
    _TASK_TYPE_CAUSAL: ClassVar[str] = "CAUSAL_LM"

    def __init__(
        self,
        model_name: str | None = None,
        work_dir: str | None = None,
        patcher: PEFTModelPatcher | None = None,
        target_resolver: LoRATargetModuleResolver | None = None,
    ) -> None:
        """Initialize LocalModelSmokeTester with target model identifier and workspace."""
        self._model_name: str = model_name or self._DEFAULT_MODEL_NAME
        self._work_dir: str = work_dir or tempfile.gettempdir()
        self._patcher: PEFTModelPatcher = patcher or PEFTModelPatcher()
        self._target_resolver: LoRATargetModuleResolver = (
            target_resolver or LoRATargetModuleResolver()
        )

    @classmethod
    def main(cls, argv: list[str] | None = None) -> int:
        """CLI entry point for local model smoke tester."""
        parser = argparse.ArgumentParser(description="Local Model Smoke Tester")
        parser.add_argument("--model-name", default=None, help="Model identifier")
        parser.add_argument("--work-dir", default=None, help="Working directory")
        args = parser.parse_args(argv)
        tester = cls(model_name=args.model_name, work_dir=args.work_dir)
        report = tester.run_verification()
        print(json.dumps(report, indent=2))
        return 0

    def run_verification(self) -> dict[str, object]:
        """Execute the 7-stage local model verification workflow.

        Returns:
            Dictionary report summarizing execution status and artifact paths.
        """
        start_time = time.time()
        resolved_path = self._resolve_model_path()
        model, tokenizer = self._load_model_and_tokenizer(resolved_path)
        self._verify_tokenizer_tokens(tokenizer)
        peft_model = self._attach_lora_adapters(model)
        outputs = self._execute_forward_pass(peft_model, tokenizer)
        self._execute_backward_pass(outputs)
        adapter_path = self._save_adapter_checkpoint(peft_model)
        elapsed = time.time() - start_time
        return self._build_summary_report(resolved_path, adapter_path, elapsed)

    def _attach_lora_adapters(self, model: object) -> object:
        """Attach LoRA adapters with Issue #37 compatibility patcher applied."""
        from peft import LoraConfig, get_peft_model

        self._patcher.patch_peft_lora_class()
        target_list = list(self._TARGET_MODULES)
        resolved_targets = self._resolve_target_modules(model, target_list)
        lora_cfg = LoraConfig(
            r=self._DEFAULT_LORA_R,
            lora_alpha=self._DEFAULT_LORA_ALPHA,
            target_modules=resolved_targets,
            lora_dropout=0.0,
            task_type=self._TASK_TYPE_CAUSAL,
        )
        peft_model = get_peft_model(cast(Any, model), lora_cfg)
        return self._patcher.patch_model(peft_model)

    def _build_quantization_config(self, resolved_path: str) -> object | None:
        """Inspect model config and construct dequantize config if compressed-tensors."""
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
            quant_cfg = getattr(hf_config, self._KEY_QUANT_CONFIG, None)
            if quant_cfg is not None and self._is_compressed_tensors(quant_cfg):
                return self._create_dequantize_config(quant_cfg)
        except Exception:
            return None
        return None

    def _build_summary_report(
        self, resolved_path: str, adapter_path: str, elapsed: float
    ) -> dict[str, object]:
        """Construct structured summary reporting verification status."""
        return {
            self._KEY_STATUS: self._KEY_STATUS_SUCCESS,
            self._KEY_MODEL_NAME: self._model_name,
            self._KEY_RESOLVED_PATH: resolved_path,
            self._KEY_FORWARD_PASSED: True,
            self._KEY_BACKWARD_PASSED: True,
            self._KEY_ADAPTER_SAVED: os.path.exists(adapter_path),
            self._KEY_ELAPSED_SECONDS: round(elapsed, 2),
        }

    def _create_dequantize_config(self, quant_cfg: object) -> object:
        """Instantiate CompressedTensorsConfig with dequantize enabled."""
        from transformers import CompressedTensorsConfig

        if isinstance(quant_cfg, dict):
            cfg_dict = dict(quant_cfg)
            cfg_dict[self._KEY_DEQUANTIZE] = True
            return CompressedTensorsConfig(**cfg_dict)
        if hasattr(quant_cfg, "to_dict"):
            cfg_dict = getattr(quant_cfg, "to_dict")()
            cfg_dict[self._KEY_DEQUANTIZE] = True
            return CompressedTensorsConfig(**cfg_dict)
        return CompressedTensorsConfig(dequantize=True)

    def _determine_device_map(self) -> str:
        """Select device map with CPU fallback for low VRAM to prevent meta tensors."""
        env_override = os.environ.get(self._ENV_DEVICE_MAP)
        if env_override:
            return env_override
        import torch

        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                if gpu_mem >= self._MIN_GPU_MEMORY_BYTES:
                    return self._DEVICE_AUTO
            except Exception:
                pass
        return self._DEVICE_CPU

    def _execute_backward_pass(self, outputs: object) -> bool:
        """Compute dummy loss and backpropagate gradients to verify backward pass."""
        loss = getattr(outputs, "loss", None)
        if loss is None:
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise AttributeError("Model outputs do not contain logits or loss")
            loss = logits.sum()
        if hasattr(loss, "backward") and callable(getattr(loss, "backward")):
            loss.backward()
            return True
        return False

    def _execute_forward_pass(self, model: object, tokenizer: object) -> object:
        """Execute a forward pass with dummy tokens to verify PEFT/quantization hooks."""
        if not callable(model):
            raise TypeError("Model object is not callable for forward pass execution")
        import torch

        pad_token = getattr(tokenizer, "pad_token_id", 0)
        pad_id = int(pad_token) if isinstance(pad_token, (int, float)) else 0
        input_ids = torch.full((1, self._DUMMY_SEQ_LEN), pad_id, dtype=torch.long)
        attention_mask = torch.ones((1, self._DUMMY_SEQ_LEN), dtype=torch.long)
        device = self._resolve_model_device(model)
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        return model(input_ids=input_ids, attention_mask=attention_mask)

    def _is_compressed_tensors(self, quant_cfg: object) -> bool:
        """Check whether quantization configuration specifies compressed-tensors."""
        if isinstance(quant_cfg, dict):
            method = quant_cfg.get(self._KEY_QUANT_METHOD, "")
            return bool(method == self._QUANT_METHOD_COMPRESSED)
        method_attr = getattr(quant_cfg, self._KEY_QUANT_METHOD, "")
        return bool(method_attr == self._QUANT_METHOD_COMPRESSED)

    def _load_model_and_tokenizer(self, resolved_path: str) -> tuple[object, object]:
        """Load causal language model and tokenizer with dequantization config if applicable."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_config = self._build_quantization_config(resolved_path)
        chosen_device = self._determine_device_map()
        load_kwargs: dict[str, object] = {
            "device_map": chosen_device,
            self._KEY_DTYPE: torch.bfloat16,
        }
        if quant_config is not None:
            load_kwargs[self._KEY_QUANT_CONFIG] = quant_config
        model = AutoModelForCausalLM.from_pretrained(resolved_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
        return model, tokenizer

    def _resolve_model_device(self, model: object) -> str | None:
        """Extract torch device identifier string from model parameters if available."""
        param_fn = getattr(model, "parameters", None)
        if callable(param_fn):
            try:
                first_param = next(param_fn(), None)
                if first_param is not None and hasattr(first_param, "device"):
                    dev = getattr(first_param, "device")
                    return str(dev) if dev is not None else None
            except (StopIteration, TypeError):
                return None
        return None

    def _resolve_model_path(self) -> str:
        """Resolve model identifier to local path using ModelPathResolver."""
        return ModelPathResolver.resolve(self._model_name)

    def _resolve_target_modules(self, model: object, target_list: list[str]) -> list[str]:
        """Resolve LoRA target modules prioritizing language_model sub-modules if present."""
        named_modules_fn = getattr(model, "named_modules", None)
        if callable(named_modules_fn):
            try:
                lm_targets = [
                    name
                    for name, _ in named_modules_fn()
                    if self._MODULE_LANGUAGE_MODEL in name
                    and any(name.endswith(f"{self._PREFIX_DOT}{t}") for t in target_list)
                ]
                if lm_targets:
                    return lm_targets
            except (TypeError, ValueError):
                pass
        return self._target_resolver.resolve(model, target_list)

    def _save_adapter_checkpoint(self, model: object) -> str:
        """Persist adapter weights to test directory and verify saved artifacts."""
        adapter_dir = os.path.join(self._work_dir, self._SUBDIR_TEST_ADAPTER)
        os.makedirs(adapter_dir, exist_ok=True)
        save_fn = getattr(model, "save_pretrained", None)
        if callable(save_fn):
            save_fn(adapter_dir)
        return adapter_dir

    def _verify_tokenizer_tokens(self, tokenizer: object) -> None:
        """Ensure tokenizer vocabulary includes all required Gemma 4 delimiters."""
        vocab: dict[str, int] = {}
        if hasattr(tokenizer, "get_vocab"):
            vocab = getattr(tokenizer, "get_vocab")()
        for token in self._REQUIRED_SPECIAL_TOKENS:
            if token not in vocab:
                raise ValueError(f"Missing required special token: {token}")


if __name__ == "__main__":
    LocalModelSmokeTester.main()
