"""Compatibility patcher for PEFT LoRA adapters applied to quantized and dequantized models."""

from __future__ import annotations

from typing import Any


class PEFTModelPatcher:
    """Ensures PEFT LoRA modules expose weight/bias attributes and handle accelerate hooks."""

    _ATTR_WEIGHT: str = "weight"
    _ATTR_BIAS: str = "bias"
    _ATTR_BASE_LAYER: str = "base_layer"
    _ATTR_GET_BASE_LAYER: str = "get_base_layer"
    _ATTR_OLD_FORWARD: str = "old_forward"
    _ATTR_HF_HOOK: str = "_hf_hook"
    _ATTR_QUANT_ENABLED: str = "quantization_enabled"
    _NAMED_MODULES_METHOD: str = "named_modules"
    _QUANTIZED_FORWARD_IDENTIFIER: str = "quantized_forward"
    _PEFT_LORA_MODULE: str = "peft.tuners.lora.layer"
    _PEFT_LINEAR_CLASS: str = "Linear"

    def __init__(self) -> None:
        """Initialize PEFTModelPatcher."""

    def patch_model(self, model: object) -> object:
        """Apply compatibility patches to PEFT LoRA class and model module instances."""
        self.patch_peft_lora_class()
        self._patch_module_instances(model)
        return model

    def patch_peft_lora_class(self) -> None:
        """Attach weight and bias delegating properties to PEFT LoRA Linear class."""
        try:
            from peft.tuners.lora.layer import Linear

            if not hasattr(Linear, self._ATTR_WEIGHT):
                setattr(Linear, self._ATTR_WEIGHT, property(self._resolve_base_weight))
            if not hasattr(Linear, self._ATTR_BIAS):
                setattr(Linear, self._ATTR_BIAS, property(self._resolve_base_bias))
        except (ImportError, AttributeError):
            pass

    def _patch_module_instances(self, model: object) -> None:
        """Traverse named modules and fix accelerate hook forwarding on LoRA layers."""
        named_modules_fn = getattr(model, self._NAMED_MODULES_METHOD, None)
        if not callable(named_modules_fn):
            return
        for _, module in named_modules_fn():
            self._fix_single_module(module)

    def _fix_single_module(self, module: object) -> None:
        """Ensure LoRA layer module hooks do not bypass LoRA forward logic."""
        if hasattr(module, self._ATTR_BASE_LAYER):
            self._align_lora_forward(module)

    def _align_lora_forward(self, module: object) -> None:
        """Align module _old_forward to class forward if pointed at quantized_forward."""
        old_fwd = getattr(module, f"_{self._ATTR_OLD_FORWARD}", None)
        if old_fwd is not None and self._is_quantized_forward(old_fwd):
            cls_fwd = getattr(module.__class__, "forward", None)
            if cls_fwd is not None:
                setattr(module, f"_{self._ATTR_OLD_FORWARD}", cls_fwd)

    def _is_quantized_forward(self, forward_callable: Any) -> bool:
        """Check whether a callable represents compressed-tensors quantized_forward."""
        qual_name = getattr(forward_callable, "__qualname__", "")
        name = getattr(forward_callable, "__name__", "")
        return (
            self._QUANTIZED_FORWARD_IDENTIFIER in qual_name
            or self._QUANTIZED_FORWARD_IDENTIFIER in name
        )

    def _resolve_base_weight(self, lora_layer: Any) -> Any:
        """Retrieve weight parameter from inner base layer."""
        base = self._get_base_layer(lora_layer)
        return getattr(base, self._ATTR_WEIGHT, None)

    def _resolve_base_bias(self, lora_layer: Any) -> Any:
        """Retrieve bias parameter from inner base layer."""
        base = self._get_base_layer(lora_layer)
        return getattr(base, self._ATTR_BIAS, None)

    def _get_base_layer(self, lora_layer: Any) -> Any:
        """Safely extract base layer from LoRA layer module."""
        getter = getattr(lora_layer, self._ATTR_GET_BASE_LAYER, None)
        if callable(getter):
            return getter()
        return getattr(lora_layer, self._ATTR_BASE_LAYER, None)
