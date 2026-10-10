"""Unit tests for PEFTModelPatcher verifying compatibility with quantized layers."""

from __future__ import annotations

import sys
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from src.training.peft_model_patcher import PEFTModelPatcher


class TestPEFTModelPatcher:
    """Tests for PEFTModelPatcher class covering class and instance patches."""

    _SAMPLE_WEIGHT: ClassVar[str] = "dummy_weight_tensor"
    _SAMPLE_BIAS: ClassVar[str] = "dummy_bias_tensor"
    _KEY_PEFT_LAYER: ClassVar[str] = "peft.tuners.lora.layer"
    _KEY_ACCELERATE_HOOKS: ClassVar[str] = "accelerate.hooks"
    _QUANTIZED_FORWARD_NAME: ClassVar[str] = "quantized_forward"
    _STANDARD_FORWARD_NAME: ClassVar[str] = "standard_forward"

    def test_instantiation_succeeds(self) -> None:
        """Verify PEFTModelPatcher can be instantiated."""
        patcher = PEFTModelPatcher()
        assert isinstance(patcher, PEFTModelPatcher)

    def test_patch_peft_lora_class_delegates_weight_and_bias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify weight and bias properties resolve from base_layer."""
        mock_linear_cls = self._create_dummy_lora_class()
        mock_module = MagicMock()
        mock_module.Linear = mock_linear_cls
        monkeypatch.setitem(sys.modules, self._KEY_PEFT_LAYER, mock_module)

        patcher = PEFTModelPatcher()
        patcher.patch_peft_lora_class()

        instance = mock_linear_cls()
        instance.base_layer = MagicMock()
        instance.base_layer.weight = self._SAMPLE_WEIGHT
        instance.base_layer.bias = self._SAMPLE_BIAS

        assert instance.weight == self._SAMPLE_WEIGHT
        assert instance.bias == self._SAMPLE_BIAS

    def test_patch_peft_lora_class_with_get_base_layer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify delegation uses get_base_layer method when available."""
        mock_linear_cls = self._create_dummy_lora_class()
        mock_module = MagicMock()
        mock_module.Linear = mock_linear_cls
        monkeypatch.setitem(sys.modules, self._KEY_PEFT_LAYER, mock_module)

        patcher = PEFTModelPatcher()
        patcher.patch_peft_lora_class()

        base_mock = MagicMock()
        base_mock.weight = self._SAMPLE_WEIGHT
        base_mock.bias = self._SAMPLE_BIAS
        instance = mock_linear_cls()
        instance.get_base_layer = MagicMock(return_value=base_mock)

        assert instance.weight == self._SAMPLE_WEIGHT
        assert instance.bias == self._SAMPLE_BIAS

    def test_patch_peft_lora_class_handles_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify patch_peft_lora_class handles missing peft gracefully."""
        monkeypatch.setitem(sys.modules, self._KEY_PEFT_LAYER, None)
        patcher = PEFTModelPatcher()
        patcher.patch_peft_lora_class()

    def test_patch_model_aligns_quantized_forward_hooks(self) -> None:
        """Verify patch_model replaces quantized_forward on _old_forward with class forward."""
        patcher = PEFTModelPatcher()
        lora_instance = self._build_lora_instance_with_quantized_hook()
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("layer0", lora_instance)]

        result = patcher.patch_model(mock_model)
        assert result is mock_model
        assert lora_instance._old_forward == lora_instance.__class__.forward

    def test_patch_model_ignores_non_quantized_hooks(self) -> None:
        """Verify patch_model does not alter standard non-quantized _old_forward."""
        patcher = PEFTModelPatcher()
        lora_instance = MagicMock()
        lora_instance.base_layer = MagicMock()

        def custom_fwd(x: object) -> object:
            return x

        custom_fwd.__name__ = self._STANDARD_FORWARD_NAME
        lora_instance._old_forward = custom_fwd

        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("layer0", lora_instance)]

        patcher.patch_model(mock_model)
        assert lora_instance._old_forward == custom_fwd

    def test_patch_model_ignores_models_without_named_modules(self) -> None:
        """Verify patch_model gracefully handles objects without named_modules callable."""
        patcher = PEFTModelPatcher()
        result = patcher.patch_model(object())
        assert result is not None

    def _create_dummy_lora_class(self) -> type:
        """Create a mock LoRA Linear class for property attachment testing."""

        class DummyLoRALinear:
            pass

        return DummyLoRALinear

    def _build_lora_instance_with_quantized_hook(self) -> MagicMock:
        """Build mock LoRA module instance with quantized_forward hook."""

        def mock_quantized_forward(self_mod: object, *args: object) -> object:
            return args

        mock_quantized_forward.__name__ = self._QUANTIZED_FORWARD_NAME
        mock_quantized_forward.__qualname__ = (
            f"compressed_tensors.{self._QUANTIZED_FORWARD_NAME}"
        )

        class DummyModule:
            def forward(self, x: object) -> object:
                return x

        instance = DummyModule()
        setattr(instance, "base_layer", MagicMock())
        setattr(instance, "_old_forward", mock_quantized_forward)
        return instance
