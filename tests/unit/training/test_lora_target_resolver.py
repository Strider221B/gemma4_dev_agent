"""Unit tests for LoRATargetModuleResolver verifying dynamic LoRA target module resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.training.lora_target_resolver import LoRATargetModuleResolver


class TestLoRATargetModuleResolver:
    """Test suite covering LoRATargetModuleResolver scenarios and edge cases."""

    _TARGET_Q: str = "q_proj"
    _TARGET_V: str = "v_proj"
    _TARGET_K: str = "k_proj"
    _SUFFIXED_Q: str = "q_proj.linear"
    _SUFFIXED_V: str = "v_proj.linear"
    _PATH_Q0: str = "model.layers.0.self_attn.q_proj"
    _PATH_V0: str = "model.layers.0.self_attn.v_proj"
    _PATH_Q1: str = "model.layers.1.self_attn.q_proj"
    _PATH_K0: str = "model.layers.0.self_attn.k_proj"
    _CLIPPABLE_CLASS: str = "Gemma4ClippableLinear"
    _LINEAR_CLASS: str = "Linear"
    _CUSTOM_WRAPPER_CLASS: str = "CustomClippableWrapper"
    _EXACT_NAME_MATCH: str = "q_proj"

    def test_resolve_with_clippable_linear_wrappers(self) -> None:
        """Verify wrapped targets are adapted to include .linear suffix."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_linear = MagicMock()
        type(mock_linear).__name__ = self._LINEAR_CLASS

        mock_q = MagicMock()
        type(mock_q).__name__ = self._CLIPPABLE_CLASS
        mock_q.linear = mock_linear

        mock_v = MagicMock()
        type(mock_v).__name__ = self._CLIPPABLE_CLASS
        mock_v.linear = mock_linear

        mock_model.named_modules.return_value = [
            (self._PATH_Q0, mock_q),
            (self._PATH_V0, mock_v),
        ]
        result = resolver.resolve(mock_model, [self._TARGET_Q, self._TARGET_V])
        assert result == [self._SUFFIXED_Q, self._SUFFIXED_V]

    def test_resolve_with_standard_linear_modules(self) -> None:
        """Verify standard nn.Linear targets retain original names without suffix."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_q = MagicMock(spec=["weight", "bias"])
        type(mock_q).__name__ = self._LINEAR_CLASS
        mock_v = MagicMock(spec=["weight", "bias"])
        type(mock_v).__name__ = self._LINEAR_CLASS

        mock_model.named_modules.return_value = [
            (self._PATH_Q0, mock_q),
            (self._PATH_V0, mock_v),
        ]
        result = resolver.resolve(mock_model, [self._TARGET_Q, self._TARGET_V])
        assert result == [self._TARGET_Q, self._TARGET_V]

    def test_resolve_with_already_suffixed_targets(self) -> None:
        """Verify already suffixed targets do not receive duplicate suffixes."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_q = MagicMock()
        type(mock_q).__name__ = self._CLIPPABLE_CLASS
        mock_q.linear = MagicMock()

        mock_model.named_modules.return_value = [(self._PATH_Q0, mock_q)]
        result = resolver.resolve(mock_model, [self._SUFFIXED_Q])
        assert result == [self._SUFFIXED_Q]

    def test_resolve_with_opaque_or_mock_model(self) -> None:
        """Verify fallback returns original targets when model lacks named_modules."""
        resolver = LoRATargetModuleResolver()
        opaque_model = object()
        result = resolver.resolve(opaque_model, [self._TARGET_Q, self._TARGET_V])
        assert result == [self._TARGET_Q, self._TARGET_V]

    def test_resolve_with_empty_target_list(self) -> None:
        """Verify empty target list returns empty list."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_model.named_modules.return_value = []
        result = resolver.resolve(mock_model, [])
        assert result == []

    def test_resolve_deduplicates_targets(self) -> None:
        """Verify duplicate targets across multiple transformer layers are deduplicated."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_q = MagicMock()
        type(mock_q).__name__ = self._CLIPPABLE_CLASS
        mock_q.linear = MagicMock()

        mock_model.named_modules.return_value = [
            (self._PATH_Q0, mock_q),
            (self._PATH_Q1, mock_q),
        ]
        result = resolver.resolve(mock_model, [self._TARGET_Q, self._TARGET_Q])
        assert result == [self._SUFFIXED_Q]

    def test_resolve_with_custom_wrapper_having_weight(self) -> None:
        """Verify wrapper with child holding weight attribute is detected."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        child_layer = MagicMock(spec=["weight"])
        mock_wrapper = MagicMock()
        type(mock_wrapper).__name__ = self._CUSTOM_WRAPPER_CLASS
        mock_wrapper.linear = child_layer

        mock_model.named_modules.return_value = [(self._PATH_Q0, mock_wrapper)]
        result = resolver.resolve(mock_model, [self._TARGET_Q])
        assert result == [self._SUFFIXED_Q]

    def test_resolve_with_mixed_wrapped_and_standard_modules(self) -> None:
        """Verify resolver accurately handles combination of wrapped and unwrapped layers."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_q = MagicMock()
        type(mock_q).__name__ = self._CLIPPABLE_CLASS
        mock_k = MagicMock(spec=["weight", "bias"])
        type(mock_k).__name__ = self._LINEAR_CLASS

        mock_model.named_modules.return_value = [
            (self._PATH_Q0, mock_q),
            (self._PATH_K0, mock_k),
        ]
        result = resolver.resolve(mock_model, [self._TARGET_Q, self._TARGET_K])
        assert result == [self._SUFFIXED_Q, self._TARGET_K]

    def test_resolve_with_named_modules_raising_exception(self) -> None:
        """Verify resolver handles named_modules raising TypeError gracefully."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_model.named_modules.side_effect = TypeError("Mock iteration error")
        result = resolver.resolve(mock_model, [self._TARGET_Q])
        assert result == [self._TARGET_Q]

    def test_resolve_with_malformed_named_modules_items(self) -> None:
        """Verify resolver skips items not matching 2-tuple structure."""
        resolver = LoRATargetModuleResolver()
        mock_model = MagicMock()
        mock_model.named_modules.return_value = ["not_a_tuple", (1, 2, 3)]
        result = resolver.resolve(mock_model, [self._TARGET_Q])
        assert result == [self._TARGET_Q]

    def test_is_clippable_wrapper_with_none_and_missing_attr(self) -> None:
        """Verify helper returns False when module is None or lacks linear child."""
        resolver = LoRATargetModuleResolver()
        assert resolver._is_clippable_wrapper(None) is False
        mock_mod = MagicMock(spec=[])
        assert resolver._is_clippable_wrapper(mock_mod) is False

    def test_matches_target_exact_and_suffix(self) -> None:
        """Verify _matches_target handles both exact names and dotted prefix paths."""
        resolver = LoRATargetModuleResolver()
        assert resolver._matches_target(self._EXACT_NAME_MATCH, self._TARGET_Q) is True
        assert resolver._matches_target(self._PATH_Q0, self._TARGET_Q) is True
        assert resolver._matches_target(self._PATH_K0, self._TARGET_Q) is False

    def test_adapt_target_name_returns_unchanged_when_already_suffixed(self) -> None:
        """Verify _adapt_target_name returns existing string when already suffixed."""
        resolver = LoRATargetModuleResolver()
        assert resolver._adapt_target_name(self._SUFFIXED_Q) == self._SUFFIXED_Q

    def test_is_clippable_wrapper_with_non_linear_child(self) -> None:
        """Verify _is_clippable_wrapper returns False when child is not linear."""
        resolver = LoRATargetModuleResolver()
        mock_mod = MagicMock(spec=["linear"])
        type(mock_mod).__name__ = self._CUSTOM_WRAPPER_CLASS
        mock_mod.linear = "not_a_linear_module"
        assert resolver._is_clippable_wrapper(mock_mod) is False
