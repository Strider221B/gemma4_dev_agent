"""Dynamic target module resolver for LoRA adapters supporting Gemma4 clippable linear layers."""

from __future__ import annotations

from typing import Any


class LoRATargetModuleResolver:
    """Resolves LoRA target module names adapting for clippable linear wrappers."""

    _CLIPPABLE_CLASS_NAME: str = "Gemma4ClippableLinear"
    _LINEAR_ATTR: str = "linear"
    _DOT_LINEAR: str = ".linear"
    _LINEAR_TYPE_KEYWORD: str = "Linear"
    _DOT: str = "."
    _NAMED_MODULES_ATTR: str = "named_modules"
    _WEIGHT_ATTR: str = "weight"

    def __init__(self) -> None:
        """Initialize LoRATargetModuleResolver."""

    def resolve(self, model: object, target_modules: list[str]) -> list[str]:
        """Resolve target module names, adapting clippable linear wrappers if present."""
        resolved: list[str] = []
        for target in target_modules:
            adapted = (
                self._adapt_target_name(target)
                if self._needs_linear_suffix(model, target)
                else target
            )
            if adapted not in resolved:
                resolved.append(adapted)
        return resolved

    def _adapt_target_name(self, target: str) -> str:
        """Append linear suffix to target name if not already suffixed."""
        if target.endswith(self._DOT_LINEAR):
            return target
        return f"{target}{self._DOT_LINEAR}"

    def _is_clippable_wrapper(self, module: object) -> bool:
        """Check if a module is a Gemma4 clippable linear wrapper."""
        if module is None:
            return False
        if type(module).__name__ == self._CLIPPABLE_CLASS_NAME:
            return True
        if hasattr(module, self._LINEAR_ATTR):
            child: Any = getattr(module, self._LINEAR_ATTR)
            if child is not None and (
                self._LINEAR_TYPE_KEYWORD in type(child).__name__
                or hasattr(child, self._WEIGHT_ATTR)
            ):
                return True
        return False

    def _matches_target(self, name: str, target: str) -> bool:
        """Check if module name matches target identifier."""
        return name == target or name.endswith(f"{self._DOT}{target}")

    def _needs_linear_suffix(self, model: object, target: str) -> bool:
        """Determine whether target module requires .linear suffix."""
        if target.endswith(self._DOT_LINEAR):
            return False
        named_modules_fn = getattr(model, self._NAMED_MODULES_ATTR, None)
        if not callable(named_modules_fn):
            return False
        try:
            modules = named_modules_fn()
            for item in modules:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    continue
                name, module = item
                if self._matches_target(name, target) and self._is_clippable_wrapper(module):
                    return True
        except (TypeError, ValueError):
            return False
        return False
