"""Checkpoint manager for saving, loading, and validating LoRA adapter artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


class CheckpointManager:
    """Manages safetensors adapter checkpoints with size and format verification."""

    _DEFAULT_MAX_SIZE_BYTES: int = 1_500_000_000
    _ADAPTER_CONFIG_FILE: str = "adapter_config.json"
    _ADAPTER_WEIGHTS_FILE: str = "adapter_model.safetensors"
    _FORBIDDEN_PATTERNS: tuple[str, ...] = ("*.bin", "*.pt", "*.pth")

    def __init__(self, max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES) -> None:
        """Initialize CheckpointManager with maximum permitted byte size."""
        self._max_size_bytes: int = max_size_bytes

    def list_checkpoints(self, directory: str) -> list[str]:
        """List all checkpoint subdirectories in the specified root directory."""
        root_path = Path(directory)
        if not root_path.exists() or not root_path.is_dir():
            return []
        return sorted(str(child) for child in root_path.iterdir() if child.is_dir())

    def load_adapter(self, base_model: object, path: str) -> object:
        """Load adapter weights onto the base model using PEFT."""
        from peft import PeftModel

        return PeftModel.from_pretrained(cast(Any, base_model), path)

    def save_adapter(self, model: object, path: str) -> str:
        """Save adapter model weights in safetensors format and validate constraints."""
        dest_path = Path(path)
        dest_path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(dest_path), safe_serialization=True)
        if not self._verify_safetensors(str(dest_path)):
            raise ValueError(f"Safetensors verification failed for checkpoint at {path}")
        if not self.validate_size(str(dest_path)):
            raise ValueError(
                f"Checkpoint size exceeds maximum limit of {self._max_size_bytes} bytes"
            )
        return str(dest_path)

    def validate_size(self, path: str) -> bool:
        """Validate that total checkpoint directory size is strictly under size limit."""
        return self._compute_total_size(path) < self._max_size_bytes

    def _check_forbidden_files(self, path: str) -> list[str]:
        """Scan directory recursively for forbidden legacy weight files."""
        target_path = Path(path)
        if not target_path.exists() or not target_path.is_dir():
            return []
        forbidden: list[str] = []
        for pattern in self._FORBIDDEN_PATTERNS:
            forbidden.extend(
                str(item) for item in target_path.rglob(pattern) if item.is_file()
            )
        return sorted(forbidden)

    def _compute_total_size(self, path: str) -> int:
        """Compute total recursive file size in bytes for the given path."""
        target_path = Path(path)
        if not target_path.exists():
            return 0
        if target_path.is_file():
            return target_path.stat().st_size
        return sum(
            item.stat().st_size for item in target_path.rglob("*") if item.is_file()
        )

    def _verify_safetensors(self, path: str) -> bool:
        """Verify safetensors weight and config files exist and no forbidden files exist."""
        dir_path = Path(path)
        has_weights = (dir_path / self._ADAPTER_WEIGHTS_FILE).is_file()
        has_config = (dir_path / self._ADAPTER_CONFIG_FILE).is_file()
        forbidden = self._check_forbidden_files(path)
        return has_weights and has_config and len(forbidden) == 0
