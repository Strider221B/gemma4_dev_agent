"""Centralized configuration manager supporting YAML files and env overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from src.config.data_paths_config import DataPathsConfig
from src.config.deploy_config import DeployConfig
from src.config.eval_config import EvalConfig
from src.config.rl_config import RLConfig
from src.config.sft_config import SFTConfig


class ConfigManager:
    """Manager for loading, merging, and providing typed configuration schemas."""

    _ENV_PREFIX: str = "SWEGEMMA_"
    _NESTED_DELIMITER: str = "__"
    _KEY_SFT: str = "sft"
    _KEY_RL: str = "rl"
    _KEY_EVAL: str = "eval"
    _KEY_DEPLOY: str = "deploy"
    _KEY_DATA: str = "data"
    _BOOL_TRUE: str = "true"
    _BOOL_FALSE: str = "false"
    _FILE_ENCODING: str = "utf-8"

    def __init__(self, raw_config: dict[str, object] | None = None) -> None:
        """Initialize ConfigManager with raw config dictionary and apply env overrides."""
        base_config: dict[str, object] = dict(raw_config) if raw_config else {}
        self._raw_config: dict[str, object] = self._apply_env_overrides(base_config)

    @classmethod
    def load(cls, config_path: str) -> ConfigManager:
        """Load YAML configuration from path and instantiate ConfigManager."""
        raw_config = cls._load_yaml(config_path)
        return cls(raw_config=raw_config)

    def get_sft_config(self) -> SFTConfig:
        """Retrieve validated SFTConfig object."""
        sub_dict = self._get_sub_dict(self._KEY_SFT)
        return SFTConfig.model_validate(sub_dict)

    def get_rl_config(self) -> RLConfig:
        """Retrieve validated RLConfig object."""
        sub_dict = self._get_sub_dict(self._KEY_RL)
        return RLConfig.model_validate(sub_dict)

    def get_eval_config(self) -> EvalConfig:
        """Retrieve validated EvalConfig object."""
        sub_dict = self._get_sub_dict(self._KEY_EVAL)
        return EvalConfig.model_validate(sub_dict)

    def get_deploy_config(self) -> DeployConfig:
        """Retrieve validated DeployConfig object."""
        sub_dict = self._get_sub_dict(self._KEY_DEPLOY)
        return DeployConfig.model_validate(sub_dict)

    def get_data_paths(self) -> DataPathsConfig:
        """Retrieve validated DataPathsConfig object."""
        sub_dict = self._get_sub_dict(self._KEY_DATA)
        return DataPathsConfig.model_validate(sub_dict)

    def get_raw_config(self) -> dict[str, object]:
        """Return the raw configuration dictionary including env overrides."""
        return dict(self._raw_config)

    def _get_sub_dict(self, key: str) -> dict[str, object]:
        """Extract sub-dictionary for the key or fallback to root config."""
        val = self._raw_config.get(key)
        if isinstance(val, dict):
            return dict(val)
        return dict(self._raw_config)

    @classmethod
    def _load_yaml(cls, path: str) -> dict[str, object]:
        """Read YAML configuration file from disk."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(file_path, "r", encoding=cls._FILE_ENCODING) as file_handle:
            content = yaml.safe_load(file_handle)
        if not isinstance(content, dict):
            return {}
        return {str(k): v for k, v in content.items()}

    @classmethod
    def _apply_env_overrides(cls, config: dict[str, object]) -> dict[str, object]:
        """Merge environment variable overrides matching the SWEGEMMA_ prefix."""
        merged: dict[str, object] = dict(config)
        for key, value in os.environ.items():
            if key.startswith(cls._ENV_PREFIX):
                stripped = key[len(cls._ENV_PREFIX):].lower()
                cls._set_nested_value(merged, stripped, cls._parse_env_value(value))
        return merged

    @classmethod
    def _set_nested_value(
        cls, target: dict[str, object], key_path: str, value: object
    ) -> None:
        """Set a value in a nested dictionary using double-underscore delimiter."""
        parts = key_path.split(cls._NESTED_DELIMITER)
        curr: dict[str, object] = target
        for part in parts[:-1]:
            nested = curr.setdefault(part, {})
            if isinstance(nested, dict):
                curr = nested
            else:
                new_dict: dict[str, object] = {}
                curr[part] = new_dict
                curr = new_dict
        curr[parts[-1]] = value

    @classmethod
    def _parse_env_value(cls, value: str) -> object:
        """Parse raw string environment variable value into appropriate Python type."""
        lower_val = value.strip().lower()
        if lower_val == cls._BOOL_TRUE:
            return True
        if lower_val == cls._BOOL_FALSE:
            return False
        try:
            return int(value.strip())
        except ValueError:
            pass
        try:
            return float(value.strip())
        except ValueError:
            pass
        return value
