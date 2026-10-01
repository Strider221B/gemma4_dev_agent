"""Unit tests for ConfigManager."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.config_manager import ConfigManager
from src.config.data_paths_config import DataPathsConfig
from src.config.deploy_config import DeployConfig
from src.config.eval_config import EvalConfig
from src.config.rl_config import RLConfig
from src.config.sft_config import SFTConfig


class TestConfigManager:
    """Test suite for ConfigManager configuration loader and accessor."""

    _SAMPLE_YAML_CONTENT: str = (
        "eval_fold: 1\n"
        "output_dir: '/custom/checkpoints'\n"
        "sft:\n"
        "  eval_fold: 2\n"
        "rl:\n"
        "  mode: 'dpo'\n"
        "eval:\n"
        "  num_folds: 5\n"
        "deploy:\n"
        "  output_dir: '/custom/dist'\n"
        "data:\n"
        "  tasks_path: '/custom/tasks.jsonl'\n"
    )
    _ENV_KEY_FOLD: str = "SWEGEMMA_EVAL_FOLD"
    _ENV_VAL_FOLD: str = "3"
    _ENV_KEY_BOOL: str = "SWEGEMMA_TRAINING__BF16"
    _ENV_VAL_BOOL: str = "false"
    _ENV_KEY_FLOAT: str = "SWEGEMMA_TRAINING__LEARNING_RATE"
    _ENV_VAL_FLOAT: str = "1e-5"
    _NONEXISTENT_FILE: str = "/nonexistent/config.yaml"

    def test_load_valid_yaml_returns_config_manager(self, tmp_path: Path) -> None:
        """Verify loading valid YAML file instantiates ConfigManager successfully."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(self._SAMPLE_YAML_CONTENT, encoding="utf-8")
        manager = ConfigManager.load(str(config_file))
        assert isinstance(manager, ConfigManager)

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Verify attempting to load nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load(self._NONEXISTENT_FILE)

    def test_load_non_dict_yaml_returns_empty_manager(self, tmp_path: Path) -> None:
        """Verify non-dictionary YAML content yields an empty configuration."""
        config_file = tmp_path / "scalar.yaml"
        config_file.write_text("just_a_string", encoding="utf-8")
        manager = ConfigManager.load(str(config_file))
        assert manager.get_raw_config() == {}

    def test_get_sft_config_returns_sft_config(self) -> None:
        """Verify get_sft_config returns a valid SFTConfig instance."""
        manager = ConfigManager()
        sft_config = manager.get_sft_config()
        assert isinstance(sft_config, SFTConfig)
        assert sft_config.eval_fold == 0

    def test_get_sft_config_from_nested_key(self, tmp_path: Path) -> None:
        """Verify get_sft_config prioritizes the nested sft section when present."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(self._SAMPLE_YAML_CONTENT, encoding="utf-8")
        manager = ConfigManager.load(str(config_file))
        sft_config = manager.get_sft_config()
        assert sft_config.eval_fold == 2

    def test_get_rl_config_returns_rl_config(self, tmp_path: Path) -> None:
        """Verify get_rl_config extracts the rl section and validates RLConfig."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(self._SAMPLE_YAML_CONTENT, encoding="utf-8")
        manager = ConfigManager.load(str(config_file))
        rl_config = manager.get_rl_config()
        assert isinstance(rl_config, RLConfig)
        assert rl_config.mode == "dpo"

    def test_get_eval_config_returns_eval_config(self, tmp_path: Path) -> None:
        """Verify get_eval_config extracts the eval section and validates EvalConfig."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(self._SAMPLE_YAML_CONTENT, encoding="utf-8")
        manager = ConfigManager.load(str(config_file))
        eval_config = manager.get_eval_config()
        assert isinstance(eval_config, EvalConfig)
        assert eval_config.num_folds == 5

    def test_get_deploy_config_returns_deploy_config(self, tmp_path: Path) -> None:
        """Verify get_deploy_config extracts the deploy section."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(self._SAMPLE_YAML_CONTENT, encoding="utf-8")
        manager = ConfigManager.load(str(config_file))
        deploy_config = manager.get_deploy_config()
        assert isinstance(deploy_config, DeployConfig)
        assert deploy_config.output_dir == "/custom/dist"

    def test_get_data_paths_returns_data_paths_config(self, tmp_path: Path) -> None:
        """Verify get_data_paths extracts data paths configuration."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(self._SAMPLE_YAML_CONTENT, encoding="utf-8")
        manager = ConfigManager.load(str(config_file))
        data_paths = manager.get_data_paths()
        assert isinstance(data_paths, DataPathsConfig)
        assert data_paths.tasks_path == "/custom/tasks.jsonl"

    def test_get_raw_config_returns_dict_copy(self) -> None:
        """Verify get_raw_config returns a copy of raw configuration dictionary."""
        manager = ConfigManager(raw_config={"key": "value"})
        raw = manager.get_raw_config()
        assert raw == {"key": "value"}
        raw["key"] = "modified"
        assert manager.get_raw_config()["key"] == "value"

    def test_env_override_replaces_value(self) -> None:
        """Verify environment variable prefix overrides existing configuration value."""
        with patch.dict(os.environ, {self._ENV_KEY_FOLD: self._ENV_VAL_FOLD}):
            manager = ConfigManager(raw_config={"eval_fold": 0})
            raw = manager.get_raw_config()
            assert raw["eval_fold"] == 3

    def test_env_override_nested_and_booleans(self) -> None:
        """Verify nested double underscore syntax and boolean/float parsing."""
        env_vars = {
            self._ENV_KEY_BOOL: self._ENV_VAL_BOOL,
            self._ENV_KEY_FLOAT: self._ENV_VAL_FLOAT,
            "SWEGEMMA_TRAINING__PACKING": "true",
            "SWEGEMMA_STRING_PARAM": "sample_text",
        }
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager()
            raw = manager.get_raw_config()
            training_dict = raw.get("training")
            assert isinstance(training_dict, dict)
            assert training_dict["bf16"] is False
            assert training_dict["learning_rate"] == 1e-5
            assert training_dict["packing"] is True
            assert raw["string_param"] == "sample_text"
