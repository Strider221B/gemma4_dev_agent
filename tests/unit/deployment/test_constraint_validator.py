"""Unit tests for ConstraintValidator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.deployment.constraint_validator import ConstraintValidator


class TestConstraintValidator:
    """Test suite for ConstraintValidator pre-flight checks."""

    _MODEL_NAME: str = "gemma-4-31b-it-qat-w4a16-ct"
    _OTHER_MODEL: str = "other-model-7b"
    _ADAPTER_DIR_NAME: str = "adapters"
    _ADAPTER_SUBDIR: str = "coder_lora"
    _CONFIG_NAME: str = "adapter_config.json"
    _WEIGHTS_NAME: str = "adapter_model.safetensors"
    _ROOT_YAML: str = "agent.yaml"
    _FORBIDDEN_FILE: str = "model.bin"
    _SYMLINK_FILE: str = "link_to_agent.yaml"
    _OVERSIZED_BYTES: int = 4_000_000_000

    @pytest.fixture
    def staging_dir(self, tmp_path: Path) -> Path:
        """Create a valid staging directory fixture."""
        return self._create_valid_staging(tmp_path)

    def test_validate_valid_staging_dir_passes(self, staging_dir: Path) -> None:
        """Verify that a compliant staging directory passes all checks."""
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        assert report.all_passed is True
        assert len(report.checks) == 10
        assert all(c.passed for c in report.checks)

    def test_validate_missing_agent_yaml_fails(self, staging_dir: Path) -> None:
        """Verify that missing agent.yaml causes root_config check to fail."""
        (staging_dir / self._ROOT_YAML).unlink()
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        assert report.all_passed is False
        check = next(c for c in report.checks if c.name == "root_config")
        assert check.passed is False

    def test_validate_oversized_submission_fails(self, staging_dir: Path) -> None:
        """Verify that size exceeding maximum limit causes check failure."""
        validator = ConstraintValidator()
        with patch.object(validator, "_compute_total_size", return_value=self._OVERSIZED_BYTES):
            report = validator.validate(str(staging_dir))
            assert report.all_passed is False
            check = next(c for c in report.checks if c.name == "total_size")
            assert check.passed is False

    def test_validate_forbidden_extension_fails(self, staging_dir: Path) -> None:
        """Verify that forbidden extensions (.bin, .pt) cause check failure."""
        (staging_dir / self._FORBIDDEN_FILE).write_bytes(b"dummy")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        assert report.all_passed is False
        check = next(c for c in report.checks if c.name == "extensions")
        assert check.passed is False

    def test_validate_symlink_detected(self, staging_dir: Path) -> None:
        """Verify that symlinks inside staging directory are detected and rejected."""
        target = staging_dir / self._ROOT_YAML
        link = staging_dir / self._SYMLINK_FILE
        os.symlink(target, link)
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        assert report.all_passed is False
        check = next(c for c in report.checks if c.name == "no_symlinks")
        assert check.passed is False

    def test_validate_too_many_adapters_fails(self, staging_dir: Path) -> None:
        """Verify that having more than 8 adapter directories causes failure."""
        adapters_root = staging_dir / self._ADAPTER_DIR_NAME
        for idx in range(9):
            d = adapters_root / f"adapter_{idx}"
            d.mkdir(parents=True, exist_ok=True)
            (d / self._CONFIG_NAME).write_text(json.dumps({"r": 32}), encoding="utf-8")
            (d / self._WEIGHTS_NAME).write_bytes(b"dummy")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        assert report.all_passed is False
        check = next(c for c in report.checks if c.name == "adapter_count")
        assert check.passed is False

    def test_validate_lora_rank_exceeded_fails(self, staging_dir: Path) -> None:
        """Verify that LoRA rank exceeding 128 causes check failure."""
        cfg_path = staging_dir / self._ADAPTER_DIR_NAME / self._ADAPTER_SUBDIR / self._CONFIG_NAME
        cfg_path.write_text(json.dumps({"r": 256}), encoding="utf-8")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        assert report.all_passed is False
        check = next(c for c in report.checks if c.name == "lora_rank")
        assert check.passed is False

    def test_validate_single_model_passes(self, staging_dir: Path) -> None:
        """Verify single declared model passes check."""
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "single_model")
        assert check.passed is True
        assert self._MODEL_NAME in check.detail

    def test_validate_multiple_models_fails(self, staging_dir: Path) -> None:
        """Verify multiple conflicting models declared causes failure."""
        subagent_yaml = staging_dir / "subagent.yaml"
        subagent_yaml.write_text(f"model: {self._OTHER_MODEL}\n", encoding="utf-8")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        assert report.all_passed is False
        check = next(c for c in report.checks if c.name == "single_model")
        assert check.passed is False

    def test_validate_token_limits_exceeded_fails(self, staging_dir: Path) -> None:
        """Verify max output tokens exceeding allowed limit fails."""
        bad_yaml = (
            f"model: {self._MODEL_NAME}\n"
            "generate_content_config:\n"
            "  max_output_tokens: 65536\n"
        )
        (staging_dir / self._ROOT_YAML).write_text(bad_yaml, encoding="utf-8")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "token_limits")
        assert check.passed is False

    def test_validate_adapter_missing_weights_fails(self, staging_dir: Path) -> None:
        """Verify adapter missing safetensors file fails."""
        weights = staging_dir / self._ADAPTER_DIR_NAME / self._ADAPTER_SUBDIR / self._WEIGHTS_NAME
        weights.unlink()
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "adapters")
        assert check.passed is False

    def test_validate_yaml_syntax_error_fails(self, staging_dir: Path) -> None:
        """Verify malformed YAML causes yaml_schema check to fail."""
        (staging_dir / "bad.yaml").write_text("key: [unclosed list", encoding="utf-8")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "yaml_schema")
        assert check.passed is False

    def test_validate_yaml_path_traversal_fails(self, staging_dir: Path) -> None:
        """Verify YAML include with path traversal fails."""
        (staging_dir / "traversal.yaml").write_text(
            "instruction: !include ../../secret.txt\n",
            encoding="utf-8",
        )
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "yaml_schema")
        assert check.passed is False

    def test_validate_no_adapters_passes(self, staging_dir: Path) -> None:
        """Verify staging directory with no adapters directory passes."""
        import shutil

        shutil.rmtree(staging_dir / self._ADAPTER_DIR_NAME)
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "adapters")
        assert check.passed is True

    def test_validate_thinking_budget_exceeded_fails(self, staging_dir: Path) -> None:
        """Verify thinking budget exceeding limit fails check."""
        bad_yaml = (
            f"model: {self._MODEL_NAME}\n"
            "generate_content_config:\n"
            "  max_output_tokens: 16384\n"
            "  thinking_config:\n"
            "    thinking_budget: 32768\n"
        )
        (staging_dir / self._ROOT_YAML).write_text(bad_yaml, encoding="utf-8")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "token_limits")
        assert check.passed is False

    def test_validate_no_models_declared_fails(self, staging_dir: Path) -> None:
        """Verify absence of model field in YAML causes failure."""
        (staging_dir / self._ROOT_YAML).write_text("name: test\n", encoding="utf-8")
        validator = ConstraintValidator()
        report = validator.validate(str(staging_dir))
        check = next(c for c in report.checks if c.name == "single_model")
        assert check.passed is False

    def _create_valid_staging(self, tmp_path: Path) -> Path:
        staging = tmp_path / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        agent_content = (
            f"name: root_coder\n"
            f"model: {self._MODEL_NAME}\n"
            f"generate_content_config:\n"
            f"  max_output_tokens: 16384\n"
            f"  thinking_config:\n"
            f"    thinking_budget: 4096\n"
        )
        (staging / self._ROOT_YAML).write_text(agent_content, encoding="utf-8")
        adapter_dir = staging / self._ADAPTER_DIR_NAME / self._ADAPTER_SUBDIR
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / self._CONFIG_NAME).write_text(
            json.dumps({"r": 64, "lora_alpha": 128}), encoding="utf-8"
        )
        (adapter_dir / self._WEIGHTS_NAME).write_bytes(b"dummy safetensors binary")
        return staging
