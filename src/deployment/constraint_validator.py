"""Submission pre-flight constraint validator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar

import yaml

from src.deployment.validation_check import ValidationCheck
from src.deployment.validation_report import ValidationReport


class ConstraintValidator:
    """Validates submission staging directories against competition constraints."""

    _MAX_TOTAL_SIZE: ClassVar[int] = 3_221_225_472
    _REQUIRED_FILES: ClassVar[tuple[str, ...]] = ("agent.yaml",)
    _ALLOWED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
    )
    _FORBIDDEN_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {".bin", ".pt", ".pth", ".pkl", ".pickle"}
    )
    _MAX_ADAPTERS: ClassVar[int] = 8
    _MAX_LORA_RANK: ClassVar[int] = 128
    _EXPECTED_MODEL_NAME: ClassVar[str] = "gemma-4-31b-it-qat-w4a16-ct"
    _MAX_OUTPUT_TOKENS: ClassVar[int] = 32768
    _MAX_THINKING_BUDGET: ClassVar[int] = 16384
    _ROOT_CONFIG_NAME: ClassVar[str] = "agent.yaml"
    _ADAPTERS_DIR_NAME: ClassVar[str] = "adapters"
    _CONFIG_FILE_NAME: ClassVar[str] = "adapter_config.json"
    _SAFETENSORS_EXT: ClassVar[str] = ".safetensors"
    _YAML_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".yaml", ".yml"})
    _MODEL_KEY: ClassVar[str] = "model"
    _GEN_CONFIG_KEY: ClassVar[str] = "generate_content_config"
    _MAX_TOKENS_KEY: ClassVar[str] = "max_output_tokens"
    _THINKING_CONFIG_KEY: ClassVar[str] = "thinking_config"
    _THINKING_BUDGET_KEY: ClassVar[str] = "thinking_budget"
    _RANK_KEY: ClassVar[str] = "r"
    _LORA_RANK_KEY: ClassVar[str] = "lora_rank"
    _TRAVERSAL_TOKEN: ClassVar[str] = ".."
    _INCLUDE_TAG: ClassVar[str] = "!include"

    def __init__(self) -> None:
        """Initialize ConstraintValidator and register YAML constructors."""
        yaml.SafeLoader.add_constructor(
            self._INCLUDE_TAG,
            lambda loader, node: (
                str(loader.construct_scalar(node))
                if isinstance(node, yaml.ScalarNode)
                else str(getattr(node, "value", ""))
            ),
        )

    def validate(self, staging_dir: str) -> ValidationReport:
        """Run all pre-flight constraint validation checks on a staging directory.

        Args:
            staging_dir: Path to directory containing submission files.

        Returns:
            ValidationReport containing check outcomes and overall status.
        """
        checks = [
            self._check_root_config(staging_dir),
            self._check_total_size(staging_dir),
            self._check_extensions(staging_dir),
            self._check_adapters(staging_dir),
            self._check_yaml_schema(staging_dir),
            self._check_single_model(staging_dir),
            self._check_no_symlinks(staging_dir),
            self._check_adapter_count(staging_dir),
            self._check_lora_rank(staging_dir),
            self._check_token_limits(staging_dir),
        ]
        all_passed = all(c.passed for c in checks)
        return ValidationReport(checks=checks, all_passed=all_passed)

    def _check_root_config(self, staging_dir: str) -> ValidationCheck:
        root_path = Path(staging_dir) / self._ROOT_CONFIG_NAME
        if not root_path.is_file():
            return ValidationCheck(
                name="root_config",
                passed=False,
                detail=f"Missing required file: {self._ROOT_CONFIG_NAME}",
            )
        return ValidationCheck(
            name="root_config",
            passed=True,
            detail=f"Found {self._ROOT_CONFIG_NAME} in root directory",
        )

    def _check_total_size(self, staging_dir: str) -> ValidationCheck:
        total_size = self._compute_total_size(staging_dir)
        passed = total_size <= self._MAX_TOTAL_SIZE
        detail = (
            f"Total size {total_size:,} bytes "
            f"(limit: {self._MAX_TOTAL_SIZE:,} bytes)"
        )
        return ValidationCheck(name="total_size", passed=passed, detail=detail)

    def _check_extensions(self, staging_dir: str) -> ValidationCheck:
        forbidden: list[str] = []
        for root, _, files in os.walk(staging_dir):
            for file_name in files:
                suffix = Path(file_name).suffix.lower()
                if suffix in self._FORBIDDEN_EXTENSIONS:
                    forbidden.append(os.path.join(root, file_name))
        if forbidden:
            return ValidationCheck(
                name="extensions",
                passed=False,
                detail=f"Found forbidden files: {', '.join(forbidden)}",
            )
        return ValidationCheck(
            name="extensions",
            passed=True,
            detail="No forbidden extensions found",
        )

    def _check_adapters(self, staging_dir: str) -> ValidationCheck:
        adapters = self._find_adapters(staging_dir)
        if not adapters:
            return ValidationCheck(
                name="adapters",
                passed=True,
                detail="No adapters found (base model only)",
            )
        for adapter_dir in adapters:
            config_path = Path(adapter_dir) / self._CONFIG_FILE_NAME
            has_safetensors = any(
                f.suffix.lower() == self._SAFETENSORS_EXT
                for f in Path(adapter_dir).iterdir()
                if f.is_file()
            )
            if not config_path.is_file() or not has_safetensors:
                return ValidationCheck(
                    name="adapters",
                    passed=False,
                    detail=f"Adapter {adapter_dir} missing config or safetensors",
                )
        return ValidationCheck(
            name="adapters",
            passed=True,
            detail=f"All {len(adapters)} adapters valid",
        )

    def _check_yaml_schema(self, staging_dir: str) -> ValidationCheck:
        yaml_files: list[Path] = []
        for root, _, files in os.walk(staging_dir):
            for f in files:
                if Path(f).suffix.lower() in self._YAML_EXTENSIONS:
                    yaml_files.append(Path(root) / f)
        for yf in yaml_files:
            try:
                content = yf.read_text(encoding="utf-8")
                if self._TRAVERSAL_TOKEN in content and self._INCLUDE_TAG in content:
                    return ValidationCheck(
                        name="yaml_schema",
                        passed=False,
                        detail=f"Path traversal detected in {yf.name}",
                    )
                yaml.safe_load(content)
            except Exception as exc:
                return ValidationCheck(
                    name="yaml_schema",
                    passed=False,
                    detail=f"Failed parsing {yf.name}: {exc}",
                )
        return ValidationCheck(
            name="yaml_schema",
            passed=True,
            detail="All YAML files valid and safe",
        )

    def _check_single_model(self, staging_dir: str) -> ValidationCheck:
        models: set[str] = set()
        for root, _, files in os.walk(staging_dir):
            for f in files:
                if Path(f).suffix.lower() in self._YAML_EXTENSIONS:
                    self._extract_models_from_yaml(os.path.join(root, f), models)
        if len(models) == 0:
            return ValidationCheck(
                name="single_model",
                passed=False,
                detail="No model declared in configuration",
            )
        if len(models) > 1:
            return ValidationCheck(
                name="single_model",
                passed=False,
                detail=f"Multiple models declared: {sorted(models)}",
            )
        return ValidationCheck(
            name="single_model",
            passed=True,
            detail=f"Single model declared: {next(iter(models))}",
        )

    def _extract_models_from_yaml(self, path: str, models: set[str]) -> None:
        try:
            content = Path(path).read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            if isinstance(data, dict) and self._MODEL_KEY in data:
                models.add(str(data[self._MODEL_KEY]))
        except Exception:
            pass

    def _check_no_symlinks(self, staging_dir: str) -> ValidationCheck:
        symlinks: list[str] = []
        for root, dirs, files in os.walk(staging_dir):
            for d in dirs:
                full_path = os.path.join(root, d)
                if os.path.islink(full_path):
                    symlinks.append(full_path)
            for f in files:
                full_path = os.path.join(root, f)
                if os.path.islink(full_path):
                    symlinks.append(full_path)
        if symlinks:
            return ValidationCheck(
                name="no_symlinks",
                passed=False,
                detail=f"Symlinks detected: {', '.join(symlinks)}",
            )
        return ValidationCheck(
            name="no_symlinks",
            passed=True,
            detail="No symlinks detected",
        )

    def _check_adapter_count(self, staging_dir: str) -> ValidationCheck:
        adapters = self._find_adapters(staging_dir)
        count = len(adapters)
        passed = count <= self._MAX_ADAPTERS
        detail = (
            f"Adapter count {count} <= {self._MAX_ADAPTERS}"
            if passed
            else f"Adapter count {count} exceeds limit {self._MAX_ADAPTERS}"
        )
        return ValidationCheck(name="adapter_count", passed=passed, detail=detail)

    def _check_lora_rank(self, staging_dir: str) -> ValidationCheck:
        adapters = self._find_adapters(staging_dir)
        for adapter_dir in adapters:
            cfg_path = os.path.join(adapter_dir, self._CONFIG_FILE_NAME)
            cfg = self._parse_adapter_config(cfg_path)
            rank = cfg.get(self._RANK_KEY) or cfg.get(self._LORA_RANK_KEY)
            if isinstance(rank, int) and rank > self._MAX_LORA_RANK:
                return ValidationCheck(
                    name="lora_rank",
                    passed=False,
                    detail=f"Adapter {adapter_dir} rank {rank} exceeds limit {self._MAX_LORA_RANK}",
                )
        return ValidationCheck(
            name="lora_rank",
            passed=True,
            detail=f"All adapter ranks <= {self._MAX_LORA_RANK}",
        )

    def _check_token_limits(self, staging_dir: str) -> ValidationCheck:
        root_yaml = Path(staging_dir) / self._ROOT_CONFIG_NAME
        if not root_yaml.is_file():
            return ValidationCheck(
                name="token_limits",
                passed=False,
                detail=f"Missing {self._ROOT_CONFIG_NAME} for token limits check",
            )
        try:
            data = yaml.safe_load(root_yaml.read_text(encoding="utf-8"))
        except Exception as exc:
            return ValidationCheck(
                name="token_limits",
                passed=False,
                detail=f"Error reading {self._ROOT_CONFIG_NAME}: {exc}",
            )
        if not isinstance(data, dict) or self._GEN_CONFIG_KEY not in data:
            return ValidationCheck(
                name="token_limits",
                passed=True,
                detail="No generate_content_config found",
            )
        return self._validate_token_values(data[self._GEN_CONFIG_KEY])

    def _validate_token_values(self, gen_cfg: object) -> ValidationCheck:
        if not isinstance(gen_cfg, dict):
            return ValidationCheck(
                name="token_limits",
                passed=False,
                detail="generate_content_config is not a dictionary",
            )
        max_tokens = gen_cfg.get(self._MAX_TOKENS_KEY)
        if isinstance(max_tokens, int) and (
            max_tokens <= 0 or max_tokens > self._MAX_OUTPUT_TOKENS
        ):
            return ValidationCheck(
                name="token_limits",
                passed=False,
                detail=f"max_output_tokens {max_tokens} out of range",
            )
        thinking_cfg = gen_cfg.get(self._THINKING_CONFIG_KEY)
        if isinstance(thinking_cfg, dict):
            budget = thinking_cfg.get(self._THINKING_BUDGET_KEY)
            if isinstance(budget, int) and (budget < 0 or budget > self._MAX_THINKING_BUDGET):
                return ValidationCheck(
                    name="token_limits",
                    passed=False,
                    detail=f"thinking_budget {budget} out of range",
                )
        return ValidationCheck(
            name="token_limits",
            passed=True,
            detail="Token limits are valid",
        )

    def _compute_total_size(self, staging_dir: str) -> int:
        total = 0
        for root, _, files in os.walk(staging_dir):
            for f in files:
                file_path = os.path.join(root, f)
                if not os.path.islink(file_path) and os.path.isfile(file_path):
                    total += os.path.getsize(file_path)
        return total

    def _find_adapters(self, staging_dir: str) -> list[str]:
        adapters_dir = Path(staging_dir) / self._ADAPTERS_DIR_NAME
        if adapters_dir.is_dir():
            return sorted(
                str(p)
                for p in adapters_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        return []

    def _parse_adapter_config(self, path: str) -> dict[str, object]:
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
