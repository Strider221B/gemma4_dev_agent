"""Submission packager for Kaggle deployment."""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import ClassVar

from src.config.deploy_config import DeployConfig
from src.deployment.constraint_validator import ConstraintValidator


class SubmissionPackager:
    """Assembles submission package from templates and adapter checkpoints."""

    _ADAPTERS_DIR_NAME: ClassVar[str] = "adapters"
    _KEY_CHECKPOINT: ClassVar[str] = "checkpoint_path"
    _KEY_NAME: ClassVar[str] = "name"
    _FAILED_VALIDATION_MSG: ClassVar[str] = "Submission pre-flight validation failed:"

    def __init__(self, validator: ConstraintValidator) -> None:
        """Initialize SubmissionPackager with a constraint validator.

        Args:
            validator: ConstraintValidator instance for pre-flight checks.
        """
        self._validator = validator

    def package(self, config: DeployConfig) -> str:
        """Assemble submission directory and zip it.

        Args:
            config: DeployConfig containing paths and adapter definitions.

        Returns:
            Path to the generated zip archive.
        """
        self._clean_output(config.output_dir)
        self._copy_templates(config.templates_dir, config.output_dir)
        self._copy_all_adapters(config)
        self._validate(config.output_dir)
        return self._create_zip(config.output_dir, config.zip_path)

    def _clean_output(self, output_dir: str) -> None:
        if os.path.exists(output_dir):
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            else:
                os.unlink(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def _copy_templates(self, templates_dir: str, output_dir: str) -> None:
        if not os.path.isdir(templates_dir):
            return
        shutil.copytree(templates_dir, output_dir, dirs_exist_ok=True)

    def _copy_all_adapters(self, config: DeployConfig) -> None:
        adapters_base = os.path.join(config.output_dir, self._ADAPTERS_DIR_NAME)
        for adapter in config.adapters:
            chk_path = self._extract_adapter_field(adapter, self._KEY_CHECKPOINT)
            name = self._extract_adapter_field(adapter, self._KEY_NAME)
            if chk_path:
                target_path = (
                    os.path.join(adapters_base, name) if name else adapters_base
                )
                self._copy_adapter(chk_path, target_path)

    def _extract_adapter_field(self, adapter: object, key: str) -> str:
        if isinstance(adapter, dict):
            return str(adapter.get(key, ""))
        return str(getattr(adapter, key, ""))

    def _copy_adapter(self, checkpoint_path: str, target_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            return
        os.makedirs(target_path, exist_ok=True)
        if os.path.isdir(checkpoint_path):
            shutil.copytree(checkpoint_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(checkpoint_path, target_path)

    def _validate(self, output_dir: str) -> None:
        report = self._validator.validate(output_dir)
        if not report.all_passed:
            failed_msgs = [
                f"  - {c.name}: {c.detail}" for c in report.checks if not c.passed
            ]
            detail_str = "\n".join(failed_msgs)
            raise ValueError(f"{self._FAILED_VALIDATION_MSG}\n{detail_str}")

    def _create_zip(self, output_dir: str, zip_path: str) -> str:
        zip_parent = os.path.dirname(zip_path)
        if zip_parent:
            os.makedirs(zip_parent, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for f in sorted(files):
                    full_path = os.path.join(root, f)
                    arcname = os.path.relpath(full_path, output_dir)
                    zf.write(full_path, arcname)
        return zip_path
