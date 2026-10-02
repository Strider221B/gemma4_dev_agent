"""Integration test for submission packaging roundtrip."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.config.deploy_config import DeployConfig
from src.deployment.constraint_validator import ConstraintValidator
from src.deployment.submission_packager import SubmissionPackager


class TestPackagingRoundtrip:
    """Integration test suite executing full packaging, extraction, and validation cycle."""

    _MODEL_NAME: str = "gemma-4-31b-it-qat-w4a16-ct"
    _ADAPTER_NAME: str = "coder_lora"
    _CONFIG_NAME: str = "adapter_config.json"
    _WEIGHTS_NAME: str = "adapter_model.safetensors"
    _AGENT_YAML: str = "agent.yaml"
    _PROMPT_PATH: str = "prompts/system.md"
    _ZIP_NAME: str = "submission.zip"
    _EXTRACT_SUBDIR: str = "unzipped_submission"

    def test_full_packaging_roundtrip(self, tmp_path: Path) -> None:
        """Execute packaging from mock inputs, extract archive, and validate constraints."""
        templates_dir = self._create_mock_templates(tmp_path)
        chk_dir = self._create_mock_checkpoint(tmp_path)
        output_dir = tmp_path / "submission_staging"
        zip_path = tmp_path / "dist" / self._ZIP_NAME

        validator = ConstraintValidator()
        packager = SubmissionPackager(validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[{"checkpoint_path": str(chk_dir), "name": self._ADAPTER_NAME}],
        )

        created_zip = packager.package(config)
        assert Path(created_zip).is_file()

        extracted_dir = tmp_path / self._EXTRACT_SUBDIR
        self._extract_archive(created_zip, extracted_dir)

        report = validator.validate(str(extracted_dir))
        assert report.all_passed is True
        self._verify_extracted_files(extracted_dir)

    def _create_mock_templates(self, tmp_path: Path) -> Path:
        templates_dir = tmp_path / "templates"
        prompts_dir = templates_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (templates_dir / self._PROMPT_PATH).write_text(
            "You are an expert autonomous software engineer.",
            encoding="utf-8",
        )
        yaml_content = (
            f"name: root_coder\n"
            f"model: {self._MODEL_NAME}\n"
            f"instruction: !include {self._PROMPT_PATH}\n"
            "generate_content_config:\n"
            "  max_output_tokens: 16384\n"
            "  temperature: 0.2\n"
            "  thinking_config:\n"
            "    thinking_budget: 4096\n"
        )
        (templates_dir / self._AGENT_YAML).write_text(yaml_content, encoding="utf-8")
        return templates_dir

    def _create_mock_checkpoint(self, tmp_path: Path) -> Path:
        chk_dir = tmp_path / "checkpoints" / self._ADAPTER_NAME
        chk_dir.mkdir(parents=True, exist_ok=True)
        (chk_dir / self._CONFIG_NAME).write_text(
            json.dumps({"r": 64, "lora_alpha": 128}),
            encoding="utf-8",
        )
        (chk_dir / self._WEIGHTS_NAME).write_bytes(b"dummy safetensors binary weights")
        return chk_dir

    def _extract_archive(self, zip_path: str, extract_dir: Path) -> None:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    def _verify_extracted_files(self, extracted_dir: Path) -> None:
        assert (extracted_dir / self._AGENT_YAML).is_file()
        assert (extracted_dir / self._PROMPT_PATH).is_file()
        adapter_dest = extracted_dir / "adapters" / self._ADAPTER_NAME
        assert (adapter_dest / self._CONFIG_NAME).is_file()
        assert (adapter_dest / self._WEIGHTS_NAME).is_file()
