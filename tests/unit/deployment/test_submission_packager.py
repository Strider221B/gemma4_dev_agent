"""Unit tests for SubmissionPackager."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config.deploy_config import DeployConfig
from src.deployment.constraint_validator import ConstraintValidator
from src.deployment.submission_packager import SubmissionPackager
from src.deployment.validation_check import ValidationCheck
from src.deployment.validation_report import ValidationReport


class TestSubmissionPackager:
    """Test suite for SubmissionPackager pipeline orchestration."""

    _AGENT_YAML: str = "agent.yaml"
    _ADAPTER_DIR_NAME: str = "adapters"
    _ADAPTER_NAME: str = "coder_lora"
    _CONFIG_NAME: str = "adapter_config.json"
    _WEIGHTS_NAME: str = "adapter_model.safetensors"
    _SUBMISSION_ZIP: str = "submission.zip"
    _OLD_FILE: str = "stale_file.txt"

    @pytest.fixture
    def mock_validator(self) -> MagicMock:
        """Create a mock ConstraintValidator that passes validation by default."""
        validator = MagicMock(spec=ConstraintValidator)
        validator.validate.return_value = ValidationReport(
            checks=[ValidationCheck(name="dummy", passed=True, detail="ok")],
            all_passed=True,
        )
        return validator

    @pytest.fixture
    def packaging_setup(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        """Setup directories for templates, checkpoints, output, and zip destination."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        (templates_dir / self._AGENT_YAML).write_text("name: root\n", encoding="utf-8")

        chk_dir = tmp_path / "checkpoints" / self._ADAPTER_NAME
        chk_dir.mkdir(parents=True, exist_ok=True)
        (chk_dir / self._CONFIG_NAME).write_text(json.dumps({"r": 64}), encoding="utf-8")
        (chk_dir / self._WEIGHTS_NAME).write_bytes(b"dummy safetensors")

        output_dir = tmp_path / "submission"
        zip_path = tmp_path / "dist" / self._SUBMISSION_ZIP
        return templates_dir, chk_dir, output_dir, zip_path

    def test_package_creates_zip(
        self,
        mock_validator: MagicMock,
        packaging_setup: tuple[Path, Path, Path, Path],
    ) -> None:
        """Verify package creates a valid zip archive file."""
        templates_dir, chk_dir, output_dir, zip_path = packaging_setup
        packager = SubmissionPackager(mock_validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[{"checkpoint_path": str(chk_dir), "name": self._ADAPTER_NAME}],
        )
        result = packager.package(config)
        assert result == str(zip_path)
        assert Path(result).is_file()
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert self._AGENT_YAML in names

    def test_package_copies_templates(
        self,
        mock_validator: MagicMock,
        packaging_setup: tuple[Path, Path, Path, Path],
    ) -> None:
        """Verify package copies agent.yaml and other template files to output."""
        templates_dir, chk_dir, output_dir, zip_path = packaging_setup
        packager = SubmissionPackager(mock_validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[],
        )
        packager.package(config)
        copied_agent = output_dir / self._AGENT_YAML
        assert copied_agent.is_file()
        assert "name: root" in copied_agent.read_text(encoding="utf-8")

    def test_package_copies_adapters(
        self,
        mock_validator: MagicMock,
        packaging_setup: tuple[Path, Path, Path, Path],
    ) -> None:
        """Verify package copies adapter files to output adapters directory."""
        templates_dir, chk_dir, output_dir, zip_path = packaging_setup
        packager = SubmissionPackager(mock_validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[{"checkpoint_path": str(chk_dir), "name": self._ADAPTER_NAME}],
        )
        packager.package(config)
        target_adapter_dir = output_dir / self._ADAPTER_DIR_NAME / self._ADAPTER_NAME
        assert (target_adapter_dir / self._CONFIG_NAME).is_file()
        assert (target_adapter_dir / self._WEIGHTS_NAME).is_file()

    def test_package_validates_before_zip(
        self,
        mock_validator: MagicMock,
        packaging_setup: tuple[Path, Path, Path, Path],
    ) -> None:
        """Verify validator is invoked with the output directory path."""
        templates_dir, chk_dir, output_dir, zip_path = packaging_setup
        packager = SubmissionPackager(mock_validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[],
        )
        packager.package(config)
        mock_validator.validate.assert_called_once_with(str(output_dir))

    def test_package_cleans_output_first(
        self,
        mock_validator: MagicMock,
        packaging_setup: tuple[Path, Path, Path, Path],
    ) -> None:
        """Verify preexisting files in output directory are cleaned prior to assembly."""
        templates_dir, chk_dir, output_dir, zip_path = packaging_setup
        output_dir.mkdir(parents=True, exist_ok=True)
        stale_file = output_dir / self._OLD_FILE
        stale_file.write_text("old data", encoding="utf-8")

        packager = SubmissionPackager(mock_validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[],
        )
        packager.package(config)
        assert not stale_file.exists()

    def test_package_validation_failure_raises_value_error(
        self,
        packaging_setup: tuple[Path, Path, Path, Path],
    ) -> None:
        """Verify validation failure halts packaging and raises ValueError."""
        templates_dir, chk_dir, output_dir, zip_path = packaging_setup
        failing_validator = MagicMock(spec=ConstraintValidator)
        failing_validator.validate.return_value = ValidationReport(
            checks=[ValidationCheck(name="root_config", passed=False, detail="missing")],
            all_passed=False,
        )
        packager = SubmissionPackager(failing_validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[],
        )
        with pytest.raises(ValueError, match="Submission pre-flight validation failed"):
            packager.package(config)

    def test_package_output_dir_initially_file(
        self,
        mock_validator: MagicMock,
        packaging_setup: tuple[Path, Path, Path, Path],
    ) -> None:
        """Verify output_dir is replaced even if it exists as a file initially."""
        templates_dir, _, output_dir, zip_path = packaging_setup
        output_dir.write_text("file obstacle", encoding="utf-8")
        packager = SubmissionPackager(mock_validator)
        config = DeployConfig(
            templates_dir=str(templates_dir),
            output_dir=str(output_dir),
            zip_path=str(zip_path),
            adapters=[],
        )
        packager.package(config)
        assert output_dir.is_dir()

    def test_package_with_single_file_adapter_and_object_adapter(
        self,
        mock_validator: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify packager handles single file adapter and non-dict adapter items."""
        from types import SimpleNamespace

        out_dir = tmp_path / "out"
        zip_p = tmp_path / "out.zip"
        file_chk = tmp_path / "single_file_checkpoint.bin"
        file_chk.write_bytes(b"content")

        adapter_obj = SimpleNamespace(checkpoint_path=str(file_chk), name="")
        packager = SubmissionPackager(mock_validator)
        config = DeployConfig(
            templates_dir=str(tmp_path / "non_existent_templates"),
            output_dir=str(out_dir),
            zip_path=str(zip_p),
            adapters=[{"checkpoint_path": "non_existent_path", "name": "dummy"}],
        )
        config.adapters.append(adapter_obj)  # type: ignore[arg-type]
        result = packager.package(config)
        assert Path(result).is_file()

