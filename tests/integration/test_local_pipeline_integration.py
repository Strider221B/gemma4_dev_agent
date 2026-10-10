"""Integration test for end-to-end local Kaggle execution pipeline."""

from __future__ import annotations

import os
import tempfile
import zipfile
from typing import ClassVar

from src.training.local_pipeline_runner import LocalPipelineRunner


class TestLocalPipelineIntegration:
    """Integration test suite executing real surrogate SFT training and packaging."""

    _STATUS_KEY: ClassVar[str] = "status"
    _STATUS_SUCCESS: ClassVar[str] = "success"
    _ZIP_KEY: ClassVar[str] = "submission_zip"
    _ADAPTER_KEY: ClassVar[str] = "adapter_path"
    _ELAPSED_KEY: ClassVar[str] = "elapsed_seconds"
    _CONFIG_NAME: ClassVar[str] = "adapter_config.json"
    _WEIGHTS_NAME: ClassVar[str] = "adapter_model.safetensors"
    _AGENT_YAML: ClassVar[str] = "agent.yaml"
    _ZIP_ADAPTER_PREFIX: ClassVar[str] = "adapters/coder_lora/"

    def test_end_to_end_smoke_pipeline_run(self) -> None:
        """Verify real end-to-end surrogate training, safetensors checkpointing, and packaging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = LocalPipelineRunner(work_dir=tmpdir)
            summary = runner.run_smoke()

            assert summary[self._STATUS_KEY] == self._STATUS_SUCCESS
            zip_path = str(summary[self._ZIP_KEY])
            adapter_path = str(summary[self._ADAPTER_KEY])

            assert os.path.isfile(zip_path)
            assert os.path.isfile(os.path.join(adapter_path, self._CONFIG_NAME))
            assert os.path.isfile(os.path.join(adapter_path, self._WEIGHTS_NAME))
            self._verify_submission_zip_contents(zip_path)

    def _verify_submission_zip_contents(self, zip_path: str) -> None:
        """Verify packaged submission zip contains root agent.yaml and adapter safetensors."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_names = set(zf.namelist())
            assert self._AGENT_YAML in file_names
            expected_adapter_cfg = f"{self._ZIP_ADAPTER_PREFIX}{self._CONFIG_NAME}"
            expected_adapter_weights = f"{self._ZIP_ADAPTER_PREFIX}{self._WEIGHTS_NAME}"
            assert expected_adapter_cfg in file_names
            assert expected_adapter_weights in file_names
