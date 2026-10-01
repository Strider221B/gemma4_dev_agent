"""Unit tests for TelemetryLogger."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.model_config import ModelConfig
from src.utils.telemetry_logger import TelemetryLogger


class TestTelemetryLogger:
    """Test suite for TelemetryLogger structured JSON Lines event logging."""

    _SAMPLE_PHASE: str = "sft_training"
    _SAMPLE_VERSION: str = "0.2.0"
    _SAMPLE_MESSAGE: str = "Starting training epoch 1"
    _SAMPLE_INSTANCE_ID: str = "fastapi_11194"
    _REWARD_TOTAL: float = 1.15

    def test_log_metrics_writes_json_line(self, tmp_path: Path) -> None:
        """Verify log_metrics appends a valid JSON line with metric details."""
        logger = TelemetryLogger(log_dir=str(tmp_path), run_version=self._SAMPLE_VERSION)
        metrics: dict[str, object] = {"loss": 0.35, "step": 100}
        logger.log_metrics(metrics)

        log_file = tmp_path / "telemetry.jsonl"
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "metric"
        assert entry["run_version"] == self._SAMPLE_VERSION
        assert entry["metrics"] == metrics

    def test_log_info_writes_message(self, tmp_path: Path) -> None:
        """Verify log_info appends an informational entry."""
        logger = TelemetryLogger(log_dir=str(tmp_path), run_version=self._SAMPLE_VERSION)
        logger.log_info(self._SAMPLE_MESSAGE)

        log_file = tmp_path / "telemetry.jsonl"
        lines = log_file.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        assert entry["event_type"] == "info"
        assert entry["message"] == self._SAMPLE_MESSAGE

    def test_log_start_includes_config(self, tmp_path: Path) -> None:
        """Verify log_start properly serializes config snapshots."""
        logger = TelemetryLogger(log_dir=str(tmp_path), run_version=self._SAMPLE_VERSION)
        config = ModelConfig()
        logger.log_start(self._SAMPLE_PHASE, config)

        log_file = tmp_path / "telemetry.jsonl"
        lines = log_file.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        assert entry["event_type"] == "start"
        assert entry["phase"] == self._SAMPLE_PHASE
        assert entry["config"]["name"] == "google/gemma-4-31b-it-qat-w4a16-ct"

    def test_log_start_with_plain_class(self, tmp_path: Path) -> None:
        """Verify log_start serializes objects with standard __dict__ attribute."""
        logger = TelemetryLogger(log_dir=str(tmp_path), run_version=self._SAMPLE_VERSION)

        class SampleConfig:
            def __init__(self) -> None:
                self.key = "sample_value"
                self._private = "hidden"

        logger.log_start(self._SAMPLE_PHASE, SampleConfig())
        log_file = tmp_path / "telemetry.jsonl"
        lines = log_file.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        assert entry["config"] == {"key": "sample_value"}

    def test_log_start_with_primitive(self, tmp_path: Path) -> None:
        """Verify log_start handles primitive string configurations."""
        logger = TelemetryLogger(log_dir=str(tmp_path), run_version=self._SAMPLE_VERSION)
        logger.log_start(self._SAMPLE_PHASE, "primitive_config")
        log_file = tmp_path / "telemetry.jsonl"
        lines = log_file.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        assert entry["config"] == "primitive_config"

    def test_log_end_includes_metrics(self, tmp_path: Path) -> None:
        """Verify log_end captures phase completion and final metrics."""
        logger = TelemetryLogger(log_dir=str(tmp_path), run_version=self._SAMPLE_VERSION)
        metrics: dict[str, object] = {"final_eval_loss": 0.21}
        logger.log_end(self._SAMPLE_PHASE, metrics)

        log_file = tmp_path / "telemetry.jsonl"
        lines = log_file.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        assert entry["event_type"] == "end"
        assert entry["phase"] == self._SAMPLE_PHASE
        assert entry["metrics"]["final_eval_loss"] == 0.21

    def test_log_reward_includes_components(self, tmp_path: Path) -> None:
        """Verify log_reward captures decomposed rollout reward items."""
        logger = TelemetryLogger(log_dir=str(tmp_path), run_version=self._SAMPLE_VERSION)
        components: dict[str, float] = {"pass_reward": 1.0, "efficiency_bonus": 0.15}
        logger.log_reward(self._SAMPLE_INSTANCE_ID, components, self._REWARD_TOTAL)

        log_file = tmp_path / "telemetry.jsonl"
        lines = log_file.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        assert entry["event_type"] == "reward"
        assert entry["instance_id"] == self._SAMPLE_INSTANCE_ID
        assert entry["components"] == components
        assert entry["total"] == self._REWARD_TOTAL
