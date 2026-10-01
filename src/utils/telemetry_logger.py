"""Telemetry and metric logger producing JSON Lines outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TelemetryLogger:
    """Logger for recording run lifecycle events, training metrics, and rewards."""

    _LOG_FORMAT: str = "json"
    _DEFAULT_LOG_DIR: str = "logs"
    _DEFAULT_FILE_NAME: str = "telemetry.jsonl"
    _TIME_FORMAT: str = "%Y-%m-%dT%H:%M:%SZ"
    _EVENT_START: str = "start"
    _EVENT_END: str = "end"
    _EVENT_METRIC: str = "metric"
    _EVENT_INFO: str = "info"
    _EVENT_REWARD: str = "reward"
    _FILE_ENCODING: str = "utf-8"

    def __init__(self, log_dir: str = _DEFAULT_LOG_DIR, run_version: str = "0.1.0") -> None:
        """Initialize TelemetryLogger with directory path and version."""
        self._log_dir: Path = Path(log_dir)
        self._run_version: str = run_version
        self._log_path: Path = self._log_dir / self._DEFAULT_FILE_NAME

    def log_start(self, phase: str, config: object) -> None:
        """Record phase initiation with a snapshot of active configuration."""
        entry: dict[str, object] = {
            "timestamp": self._get_timestamp(),
            "run_version": self._run_version,
            "event_type": self._EVENT_START,
            "phase": phase,
            "config": self._serialize_config(config),
        }
        self._write_entry(entry)

    def log_end(self, phase: str, metrics: dict[str, object]) -> None:
        """Record phase completion along with final metrics."""
        entry: dict[str, object] = {
            "timestamp": self._get_timestamp(),
            "run_version": self._run_version,
            "event_type": self._EVENT_END,
            "phase": phase,
            "metrics": metrics,
        }
        self._write_entry(entry)

    def log_metrics(self, metrics: dict[str, object]) -> None:
        """Append training or evaluation metrics record."""
        entry: dict[str, object] = {
            "timestamp": self._get_timestamp(),
            "run_version": self._run_version,
            "event_type": self._EVENT_METRIC,
            "metrics": metrics,
        }
        self._write_entry(entry)

    def log_info(self, message: str) -> None:
        """Record an informational status message."""
        entry: dict[str, object] = {
            "timestamp": self._get_timestamp(),
            "run_version": self._run_version,
            "event_type": self._EVENT_INFO,
            "message": message,
        }
        self._write_entry(entry)

    def log_reward(
        self, instance_id: str, components: dict[str, float], total: float
    ) -> None:
        """Record decomposed reward values for a task trajectory rollout."""
        entry: dict[str, object] = {
            "timestamp": self._get_timestamp(),
            "run_version": self._run_version,
            "event_type": self._EVENT_REWARD,
            "instance_id": instance_id,
            "components": components,
            "total": total,
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict[str, object]) -> None:
        """Append a JSON-serialized entry to the log file."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(entry) + "\n"
        with open(self._log_path, "a", encoding=self._FILE_ENCODING) as file_handle:
            file_handle.write(serialized)

    def _get_timestamp(self) -> str:
        """Return current UTC timestamp in ISO 8601 string format."""
        return datetime.now(timezone.utc).strftime(self._TIME_FORMAT)

    def _serialize_config(self, config: object) -> object:
        """Convert config object to a serializable dictionary or string representation."""
        if hasattr(config, "model_dump"):
            return getattr(config, "model_dump")()
        if hasattr(config, "__dict__"):
            return {
                str(key): val
                for key, val in vars(config).items()
                if not str(key).startswith("_")
            }
        return str(config)
