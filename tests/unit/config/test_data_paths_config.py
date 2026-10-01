"""Unit tests for DataPathsConfig schema."""

from __future__ import annotations

from src.config.data_paths_config import DataPathsConfig


class TestDataPathsConfig:
    """Test suite for DataPathsConfig Pydantic model validation."""

    _CUSTOM_TASKS: str = "/custom/data/tasks.jsonl"
    _CUSTOM_GRAPHS: str = "/custom/data/graphs"

    def test_data_paths_config_defaults(self) -> None:
        """Verify default file path configurations for Kaggle environment."""
        config = DataPathsConfig()
        assert "tasks.jsonl" in config.tasks_path
        assert "graphs" in config.graphs_dir
        assert "embeddings" in config.embeddings_dir
        assert "snapshots" in config.snapshots_dir

    def test_data_paths_config_custom_values(self) -> None:
        """Verify custom data paths override defaults."""
        config = DataPathsConfig(
            tasks_path=self._CUSTOM_TASKS,
            graphs_dir=self._CUSTOM_GRAPHS,
        )
        assert config.tasks_path == self._CUSTOM_TASKS
        assert config.graphs_dir == self._CUSTOM_GRAPHS
