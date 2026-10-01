"""Pydantic schema for dataset and artifact file paths."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class DataPathsConfig(BaseModel):
    """Configuration settings for dataset and graph artifact file paths."""

    _DEFAULT_TASKS_PATH: ClassVar[str] = (
        "/kaggle/input/gemma-4-developer-agent/published/tasks.jsonl"
    )
    _DEFAULT_GRAPHS_DIR: ClassVar[str] = (
        "/kaggle/input/gemma-4-developer-agent/published/graphs"
    )
    _DEFAULT_EMBEDDINGS_DIR: ClassVar[str] = (
        "/kaggle/input/gemma-4-developer-agent/published/embeddings"
    )
    _DEFAULT_SNAPSHOTS_DIR: ClassVar[str] = (
        "/kaggle/input/gemma-4-developer-agent/published/snapshots"
    )

    model_config = ConfigDict(extra="ignore")

    tasks_path: str = _DEFAULT_TASKS_PATH
    graphs_dir: str = _DEFAULT_GRAPHS_DIR
    embeddings_dir: str = _DEFAULT_EMBEDDINGS_DIR
    snapshots_dir: str = _DEFAULT_SNAPSHOTS_DIR
