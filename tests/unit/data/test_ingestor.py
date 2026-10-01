"""Unit tests for DataIngestor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config.data_paths_config import DataPathsConfig
from src.data.complexity_tier import ComplexityTier
from src.data.ingestor import DataIngestor
from src.data.task import Task


class TestIngestor:
    """Test suite for DataIngestor functionality."""

    _SAMPLE_TASK_ID: str = "task_001"
    _SAMPLE_REPO: str = "astral-sh/uv"
    _SAMPLE_COMMIT: str = "abcdef123456"
    _SAMPLE_PROBLEM: str = "Bug in resolution"
    _SAMPLE_HINTS: str = "Check solver.py"
    _SAMPLE_DATE: str = "2026-01-01"
    _EMPTY_STR: str = ""
    _SIMPLE_PATCH: str = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-old\n+new\n"
    )
    _MODERATE_PATCH: str = (
        "diff --git a/a.py b/a.py\n@@ -1,25 +1,25 @@\n"
        + "\n".join(f"+line_{i}" for i in range(25))
        + "\ndiff --git a/b.py b/b.py\n@@ -1,25 +1,25 @@\n"
        + "\n".join(f"+line_{i}" for i in range(25))
    )
    _COMPLEX_PATCH: str = (
        "diff --git a/a.py b/a.py\n+a\n"
        "diff --git a/b.py b/b.py\n+b\n"
        "diff --git a/c.py b/c.py\n+c\n"
        "diff --git a/d.py b/d.py\n+d\n"
    )
    _HIGH_LINE_PATCH: str = (
        "diff --git a/a.py b/a.py\n"
        + "\n".join(f"+line_{i}" for i in range(120))
    )
    _GRAPH_ID: str = "inst_graph"
    _EMBEDDINGS_ID: str = "inst_embed"
    _MISSING_ID: str = "missing_instance"
    _NODE_KEY: str = "test_node"

    def test_load_tasks_reads_jsonl(self, tmp_path: Path) -> None:
        """Verify reading a populated tasks.jsonl creates Task objects."""
        tasks_file = tmp_path / "tasks.jsonl"
        task_data = {
            "instance_id": self._SAMPLE_TASK_ID,
            "repo": self._SAMPLE_REPO,
            "base_commit": self._SAMPLE_COMMIT,
            "problem_statement": self._SAMPLE_PROBLEM,
            "hints_text": self._SAMPLE_HINTS,
            "patch": self._SIMPLE_PATCH,
            "test_patch": self._EMPTY_STR,
            "created_at": self._SAMPLE_DATE,
        }
        tasks_file.write_text(json.dumps(task_data) + "\n", encoding="utf-8")
        config = DataPathsConfig(
            tasks_path=str(tasks_file),
            graphs_dir=str(tmp_path),
            embeddings_dir=str(tmp_path),
            snapshots_dir=str(tmp_path),
        )
        ingestor = DataIngestor(config)
        tasks = ingestor.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].instance_id == self._SAMPLE_TASK_ID
        assert tasks[0].repo == self._SAMPLE_REPO

    def test_load_tasks_empty_file_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        """Verify reading an empty tasks.jsonl yields an empty list."""
        tasks_file = tmp_path / "empty_tasks.jsonl"
        tasks_file.write_text("", encoding="utf-8")
        config = DataPathsConfig(tasks_path=str(tasks_file))
        ingestor = DataIngestor(config)
        assert ingestor.load_tasks() == []

    def test_load_tasks_missing_file_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        """Verify non-existent tasks.jsonl returns empty list without error."""
        config = DataPathsConfig(tasks_path=str(tmp_path / "missing.jsonl"))
        ingestor = DataIngestor(config)
        assert ingestor.load_tasks() == []

    def test_classify_complexity_simple(self) -> None:
        """Verify 1 file and < 20 lines is classified as SIMPLE."""
        config = DataPathsConfig()
        ingestor = DataIngestor(config)
        task = Task(
            instance_id=self._SAMPLE_TASK_ID,
            repo=self._SAMPLE_REPO,
            base_commit=self._SAMPLE_COMMIT,
            problem_statement=self._SAMPLE_PROBLEM,
            hints_text=self._SAMPLE_HINTS,
            patch=self._SIMPLE_PATCH,
            test_patch=self._EMPTY_STR,
            created_at=self._SAMPLE_DATE,
        )
        assert ingestor.classify_complexity(task) == ComplexityTier.SIMPLE

    def test_classify_complexity_moderate(self) -> None:
        """Verify 2 files and 50 lines is classified as MODERATE."""
        config = DataPathsConfig()
        ingestor = DataIngestor(config)
        task = Task(
            instance_id=self._SAMPLE_TASK_ID,
            repo=self._SAMPLE_REPO,
            base_commit=self._SAMPLE_COMMIT,
            problem_statement=self._SAMPLE_PROBLEM,
            hints_text=self._SAMPLE_HINTS,
            patch=self._MODERATE_PATCH,
            test_patch=self._EMPTY_STR,
            created_at=self._SAMPLE_DATE,
        )
        assert ingestor.classify_complexity(task) == ComplexityTier.MODERATE

    def test_classify_complexity_complex_by_files(self) -> None:
        """Verify 4 files is classified as COMPLEX."""
        config = DataPathsConfig()
        ingestor = DataIngestor(config)
        task = Task(
            instance_id=self._SAMPLE_TASK_ID,
            repo=self._SAMPLE_REPO,
            base_commit=self._SAMPLE_COMMIT,
            problem_statement=self._SAMPLE_PROBLEM,
            hints_text=self._SAMPLE_HINTS,
            patch=self._COMPLEX_PATCH,
            test_patch=self._EMPTY_STR,
            created_at=self._SAMPLE_DATE,
        )
        assert ingestor.classify_complexity(task) == ComplexityTier.COMPLEX

    def test_classify_complexity_complex_by_lines(self) -> None:
        """Verify 1 file with > 100 lines is classified as COMPLEX."""
        config = DataPathsConfig()
        ingestor = DataIngestor(config)
        task = Task(
            instance_id=self._SAMPLE_TASK_ID,
            repo=self._SAMPLE_REPO,
            base_commit=self._SAMPLE_COMMIT,
            problem_statement=self._SAMPLE_PROBLEM,
            hints_text=self._SAMPLE_HINTS,
            patch=self._HIGH_LINE_PATCH,
            test_patch=self._EMPTY_STR,
            created_at=self._SAMPLE_DATE,
        )
        assert ingestor.classify_complexity(task) == ComplexityTier.COMPLEX

    def test_load_graph_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Verify loading graph for missing file returns None."""
        config = DataPathsConfig(graphs_dir=str(tmp_path))
        ingestor = DataIngestor(config)
        assert ingestor.load_graph(self._MISSING_ID) is None

    def test_load_graph_success(self, tmp_path: Path) -> None:
        """Verify successfully loading graph from JSON file."""
        graph_file = tmp_path / f"{self._GRAPH_ID}.json"
        graph_data = {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [{"id": self._NODE_KEY}],
            "links": [],
        }
        graph_file.write_text(json.dumps(graph_data), encoding="utf-8")
        config = DataPathsConfig(graphs_dir=str(tmp_path))
        ingestor = DataIngestor(config)
        graph = ingestor.load_graph(self._GRAPH_ID)
        assert graph is not None

    def test_load_embeddings_missing_file_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """Verify loading embeddings for missing file returns empty dictionary."""
        config = DataPathsConfig(embeddings_dir=str(tmp_path))
        ingestor = DataIngestor(config)
        assert ingestor.load_embeddings(self._MISSING_ID) == {}

    def test_load_embeddings_success(self, tmp_path: Path) -> None:
        """Verify successfully loading embeddings from .npz file."""
        embed_file = tmp_path / f"{self._EMBEDDINGS_ID}.npz"
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.savez(embed_file, **{self._NODE_KEY: vector})
        config = DataPathsConfig(embeddings_dir=str(tmp_path))
        ingestor = DataIngestor(config)
        embeddings = ingestor.load_embeddings(self._EMBEDDINGS_ID)
        assert self._NODE_KEY in embeddings
        assert len(embeddings[self._NODE_KEY]) == 3
