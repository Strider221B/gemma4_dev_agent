"""Data ingestion module for loading benchmark tasks, graphs, and embeddings."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.data_paths_config import DataPathsConfig
from src.data.complexity_tier import ComplexityTier
from src.data.task import Task


class DataIngestor:
    """Ingests dataset files including tasks JSONL, code graphs, and embeddings."""

    _COMPLEXITY_SIMPLE_MAX_LINES: int = 20
    _COMPLEXITY_SIMPLE_MAX_FILES: int = 1
    _COMPLEXITY_MODERATE_MAX_LINES: int = 100
    _COMPLEXITY_MODERATE_MAX_FILES: int = 3
    _DIFF_HEADER: str = "diff --git"
    _DIFF_OLD_PREFIX: str = "--- "
    _DEV_NULL: str = "/dev/null"
    _DEV_NULL_REL: str = "dev/null"
    _ADD_PREFIX: str = "+"
    _REMOVE_PREFIX: str = "-"
    _ADD_HEADER_PREFIX: str = "+++"
    _REMOVE_HEADER_PREFIX: str = "---"
    _JSON_EXT: str = ".json"
    _NPZ_EXT: str = ".npz"
    _ENCODING: str = "utf-8"
    _KEY_INSTANCE_ID: str = "instance_id"
    _KEY_REPO: str = "repo"
    _KEY_BASE_COMMIT: str = "base_commit"
    _KEY_PROBLEM_STATEMENT: str = "problem_statement"
    _KEY_HINTS_TEXT: str = "hints_text"
    _KEY_PATCH: str = "patch"
    _KEY_TEST_PATCH: str = "test_patch"
    _KEY_CREATED_AT: str = "created_at"
    _EMPTY_STR: str = ""

    def __init__(self, data_paths: DataPathsConfig) -> None:
        """Initialize DataIngestor with configured artifact paths."""
        self._data_paths: DataPathsConfig = data_paths

    def load_tasks(self) -> list[Task]:
        """Read tasks.jsonl and parse each line into a Task object."""
        path = Path(self._data_paths.tasks_path)
        if not path.is_file():
            return []
        tasks: list[Task] = []
        with open(path, mode="r", encoding=self._ENCODING) as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if stripped:
                    tasks.append(self._parse_task_line(stripped))
        return tasks

    def load_graph(self, instance_id: str) -> object:
        """Load code graph for a given instance ID from disk."""
        path = (
            Path(self._data_paths.graphs_dir) / f"{instance_id}{self._JSON_EXT}"
        )
        if not path.is_file():
            return None
        data = self._load_json_file(str(path))
        try:
            from networkx.readwrite import json_graph

            return json_graph.node_link_graph(
                data, directed=True, multigraph=True
            )
        except (ImportError, Exception):
            return data

    def load_embeddings(self, instance_id: str) -> dict[str, object]:
        """Load embeddings array dictionary for a given instance ID."""
        path = (
            Path(self._data_paths.embeddings_dir)
            / f"{instance_id}{self._NPZ_EXT}"
        )
        if not path.is_file():
            return {}
        try:
            import numpy as np

            loaded = np.load(str(path))
            return {str(key): loaded[key] for key in loaded.files}
        except (ImportError, Exception):
            return {}

    def classify_complexity(self, task: Task) -> ComplexityTier:
        """Classify task complexity based on patch files and lines touched."""
        num_files = self._count_changed_files(task.patch)
        num_lines = self._count_changed_lines(task.patch)
        if (
            num_files <= self._COMPLEXITY_SIMPLE_MAX_FILES
            and num_lines < self._COMPLEXITY_SIMPLE_MAX_LINES
        ):
            return ComplexityTier.SIMPLE
        if (
            num_files <= self._COMPLEXITY_MODERATE_MAX_FILES
            and num_lines <= self._COMPLEXITY_MODERATE_MAX_LINES
        ):
            return ComplexityTier.MODERATE
        return ComplexityTier.COMPLEX

    def _parse_task_line(self, line: str) -> Task:
        """Parse a single JSON line into a Task instance."""
        data = json.loads(line)
        return Task(
            instance_id=str(data.get(self._KEY_INSTANCE_ID, self._EMPTY_STR)),
            repo=str(data.get(self._KEY_REPO, self._EMPTY_STR)),
            base_commit=str(data.get(self._KEY_BASE_COMMIT, self._EMPTY_STR)),
            problem_statement=str(
                data.get(self._KEY_PROBLEM_STATEMENT, self._EMPTY_STR)
            ),
            hints_text=str(data.get(self._KEY_HINTS_TEXT, self._EMPTY_STR)),
            patch=str(data.get(self._KEY_PATCH, self._EMPTY_STR)),
            test_patch=str(data.get(self._KEY_TEST_PATCH, self._EMPTY_STR)),
            created_at=str(data.get(self._KEY_CREATED_AT, self._EMPTY_STR)),
        )

    def _count_changed_files(self, patch: str) -> int:
        """Count the number of distinct files touched in the patch."""
        git_diff_count = 0
        old_file_count = 0
        for line in patch.splitlines():
            if line.startswith(self._DIFF_HEADER):
                git_diff_count += 1
            elif line.startswith(self._DIFF_OLD_PREFIX):
                target = line[len(self._DIFF_OLD_PREFIX) :].strip()
                if target not in (self._DEV_NULL, self._DEV_NULL_REL):
                    old_file_count += 1
        if git_diff_count > 0:
            return git_diff_count
        return old_file_count

    def _count_changed_lines(self, patch: str) -> int:
        """Count added and removed lines in the patch excluding headers."""
        count = 0
        for line in patch.splitlines():
            if line.startswith(self._ADD_PREFIX) and not line.startswith(
                self._ADD_HEADER_PREFIX
            ):
                count += 1
            elif line.startswith(self._REMOVE_PREFIX) and not line.startswith(
                self._REMOVE_HEADER_PREFIX
            ):
                count += 1
        return count

    def _load_json_file(self, path: str) -> dict[str, object]:
        """Read and decode JSON file from given file path."""
        with open(path, mode="r", encoding=self._ENCODING) as file_handle:
            content: object = json.load(file_handle)
            if isinstance(content, dict):
                return {str(k): v for k, v in content.items()}
            return {}
