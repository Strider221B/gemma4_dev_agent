"""Jupyter notebook parser extracting code and markdown cells for inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NotebookParser:
    """Parses Jupyter notebook files (.ipynb) into structured cells."""

    _KEY_CELLS: str = "cells"
    _KEY_CELL_TYPE: str = "cell_type"
    _KEY_SOURCE: str = "source"
    _TYPE_CODE: str = "code"
    _TYPE_MARKDOWN: str = "markdown"
    _FIELD_INDEX: str = "index"
    _ENCODING: str = "utf-8"

    def __init__(self) -> None:
        """Initialize the notebook parser."""

    def load_notebook(self, path: str) -> dict[str, Any]:
        """Load an .ipynb file from path and return parsed JSON dictionary."""
        content = self._read_file(path)
        parsed: Any = json.loads(content)
        if not isinstance(parsed, dict):
            return {}
        return parsed

    def extract_code_cells(
        self, notebook: dict[str, Any]
    ) -> list[dict[str, object]]:
        """Extract all code cells with their index and consolidated source text."""
        return self._extract_cells_by_type(notebook, self._TYPE_CODE)

    def extract_markdown_cells(
        self, notebook: dict[str, Any]
    ) -> list[dict[str, object]]:
        """Extract all markdown cells with their index and consolidated source text."""
        return self._extract_cells_by_type(notebook, self._TYPE_MARKDOWN)

    def _extract_cells_by_type(
        self, notebook: dict[str, Any], cell_type: str
    ) -> list[dict[str, object]]:
        """Extract cells matching the given cell_type."""
        raw_cells = notebook.get(self._KEY_CELLS, [])
        if not isinstance(raw_cells, list):
            return []
        extracted: list[dict[str, object]] = []
        for idx, cell in enumerate(raw_cells):
            if isinstance(cell, dict) and cell.get(self._KEY_CELL_TYPE) == cell_type:
                source_str = self._normalize_source(cell.get(self._KEY_SOURCE, ""))
                extracted.append({
                    self._FIELD_INDEX: idx,
                    self._KEY_SOURCE: source_str,
                })
        return extracted

    def _normalize_source(self, raw_source: object) -> str:
        """Consolidate source representation whether string or sequence of lines."""
        if isinstance(raw_source, list):
            return "".join(str(item) for item in raw_source)
        if isinstance(raw_source, str):
            return raw_source
        return ""

    def _read_file(self, path: str) -> str:
        """Read and return complete content of the file at path."""
        return Path(path).read_text(encoding=self._ENCODING)
