"""Unit tests for NotebookParser."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.notebook_parser import NotebookParser


class TestNotebookParser:
    """Test suite validating NotebookParser JSON parsing and cell extraction."""

    _SAMPLE_CODE: str = "import os\nprint('hello')"
    _SAMPLE_MD: str = "# Analysis Header\nThis is a notebook."

    def test_load_notebook_reads_file(self, tmp_path: Path) -> None:
        """Verify load_notebook correctly reads and deserializes an .ipynb file."""
        nb_path = tmp_path / "test_nb.ipynb"
        payload = {
            "cells": [
                {"cell_type": "code", "source": ["import os\n", "print('hello')"]},
                {"cell_type": "markdown", "source": ["# Title\n"]},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 2,
        }
        nb_path.write_text(json.dumps(payload), encoding="utf-8")
        parser = NotebookParser()
        result = parser.load_notebook(str(nb_path))
        assert "cells" in result
        assert len(result["cells"]) == 2

    def test_load_notebook_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        """Verify load_notebook returns empty dict when top-level object is not dict."""
        nb_path = tmp_path / "invalid.ipynb"
        nb_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        parser = NotebookParser()
        result = parser.load_notebook(str(nb_path))
        assert result == {}

    def test_extract_code_cells_returns_list(self) -> None:
        """Verify extract_code_cells returns code cell entries with index and source."""
        parser = NotebookParser()
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": self._SAMPLE_MD},
                {"cell_type": "code", "source": [self._SAMPLE_CODE]},
            ]
        }
        cells = parser.extract_code_cells(notebook)
        assert len(cells) == 1
        assert cells[0]["index"] == 1
        assert cells[0]["source"] == self._SAMPLE_CODE

    def test_extract_code_cells_correct_count(self) -> None:
        """Verify extract_code_cells extracts exact number of code cells."""
        parser = NotebookParser()
        notebook = {
            "cells": [
                {"cell_type": "code", "source": "a = 1"},
                {"cell_type": "code", "source": "b = 2"},
                {"cell_type": "markdown", "source": "text"},
                {"cell_type": "code", "source": "c = 3"},
            ]
        }
        cells = parser.extract_code_cells(notebook)
        assert len(cells) == 3
        indices = [c["index"] for c in cells]
        assert indices == [0, 1, 3]

    def test_extract_markdown_cells_returns_list(self) -> None:
        """Verify extract_markdown_cells returns markdown cell entries."""
        parser = NotebookParser()
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": self._SAMPLE_MD},
                {"cell_type": "code", "source": "pass"},
                {"cell_type": "markdown", "source": ["Second markdown line\n"]},
            ]
        }
        md_cells = parser.extract_markdown_cells(notebook)
        assert len(md_cells) == 2
        assert md_cells[0]["index"] == 0
        assert md_cells[0]["source"] == self._SAMPLE_MD
        assert md_cells[1]["index"] == 2
        assert md_cells[1]["source"] == "Second markdown line\n"

    def test_extract_cells_empty_or_non_list(self) -> None:
        """Verify extract methods handle missing or malformed cells gracefully."""
        parser = NotebookParser()
        assert parser.extract_code_cells({}) == []
        assert parser.extract_markdown_cells({"cells": "invalid"}) == []

    def test_extract_cells_with_non_string_non_list_source(self) -> None:
        """Verify normalize_source returns empty string for unexpected source types."""
        parser = NotebookParser()
        notebook = {"cells": [{"cell_type": "code", "source": 12345}]}
        cells = parser.extract_code_cells(notebook)
        assert len(cells) == 1
        assert cells[0]["source"] == ""
