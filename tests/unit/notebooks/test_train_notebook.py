"""Unit tests verifying train_notebook.ipynb structure, cell count, and contents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from src.evaluation.notebook_parser import NotebookParser


class TestTrainNotebook:
    """Test suite verifying Kaggle training notebook format and cell structure."""

    _NOTEBOOK_PATH: ClassVar[str] = "notebooks/train_notebook.ipynb"
    _EXPECTED_CODE_CELL_COUNT: ClassVar[int] = 8
    _KEY_CELLS: ClassVar[str] = "cells"
    _KEY_CELL_TYPE: ClassVar[str] = "cell_type"
    _KEY_SOURCE: ClassVar[str] = "source"
    _TYPE_CODE: ClassVar[str] = "code"
    _REQUIRED_IMPORTS: ClassVar[tuple[str, ...]] = (
        "ConfigManager",
        "DatasetBuilder",
        "DataIngestor",
        "TrajectorySynthesiser",
        "SFTTrainerPipeline",
        "SubmissionPackager",
        "ConstraintValidator",
    )

    def test_train_notebook_exists_and_parses_json(self) -> None:
        """Verify train_notebook.ipynb exists on disk and contains valid JSON."""
        path = Path(self._NOTEBOOK_PATH)
        assert path.is_file(), f"Notebook not found at {self._NOTEBOOK_PATH}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert self._KEY_CELLS in data

    def test_train_notebook_has_eight_code_cells(self) -> None:
        """Verify the notebook contains precisely eight code cells matching design."""
        parser = NotebookParser()
        nb = parser.load_notebook(self._NOTEBOOK_PATH)
        code_cells = parser.extract_code_cells(nb)
        assert len(code_cells) == self._EXPECTED_CODE_CELL_COUNT

    def test_train_notebook_imports_required_components(self) -> None:
        """Verify cell 2 imports all primary architecture pipeline components."""
        parser = NotebookParser()
        nb = parser.load_notebook(self._NOTEBOOK_PATH)
        code_cells = parser.extract_code_cells(nb)
        cell2_source = str(code_cells[1][self._KEY_SOURCE])
        for required_import in self._REQUIRED_IMPORTS:
            assert required_import in cell2_source, f"Missing import: {required_import}"

    def test_train_notebook_contains_install_and_packaging_cells(self) -> None:
        """Verify initial cell contains install and final cell packages submission."""
        parser = NotebookParser()
        nb = parser.load_notebook(self._NOTEBOOK_PATH)
        code_cells = parser.extract_code_cells(nb)
        cell1_source = str(code_cells[0][self._KEY_SOURCE])
        cell8_source = str(code_cells[7][self._KEY_SOURCE])
        assert "pip install" in cell1_source
        assert "SubmissionPackager" in cell8_source
        assert "packager.package" in cell8_source
