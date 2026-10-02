"""In-memory dataset representation when HuggingFace datasets is optional."""

from __future__ import annotations

from typing import Iterator


class InMemoryDataset:
    """Lightweight in-memory dataset providing column access and row indexing."""

    _EMPTY_INDEX_ERR: str = "Dataset is empty"
    _INDEX_ERR: str = "Index out of range: "
    _KEY_ERR: str = "Column not found: "

    def __init__(self, records: list[dict[str, object]]) -> None:
        """Initialize InMemoryDataset with list of record dictionaries."""
        self._records: list[dict[str, object]] = list(records)
        self._column_names: list[str] = (
            list(records[0].keys()) if records else []
        )

    def __len__(self) -> int:
        """Return total number of rows in the dataset."""
        return len(self._records)

    def __getitem__(self, item: int | str) -> object:
        """Access either a single row dict by index or a column list by name."""
        if isinstance(item, int):
            if item < 0 or item >= len(self._records):
                raise IndexError(f"{self._INDEX_ERR}{item}")
            return self._records[item]
        if item not in self._column_names:
            raise KeyError(f"{self._KEY_ERR}{item}")
        return [record.get(item) for record in self._records]

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Iterate over records in the dataset."""
        return iter(self._records)

    @property
    def column_names(self) -> list[str]:
        """Return list of column names present in the dataset."""
        return list(self._column_names)

    def to_dict(self) -> dict[str, list[object]]:
        """Convert dataset into columnar dictionary mapping."""
        return {
            col: [rec.get(col) for rec in self._records]
            for col in self._column_names
        }
