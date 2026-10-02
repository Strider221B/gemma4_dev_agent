"""Integration test for end-to-end data preprocessing pipeline."""

from __future__ import annotations

from src.data.dataset_builder import DatasetBuilder


class TestDataPipeline:
    """Integration test suite executing the entire mock data transformation pipeline."""

    _SCHEMA_COLUMNS: tuple[str, ...] = (
        "instance_id",
        "repo",
        "complexity",
        "messages",
        "formatted_text",
        "token_count",
        "num_tool_calls",
        "num_files_changed",
        "fold",
    )
    _VALID_COMPLEXITIES: tuple[str, ...] = ("SIMPLE", "MODERATE", "COMPLEX")
    _TOKEN_BOS: str = "<bos>"
    _TOKEN_START: str = "<start_of_turn>"
    _TOKEN_END: str = "<end_of_turn>"
    _KEY_ROLE: str = "role"
    _KEY_CONTENT: str = "content"

    def test_full_mock_pipeline(self) -> None:
        """Run full mock pipeline and verify end-to-end dataset structure and integrity."""
        builder = DatasetBuilder(mock_mode=True)
        dataset = builder.build_mock()

        assert len(dataset) >= 4
        for col in self._SCHEMA_COLUMNS:
            assert col in dataset.column_names

        for row_idx in range(len(dataset)):
            row = dataset[row_idx]
            self._verify_row_structure(row)

    def _verify_row_structure(self, row: dict[str, object]) -> None:
        """Validate structure and data types of an individual dataset record."""
        assert isinstance(row["instance_id"], str) and len(row["instance_id"]) > 0
        assert isinstance(row["repo"], str) and len(row["repo"]) > 0
        assert row["complexity"] in self._VALID_COMPLEXITIES
        assert isinstance(row["messages"], list) and len(row["messages"]) > 0
        self._verify_messages(row["messages"])
        assert isinstance(row["formatted_text"], str)
        assert self._TOKEN_BOS in row["formatted_text"]
        assert self._TOKEN_START in row["formatted_text"]
        assert self._TOKEN_END in row["formatted_text"]
        assert isinstance(row["token_count"], int) and row["token_count"] > 0
        assert isinstance(row["num_tool_calls"], int) and row["num_tool_calls"] > 0
        assert isinstance(row["num_files_changed"], int) and row["num_files_changed"] > 0
        assert isinstance(row["fold"], int) and row["fold"] >= 0

    def _verify_messages(self, messages: object) -> None:
        """Validate message dictionaries inside the HF conversation list."""
        assert isinstance(messages, list)
        for msg in messages:
            assert isinstance(msg, dict)
            assert self._KEY_ROLE in msg
            assert self._KEY_CONTENT in msg
