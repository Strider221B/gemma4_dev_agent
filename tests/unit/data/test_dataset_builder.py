"""Unit tests for DatasetBuilder."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data.complexity_tier import ComplexityTier
from src.data.dataset_builder import DatasetBuilder
from src.data.mock_data_factory import MockDataFactory
from src.data.trajectory import Trajectory
from src.data.turn import Turn
from src.data.validation_result import ValidationResult


class TestDatasetBuilder:
    """Test suite for DatasetBuilder orchestrating the data processing pipeline."""

    _REQUIRED_COLUMNS: tuple[str, ...] = (
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
    _TOKEN_BOS: str = "<bos>"
    _TOKEN_START: str = "<|turn>"
    _TOKEN_END: str = "<turn|>"

    def test_build_mock_returns_dataset(self) -> None:
        """Verify build_mock executes end-to-end and returns a non-empty dataset."""
        builder = DatasetBuilder(mock_mode=True)
        dataset = builder.build_mock()
        assert len(dataset) >= 1

    def test_build_mock_has_required_columns(self) -> None:
        """Verify the generated mock dataset contains all schema columns."""
        builder = DatasetBuilder(mock_mode=True)
        dataset = builder.build_mock()
        for column in self._REQUIRED_COLUMNS:
            assert column in dataset.column_names

    def test_build_mock_formatted_text_contains_special_tokens(self) -> None:
        """Verify formatted_text in the output contains Gemma 4 special tokens."""
        builder = DatasetBuilder(mock_mode=True)
        dataset = builder.build_mock()
        first_row = dataset[0]
        text = str(first_row["formatted_text"])
        assert self._TOKEN_BOS in text
        assert self._TOKEN_START in text
        assert self._TOKEN_END in text

    def test_build_filters_invalid_trajectories(self) -> None:
        """Verify trajectories failing validation are discarded from final dataset."""
        validator = MagicMock()
        valid_res = ValidationResult(valid=True, errors=[], warnings=[])
        invalid_res = ValidationResult(valid=False, errors=["err"], warnings=[])
        validator.validate.side_effect = [valid_res, invalid_res, valid_res, valid_res]

        builder = DatasetBuilder(validator=validator, mock_mode=True)
        trajectories = [
            self._build_dummy_trajectory("id_1", 100),
            self._build_dummy_trajectory("id_2", 100),
        ]
        validator.validate.side_effect = [valid_res, invalid_res]
        filtered = builder._validate_trajectories(trajectories)
        assert len(filtered) == 1
        assert filtered[0].instance_id == "id_1"

    def test_build_assigns_folds(self) -> None:
        """Verify fold numbers are mapped onto dataset records from CVSplitter."""
        builder = DatasetBuilder(mock_mode=True)
        dataset = builder.build_mock()
        folds = [row["fold"] for row in dataset]
        assert all(isinstance(f, int) for f in folds)
        assert len(set(folds)) > 1

    def test_build_real_mode_delegates_to_ingestor(self) -> None:
        """Verify build() in non-mock mode invokes load_tasks on ingestor."""
        mock_ingestor = MagicMock()
        task = MockDataFactory().create_mock_task()
        mock_ingestor.load_tasks.return_value = [task]

        builder = DatasetBuilder(ingestor=mock_ingestor, mock_mode=False)
        builder.build()
        mock_ingestor.load_tasks.assert_called_once()

    def test_filter_by_token_budget_removes_over_budget(self) -> None:
        """Verify trajectories with token_count exceeding budget are excluded."""
        builder = DatasetBuilder(mock_mode=True)
        under_budget = self._build_dummy_trajectory("under", 1000)
        over_budget = self._build_dummy_trajectory("over", 35000)
        result = builder._filter_by_token_budget([under_budget, over_budget])
        assert len(result) == 1
        assert result[0].instance_id == "under"

    def test_build_with_mock_mode_flag_in_build(self) -> None:
        """Verify build() redirects to build_mock when mock_mode is set."""
        builder = DatasetBuilder(mock_mode=True)
        dataset = builder.build()
        assert len(dataset) >= 1

    def _build_dummy_trajectory(
        self, instance_id: str, token_count: int
    ) -> Trajectory:
        """Construct a minimal Trajectory instance for testing filters."""
        return Trajectory(
            instance_id=instance_id,
            repo="mock/repo",
            complexity=ComplexityTier.SIMPLE,
            turns=[Turn(role="user", text="hello")],
            token_count=token_count,
            num_tool_calls=1,
            num_files_changed=1,
        )
