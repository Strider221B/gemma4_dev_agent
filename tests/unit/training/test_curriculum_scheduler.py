"""Unit tests for CurriculumScheduler evaluating progressive complexity tier filtering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.training.curriculum_scheduler import CurriculumScheduler


class TestCurriculumScheduler:
    """Test suite covering CurriculumScheduler epoch data filtering."""

    _TIER_SIMPLE: str = "SIMPLE"
    _TIER_MODERATE: str = "MODERATE"
    _TIER_COMPLEX: str = "COMPLEX"
    _TOTAL_EPOCHS: int = 5

    def test_early_stage_returns_simple_only(self) -> None:
        """Verify early stage (epoch 1/5) retains only SIMPLE items."""
        scheduler = CurriculumScheduler()
        dataset = [
            {"complexity": self._TIER_SIMPLE, "id": "t1"},
            {"complexity": self._TIER_MODERATE, "id": "t2"},
            {"complexity": self._TIER_COMPLEX, "id": "t3"},
        ]
        result = scheduler.get_epoch_data(dataset, epoch=1, total_epochs=self._TOTAL_EPOCHS)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "t1"

    def test_middle_stage_returns_simple_and_moderate(self) -> None:
        """Verify middle stage (epoch 3/5) retains SIMPLE and MODERATE items."""
        scheduler = CurriculumScheduler()
        dataset = [
            {"complexity": self._TIER_SIMPLE, "id": "t1"},
            {"complexity": self._TIER_MODERATE, "id": "t2"},
            {"complexity": self._TIER_COMPLEX, "id": "t3"},
        ]
        result = scheduler.get_epoch_data(dataset, epoch=3, total_epochs=self._TOTAL_EPOCHS)
        assert isinstance(result, list)
        assert len(result) == 2
        assert {item["id"] for item in result} == {"t1", "t2"}

    def test_late_stage_returns_all(self) -> None:
        """Verify late stage (epoch 5/5) retains all dataset items."""
        scheduler = CurriculumScheduler()
        dataset = [
            {"complexity": self._TIER_SIMPLE, "id": "t1"},
            {"complexity": self._TIER_MODERATE, "id": "t2"},
            {"complexity": self._TIER_COMPLEX, "id": "t3"},
        ]
        result = scheduler.get_epoch_data(dataset, epoch=5, total_epochs=self._TOTAL_EPOCHS)
        assert result == dataset

    def test_invalid_total_epochs_raises_value_error(self) -> None:
        """Verify ValueError raised when total_epochs <= 0."""
        scheduler = CurriculumScheduler()
        with pytest.raises(ValueError, match="must be greater than zero"):
            scheduler.get_epoch_data([], epoch=1, total_epochs=0)

    def test_filter_dataset_with_filter_method(self) -> None:
        """Verify scheduler interacts with datasets having a filter() API."""
        scheduler = CurriculumScheduler()
        mock_ds = MagicMock()
        mock_filtered = MagicMock()
        mock_ds.filter.return_value = mock_filtered

        result = scheduler.get_epoch_data(mock_ds, epoch=1, total_epochs=self._TOTAL_EPOCHS)
        assert result == mock_filtered
        mock_ds.filter.assert_called_once()

    def test_filter_dataset_with_object_attributes(self) -> None:
        """Verify filtering items represented as objects with attributes."""
        scheduler = CurriculumScheduler()
        items = [
            SimpleNamespace(complexity=self._TIER_SIMPLE, id="s1"),
            SimpleNamespace(complexity=self._TIER_COMPLEX, id="c1"),
        ]
        result = scheduler.get_epoch_data(items, epoch=1, total_epochs=self._TOTAL_EPOCHS)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].id == "s1"
