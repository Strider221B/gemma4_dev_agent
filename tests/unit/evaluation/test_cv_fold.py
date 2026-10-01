"""Unit tests for CVFold data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.cv_fold import CVFold


class TestCVFold:
    """Test suite for CVFold frozen dataclass."""

    _FOLD_IDX: int = 1
    _TRAIN_IDS: list[str] = ["task-1", "task-2"]
    _VAL_IDS: list[str] = ["task-3"]
    _VAL_REPO: str = "django/django"
    _COMPLEXITY_DIST: dict[str, int] = {"SIMPLE": 1, "COMPLEX": 2}
    _MODIFIED_VAL_REPO: str = "astropy/astropy"

    def _create_fold(self) -> CVFold:
        """Helper to create a standard CVFold instance."""
        return CVFold(
            fold_idx=self._FOLD_IDX,
            train_ids=self._TRAIN_IDS,
            val_ids=self._VAL_IDS,
            val_repo=self._VAL_REPO,
            complexity_distribution=self._COMPLEXITY_DIST,
        )

    def test_cv_fold_creation(self) -> None:
        """Verify CVFold attributes are correctly initialized."""
        fold = self._create_fold()
        assert fold.fold_idx == self._FOLD_IDX
        assert fold.train_ids == self._TRAIN_IDS
        assert fold.val_ids == self._VAL_IDS
        assert fold.val_repo == self._VAL_REPO
        assert fold.complexity_distribution == self._COMPLEXITY_DIST

    def test_cv_fold_is_frozen(self) -> None:
        """Verify modifying field on frozen CVFold raises FrozenInstanceError."""
        fold = self._create_fold()
        with pytest.raises(FrozenInstanceError):
            fold.val_repo = self._MODIFIED_VAL_REPO
