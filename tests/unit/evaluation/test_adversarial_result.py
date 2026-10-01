"""Unit tests for AdversarialResult data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.adversarial_result import AdversarialResult


class TestAdversarialResult:
    """Test suite for AdversarialResult frozen dataclass."""

    _PEER_CV: float = 0.52
    _BASELINE_CV: float = 0.44
    _IMPROVEMENT: float = 0.08
    _CV_LB_GAP: float = 0.06
    _IS_SIGNIFICANT: bool = True
    _FOLD_KEY: str = "fold_0"
    _DELTA_KEY: str = "delta"
    _DELTA_VAL: float = 0.08
    _MODIFIED_IMPROVEMENT: float = 0.10

    def _create_result(self) -> AdversarialResult:
        """Helper to create an AdversarialResult instance."""
        return AdversarialResult(
            peer_cv_score=self._PEER_CV,
            baseline_cv_score=self._BASELINE_CV,
            improvement=self._IMPROVEMENT,
            cv_lb_gap=self._CV_LB_GAP,
            is_significant=self._IS_SIGNIFICANT,
            per_fold_comparison={self._FOLD_KEY: {self._DELTA_KEY: self._DELTA_VAL}},
        )

    def test_adversarial_result_creation(self) -> None:
        """Verify AdversarialResult attributes."""
        res = self._create_result()
        assert res.peer_cv_score == self._PEER_CV
        assert res.baseline_cv_score == self._BASELINE_CV
        assert res.improvement == self._IMPROVEMENT
        assert res.cv_lb_gap == self._CV_LB_GAP
        assert res.is_significant == self._IS_SIGNIFICANT
        assert res.per_fold_comparison[self._FOLD_KEY][self._DELTA_KEY] == self._DELTA_VAL

    def test_adversarial_result_is_frozen(self) -> None:
        """Verify modifying field on AdversarialResult raises FrozenInstanceError."""
        res = self._create_result()
        with pytest.raises(FrozenInstanceError):
            res.improvement = self._MODIFIED_IMPROVEMENT
