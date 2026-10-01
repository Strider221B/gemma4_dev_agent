"""Unit tests for CVResult, CVReport, VerificationResult, and ExecutionResult."""

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.cv_report import CVReport
from src.evaluation.cv_result import CVResult
from src.evaluation.execution_result import ExecutionResult
from src.evaluation.task_result import TaskResult
from src.evaluation.verification_result import VerificationResult


class TestCVResult:
    """Test suite for evaluation result dataclasses."""

    _VAL_REPO: str = "astropy/astropy"
    _RESOLUTION_RATE: float = 0.45
    _VERSION: str = "v1.0"
    _TIMESTAMP: str = "2026-10-01T00:00:00Z"
    _OVERFITTING_SIGNAL: str = "NONE"
    _SAMPLE_RECOMMENDATION: str = "Increase SFT epochs"
    _INSTANCE_ID: str = "repo__task-1"
    _MODIFIED_RATE: float = 0.50

    def _create_task_result(self) -> TaskResult:
        """Helper to create a sample TaskResult."""
        return TaskResult(instance_id=self._INSTANCE_ID, repo=self._VAL_REPO, resolved=True)

    def _create_cv_result(self) -> CVResult:
        """Helper to create a sample CVResult."""
        return CVResult(
            fold_idx=0,
            val_repo=self._VAL_REPO,
            resolution_rate=self._RESOLUTION_RATE,
            resolved_count=9,
            total_count=20,
            per_task_results=[self._create_task_result()],
        )

    def test_cv_result_creation(self) -> None:
        """Verify CVResult initialization and field values."""
        cv_result = self._create_cv_result()
        assert cv_result.fold_idx == 0
        assert cv_result.val_repo == self._VAL_REPO
        assert cv_result.resolution_rate == self._RESOLUTION_RATE
        assert cv_result.resolved_count == 9
        assert cv_result.total_count == 20
        assert len(cv_result.per_task_results) == 1
        assert cv_result.avg_tool_calls == 0.0
        assert cv_result.avg_tokens == 0.0
        assert cv_result.truncation_rate == 0.0
        assert cv_result.per_complexity is None

    def test_cv_result_is_frozen(self) -> None:
        """Verify CVResult is immutable."""
        cv_result = self._create_cv_result()
        with pytest.raises(FrozenInstanceError):
            cv_result.resolution_rate = self._MODIFIED_RATE

    def test_cv_report_creation(self) -> None:
        """Verify CVReport creation and aggregated metrics."""
        cv_result = self._create_cv_result()
        report = CVReport(
            version=self._VERSION,
            timestamp=self._TIMESTAMP,
            fold_results=[cv_result],
            aggregate_resolution_rate=self._RESOLUTION_RATE,
            per_repo_rates={self._VAL_REPO: self._RESOLUTION_RATE},
            per_complexity_rates={"SIMPLE": 0.6},
            overfitting_signal=self._OVERFITTING_SIGNAL,
            recommendations=[self._SAMPLE_RECOMMENDATION],
            total_resolved=9,
            total_tasks=20,
        )
        assert report.version == self._VERSION
        assert report.timestamp == self._TIMESTAMP
        assert len(report.fold_results) == 1
        assert report.aggregate_resolution_rate == self._RESOLUTION_RATE
        assert report.total_resolved == 9
        assert report.total_tasks == 20

    def test_verification_result_defaults(self) -> None:
        """Verify default values and custom fields of VerificationResult."""
        res_default = VerificationResult(resolved=True)
        assert res_default.resolved is True
        assert res_default.error is None
        assert res_default.passed_tests == 0
        assert res_default.failed_tests == 0
        assert res_default.total_tests == 0

        res_custom = VerificationResult(
            resolved=False,
            error="AssertionError in test_x",
            passed_tests=5,
            failed_tests=1,
            total_tests=6,
        )
        assert res_custom.resolved is False
        assert res_custom.error == "AssertionError in test_x"
        assert res_custom.passed_tests == 5
        assert res_custom.failed_tests == 1
        assert res_custom.total_tests == 6

    def test_execution_result_creation(self) -> None:
        """Verify ExecutionResult initialization and defaults."""
        exec_res = ExecutionResult(instance_id=self._INSTANCE_ID)
        assert exec_res.instance_id == self._INSTANCE_ID
        assert exec_res.patch is None
        assert exec_res.num_tool_calls == 0
        assert exec_res.total_tokens == 0
        assert exec_res.had_truncation is False
        assert exec_res.elapsed_seconds == 0.0
