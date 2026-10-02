"""Unit tests for CVEvaluator."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.evaluation.cv_evaluator import CVEvaluator
from src.evaluation.cv_fold import CVFold
from src.evaluation.cv_result import CVResult
from src.evaluation.cv_splitter import CVSplitter
from src.evaluation.execution_result import ExecutionResult
from src.evaluation.metric_tracker import MetricTracker
from src.evaluation.overfitting_signal import OverfittingSignal
from src.evaluation.trajectory_executor import TrajectoryExecutor
from src.evaluation.verification_result import VerificationResult
from src.evaluation.verification_runner import VerificationRunner
from src.utils.telemetry_logger import TelemetryLogger


class TestCVEvaluator:
    """Test suite for CVEvaluator cross-validation pipeline orchestration."""

    _MOCK_COMMIT: str = "1234567890abcdef"
    _MOCK_CREATED: str = "2026-01-01T00:00:00Z"
    _ADAPTER_PATH: str = "/kaggle/working/adapter"
    _VERSION: str = "0.2.0"

    def test_evaluate_fold_returns_cv_result(self) -> None:
        """Verify evaluate_fold executes tasks and computes fold resolution metrics."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        fold = CVFold(
            fold_idx=0,
            train_ids=["task_02"],
            val_ids=["task_01"],
            val_repo="fastapi",
            complexity_distribution={"SIMPLE": 1, "MODERATE": 0, "COMPLEX": 0},
        )
        task = self._create_task("task_01", "fastapi")
        tasks = [task]
        executor.execute.return_value = ExecutionResult(
            instance_id="task_01",
            patch="diff --git",
            num_tool_calls=5,
            total_tokens=1000,
            had_truncation=False,
            elapsed_seconds=2.0,
        )
        verifier.verify.return_value = VerificationResult(resolved=True, passed_tests=2)
        result = evaluator.evaluate_fold(self._ADAPTER_PATH, fold, tasks)
        assert isinstance(result, CVResult)
        assert result.fold_idx == 0
        assert result.resolution_rate == 1.0
        assert result.resolved_count == 1
        assert len(result.per_task_results) == 1

    def test_evaluate_fold_empty_patch_records_unresolved(self) -> None:
        """Verify task without patch is marked unresolved without invoking verifier."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        fold = CVFold(
            fold_idx=0,
            train_ids=[],
            val_ids=["task_01"],
            val_repo="fastapi",
            complexity_distribution={},
        )
        tasks = [self._create_task("task_01", "fastapi")]
        executor.execute.return_value = ExecutionResult(instance_id="task_01", patch="")
        result = evaluator.evaluate_fold(self._ADAPTER_PATH, fold, tasks)
        assert result.resolved_count == 0
        verifier.verify.assert_not_called()

    def test_evaluate_all_folds_runs_all_folds(self) -> None:
        """Verify evaluate_all_folds iterates through all folds and creates report."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        fold1 = CVFold(0, ["t2"], ["t1"], "fastapi", {})
        fold2 = CVFold(1, ["t1"], ["t2"], "rich", {})
        splitter.create_splits.return_value = [fold1, fold2]
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        tasks = [self._create_task("t1", "fastapi"), self._create_task("t2", "rich")]
        executor.execute.return_value = ExecutionResult(instance_id="dummy", patch="diff")
        verifier.verify.return_value = VerificationResult(resolved=True, passed_tests=1)
        report = evaluator.evaluate_all_folds(self._ADAPTER_PATH, tasks, self._VERSION)
        assert len(report.fold_results) == 2
        assert report.version == self._VERSION
        tracker.record.assert_called_once()
        assert telemetry.log_metrics.call_count == 2

    def test_aggregate_results_computes_weighted_rate(self) -> None:
        """Verify aggregation correctly calculates weighted aggregate rate."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        res1 = CVResult(0, "fastapi", 1.0, 2, 2, [])
        res2 = CVResult(1, "rich", 0.0, 0, 2, [])
        report = evaluator._aggregate_results([res1, res2], self._VERSION)
        assert report.total_resolved == 2
        assert report.total_tasks == 4
        assert report.aggregate_resolution_rate == 0.5
        assert report.per_repo_rates["fastapi"] == 1.0
        assert report.per_repo_rates["rich"] == 0.0

    def test_generate_recommendations_low_rate(self) -> None:
        """Verify recommendation generated when resolution rate is below 20%."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        recs = evaluator._generate_recommendations(0.15, {"repo1": 0.15}, {}, [])
        assert any("Resolution rate below 20%" in r for r in recs)

    def test_generate_recommendations_high_truncation(self) -> None:
        """Verify recommendation generated when truncation rate exceeds 5%."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        fold_res = [CVResult(0, "r1", 0.3, 3, 10, [], truncation_rate=0.08)]
        recs = evaluator._generate_recommendations(0.3, {"r1": 0.3}, {}, fold_res)
        assert any("Truncation rate" in r for r in recs)

    def test_generate_recommendations_weak_repo_and_tools_and_gap(self) -> None:
        """Verify recommendations for weak repo, tool usage, and complexity gap."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        per_repo = {"fastapi": 0.6, "httpx": 0.1}
        per_comp = {"SIMPLE": 0.8, "COMPLEX": 0.1}
        fold_res = [CVResult(0, "r1", 0.4, 4, 10, [], avg_tool_calls=75.0)]
        recs = evaluator._generate_recommendations(0.4, per_repo, per_comp, fold_res)
        assert any("Weak repository" in r for r in recs)
        assert any("Complex tasks resolve at <30%" in r for r in recs)
        assert any("Average tool usage is high" in r for r in recs)

    def test_quick_evaluate_subset(self) -> None:
        """Verify quick_evaluate evaluates only requested task IDs."""
        splitter, executor, verifier, tracker, telemetry = self._build_mocks()
        evaluator = CVEvaluator(splitter, executor, verifier, tracker, telemetry)
        t1 = self._create_task("t1", "repo1")
        t2 = self._create_task("t2", "repo2")
        executor.execute.return_value = ExecutionResult("t1", patch="patch")
        verifier.verify.return_value = VerificationResult(resolved=True, passed_tests=1)
        res = evaluator.quick_evaluate(self._ADAPTER_PATH, ["t1"], [t1, t2])
        assert res.total_count == 1
        assert res.resolved_count == 1
        assert res.val_repo == "mixed"
        assert res.fold_idx == -1

    def _build_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
        """Construct mock dependencies for CVEvaluator."""
        splitter = MagicMock(spec=CVSplitter)
        splitter._classify_complexity.return_value = ComplexityTier.SIMPLE
        executor = MagicMock(spec=TrajectoryExecutor)
        verifier = MagicMock(spec=VerificationRunner)
        tracker = MagicMock(spec=MetricTracker)
        tracker.detect_trend.return_value = OverfittingSignal.NONE
        telemetry = MagicMock(spec=TelemetryLogger)
        return splitter, executor, verifier, tracker, telemetry

    def _create_task(self, instance_id: str, repo: str) -> Task:
        """Helper to create dummy Task."""
        return Task(
            instance_id=instance_id,
            repo=repo,
            base_commit=self._MOCK_COMMIT,
            problem_statement="Description",
            hints_text="Hints",
            patch="",
            test_patch="",
            created_at=self._MOCK_CREATED,
        )
