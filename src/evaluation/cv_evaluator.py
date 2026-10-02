"""Cross-validation evaluator orchestrating all-fold and single-fold benchmarking."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.evaluation.cv_fold import CVFold
from src.evaluation.cv_report import CVReport
from src.evaluation.cv_result import CVResult
from src.evaluation.cv_splitter import CVSplitter
from src.evaluation.execution_result import ExecutionResult
from src.evaluation.metric_tracker import MetricTracker
from src.evaluation.task_result import TaskResult
from src.evaluation.trajectory_executor import TrajectoryExecutor
from src.evaluation.verification_result import VerificationResult
from src.evaluation.verification_runner import VerificationRunner
from src.utils.telemetry_logger import TelemetryLogger


class CVEvaluator:
    """End-to-end cross-validation engine executing Phase 1 and 2 pipelines."""

    _MIN_RESOLUTION_RATE: float = 0.10
    _CRITICAL_RATE_THRESHOLD: float = 0.20
    _REPO_WEAKNESS_RATIO: float = 0.5
    _COMPLEXITY_GAP_RATIO: float = 0.3
    _TRUNCATION_ALERT_THRESHOLD: float = 0.05
    _HIGH_TOOL_USAGE_THRESHOLD: float = 60.0
    _MIXED_REPO_LABEL: str = "mixed"
    _DEFAULT_QUICK_FOLD_IDX: int = -1
    _EMPTY_ERROR_MSG: str = "No patch generated"

    _REC_LOW_RATE: str = (
        "CRITICAL: Resolution rate below 20%. Focus on SFT data quality "
        "and tool-call syntax accuracy before RL training."
    )
    _REC_WEAK_REPO_PREFIX: str = "WARNING: Weak repository resolution in "
    _REC_WEAK_REPO_SUFFIX: str = ". Add more diverse training trajectories for this repo type."
    _REC_COMPLEXITY_GAP: str = (
        "WARNING: Complex tasks resolve at <30% of simple task rate. "
        "Increase multi-file trajectory training and raise tool budget."
    )
    _REC_HIGH_TRUNCATION: str = (
        "WARNING: Truncation rate exceeds 5% threshold. "
        "Reduce thinking_budget or split edit_file payloads."
    )
    _REC_HIGH_TOOL_USAGE: str = (
        "WARNING: Average tool usage is high (>60 calls). "
        "Add efficiency bonus to RL reward signal."
    )

    _METRIC_KEY_PHASE: str = "phase"
    _METRIC_KEY_FOLD: str = "fold"
    _METRIC_KEY_VAL_REPO: str = "val_repo"
    _METRIC_KEY_RESOLUTION_RATE: str = "resolution_rate"
    _METRIC_KEY_AVG_TOOL_CALLS: str = "avg_tool_calls"
    _METRIC_KEY_AVG_TOKENS: str = "avg_tokens"
    _METRIC_KEY_TRUNCATION_RATE: str = "truncation_rate"
    _PHASE_CV: str = "cv"
    _LOG_EVAL_FOLD_PREFIX: str = "Evaluating fold "
    _LOG_VAL_REPO_PREFIX: str = " (val_repo="

    def __init__(
        self,
        splitter: CVSplitter,
        executor: TrajectoryExecutor,
        verifier: VerificationRunner,
        tracker: MetricTracker,
        telemetry: TelemetryLogger,
    ) -> None:
        """Initialize evaluator with pipeline runner dependencies."""
        self._splitter: CVSplitter = splitter
        self._executor: TrajectoryExecutor = executor
        self._verifier: VerificationRunner = verifier
        self._tracker: MetricTracker = tracker
        self._telemetry: TelemetryLogger = telemetry

    def evaluate_all_folds(
        self, adapter_path: str, tasks: list[Task], version: str
    ) -> CVReport:
        """Execute full cross-validation over all K folds and aggregate report."""
        folds = self._splitter.create_splits(tasks)
        fold_results: list[CVResult] = []
        for fold in folds:
            msg = (
                f"{self._LOG_EVAL_FOLD_PREFIX}{fold.fold_idx}"
                f"{self._LOG_VAL_REPO_PREFIX}{fold.val_repo})"
            )
            self._telemetry.log_info(msg)
            result = self.evaluate_fold(adapter_path, fold, tasks)
            fold_results.append(result)
            self._log_fold_metrics(fold, result)
        report = self._aggregate_results(fold_results, version)
        self._tracker.record(version, report.aggregate_resolution_rate, lb_score=None)
        return report

    def evaluate_fold(
        self, adapter_path: str, fold: CVFold, tasks: list[Task]
    ) -> CVResult:
        """Evaluate agent checkpoint on a single hold-out validation fold."""
        val_id_set = set(fold.val_ids)
        val_tasks = [t for t in tasks if t.instance_id in val_id_set]
        task_results = [self._evaluate_task(adapter_path, t) for t in val_tasks]
        resolved_count = sum(1 for r in task_results if r.resolved)
        total_count = len(task_results)
        rate = resolved_count / total_count if total_count > 0 else 0.0
        return CVResult(
            fold_idx=fold.fold_idx,
            val_repo=fold.val_repo,
            resolution_rate=rate,
            resolved_count=resolved_count,
            total_count=total_count,
            per_task_results=task_results,
            avg_tool_calls=self._mean([float(r.tool_calls_used) for r in task_results]),
            avg_tokens=self._mean([float(r.tokens_used) for r in task_results]),
            truncation_rate=self._mean([float(r.had_truncation) for r in task_results]),
            per_complexity={
                tier.value: self._compute_tier_rate(task_results, tier)
                for tier in ComplexityTier
            },
        )

    def quick_evaluate(
        self, adapter_path: str, task_ids: list[str], tasks: list[Task]
    ) -> CVResult:
        """Perform rapid evaluation on an ad-hoc subset of task identifiers."""
        target_ids = set(task_ids)
        subset = [t for t in tasks if t.instance_id in target_ids]
        task_results = [self._evaluate_task(adapter_path, t) for t in subset]
        resolved = sum(1 for r in task_results if r.resolved)
        total = len(task_results)
        rate = resolved / total if total > 0 else 0.0
        return CVResult(
            fold_idx=self._DEFAULT_QUICK_FOLD_IDX,
            val_repo=self._MIXED_REPO_LABEL,
            resolution_rate=rate,
            resolved_count=resolved,
            total_count=total,
            per_task_results=task_results,
            avg_tool_calls=self._mean([float(r.tool_calls_used) for r in task_results]),
            avg_tokens=self._mean([float(r.tokens_used) for r in task_results]),
            truncation_rate=self._mean([float(r.had_truncation) for r in task_results]),
            per_complexity={
                tier.value: self._compute_tier_rate(task_results, tier)
                for tier in ComplexityTier
            },
        )

    def _evaluate_task(self, adapter_path: str, task: Task) -> TaskResult:
        """Execute Phase 1 generation and Phase 2 verification for a single task."""
        exec_res = self._executor.execute(adapter_path=adapter_path, task=task)
        verif_res = self._run_task_verification(exec_res, task)
        patch_len = len(exec_res.patch) if exec_res.patch else 0
        return TaskResult(
            instance_id=task.instance_id,
            repo=task.repo,
            complexity=self._splitter._classify_complexity(task),
            resolved=verif_res.resolved,
            patch_size=patch_len,
            tool_calls_used=exec_res.num_tool_calls,
            tokens_used=exec_res.total_tokens,
            had_truncation=exec_res.had_truncation,
            time_seconds=exec_res.elapsed_seconds,
            error=verif_res.error,
        )

    def _run_task_verification(
        self, exec_res: ExecutionResult, task: Task
    ) -> VerificationResult:
        """Run verification if patch is available, or return failed result."""
        if exec_res.patch and exec_res.patch.strip():
            return self._verifier.verify(patch=exec_res.patch, task=task)
        return VerificationResult(resolved=False, error=self._EMPTY_ERROR_MSG)

    def _aggregate_results(
        self, fold_results: list[CVResult], version: str
    ) -> CVReport:
        """Aggregate fold evaluation outcomes into structured CV report."""
        total_resolved = sum(r.resolved_count for r in fold_results)
        total_tasks = sum(r.total_count for r in fold_results)
        rate = total_resolved / total_tasks if total_tasks > 0 else 0.0
        per_repo = {r.val_repo: r.resolution_rate for r in fold_results}
        all_tasks = [res for fold in fold_results for res in fold.per_task_results]
        per_comp = {
            tier.value: self._compute_tier_rate(all_tasks, tier)
            for tier in ComplexityTier
        }
        signal = self._tracker.detect_trend()
        recs = self._generate_recommendations(rate, per_repo, per_comp, fold_results)
        return CVReport(
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            fold_results=fold_results,
            aggregate_resolution_rate=rate,
            per_repo_rates=per_repo,
            per_complexity_rates=per_comp,
            overfitting_signal=signal.value,
            recommendations=recs,
            total_resolved=total_resolved,
            total_tasks=total_tasks,
        )

    def _generate_recommendations(
        self,
        rate: float,
        per_repo: dict[str, float],
        per_complexity: dict[str, float],
        fold_results: list[CVResult],
    ) -> list[str]:
        """Generate actionable diagnostic recommendations from performance discrepancies."""
        recs: list[str] = []
        if rate < self._CRITICAL_RATE_THRESHOLD:
            recs.append(self._REC_LOW_RATE)
        self._check_repo_weakness(recs, rate, per_repo)
        self._check_complexity_gap(recs, per_complexity)
        self._check_truncation(recs, fold_results)
        self._check_tool_usage(recs, fold_results)
        return recs

    def _check_repo_weakness(
        self, recs: list[str], rate: float, per_repo: dict[str, float]
    ) -> None:
        """Check for single-repository resolution underperformance."""
        if not per_repo:
            return
        weakest = min(per_repo, key=lambda k: per_repo[k])
        if per_repo[weakest] < rate * self._REPO_WEAKNESS_RATIO:
            recs.append(f"{self._REC_WEAK_REPO_PREFIX}{weakest}{self._REC_WEAK_REPO_SUFFIX}")

    def _check_complexity_gap(
        self, recs: list[str], per_complexity: dict[str, float]
    ) -> None:
        """Check for resolution rate drop between simple and complex tasks."""
        complex_rate = per_complexity.get(ComplexityTier.COMPLEX.value, 0.0)
        simple_rate = per_complexity.get(ComplexityTier.SIMPLE.value, 0.0)
        if complex_rate < simple_rate * self._COMPLEXITY_GAP_RATIO:
            recs.append(self._REC_COMPLEXITY_GAP)

    def _check_truncation(
        self, recs: list[str], fold_results: list[CVResult]
    ) -> None:
        """Check if trajectory truncation rate exceeds threshold."""
        avg_trunc = self._mean([r.truncation_rate for r in fold_results])
        if avg_trunc > self._TRUNCATION_ALERT_THRESHOLD:
            recs.append(self._REC_HIGH_TRUNCATION)

    def _check_tool_usage(
        self, recs: list[str], fold_results: list[CVResult]
    ) -> None:
        """Check if average tool call count exceeds efficiency budget."""
        avg_tools = self._mean([r.avg_tool_calls for r in fold_results])
        if avg_tools > self._HIGH_TOOL_USAGE_THRESHOLD:
            recs.append(self._REC_HIGH_TOOL_USAGE)

    def _compute_tier_rate(
        self, results: list[TaskResult], tier: ComplexityTier
    ) -> float:
        """Calculate task resolution rate for a specific complexity tier."""
        matching = [r for r in results if r.complexity == tier]
        if not matching:
            return 0.0
        resolved = sum(1 for r in matching if r.resolved)
        return resolved / len(matching)

    def _mean(self, values: list[float]) -> float:
        """Compute arithmetic mean safely handling empty collections."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _log_fold_metrics(self, fold: CVFold, result: CVResult) -> None:
        """Emit telemetry log for completed fold evaluation."""
        self._telemetry.log_metrics({
            self._METRIC_KEY_PHASE: self._PHASE_CV,
            self._METRIC_KEY_FOLD: fold.fold_idx,
            self._METRIC_KEY_VAL_REPO: fold.val_repo,
            self._METRIC_KEY_RESOLUTION_RATE: result.resolution_rate,
            self._METRIC_KEY_AVG_TOOL_CALLS: result.avg_tool_calls,
            self._METRIC_KEY_AVG_TOKENS: result.avg_tokens,
            self._METRIC_KEY_TRUNCATION_RATE: result.truncation_rate,
        })
