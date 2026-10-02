"""Unit tests for TrajectoryExecutor."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.data.trajectory import Trajectory
from src.evaluation.execution_budget import ExecutionBudget
from src.evaluation.trajectory_executor import TrajectoryExecutor
from src.utils.telemetry_logger import TelemetryLogger


class TestTrajectoryExecutor:
    """Test suite for TrajectoryExecutor Phase 1 simulation and budget enforcement."""

    _MOCK_COMMIT: str = "0123456789abcdef"
    _MOCK_CREATED: str = "2026-01-01T00:00:00Z"
    _SAMPLE_PATCH: str = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"

    def test_execute_returns_execution_result(self) -> None:
        """Verify execute returns ExecutionResult with synthetic trajectory attributes."""
        budget = ExecutionBudget()
        telemetry = MagicMock(spec=TelemetryLogger)
        executor = TrajectoryExecutor(budget=budget, telemetry=telemetry)
        task = self._create_task("task_01", patch=self._SAMPLE_PATCH)
        res = executor.execute("dummy/adapter", task)
        assert res.instance_id == "task_01"
        assert res.patch == self._SAMPLE_PATCH
        assert res.num_tool_calls > 0
        assert res.had_truncation is False

    def test_execute_batch_returns_results_for_all_tasks(self) -> None:
        """Verify execute_batch processes all tasks in list."""
        budget = ExecutionBudget()
        telemetry = MagicMock(spec=TelemetryLogger)
        executor = TrajectoryExecutor(budget=budget, telemetry=telemetry)
        tasks = [self._create_task("task_01"), self._create_task("task_02")]
        results = executor.execute_batch("dummy/adapter", tasks)
        assert len(results) == 2
        assert results[0].instance_id == "task_01"
        assert results[1].instance_id == "task_02"

    def test_execute_with_custom_model_callable(self) -> None:
        """Verify executor uses injected model callable when provided."""
        budget = ExecutionBudget()
        telemetry = MagicMock(spec=TelemetryLogger)
        custom_traj = Trajectory(
            instance_id="custom_01",
            repo="tiangolo/fastapi",
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=120,
            num_tool_calls=2,
            num_files_changed=1,
            patch="diff patch",
            had_truncation=False,
            elapsed_seconds=0.5,
        )
        mock_model = MagicMock()
        mock_model.generate_trajectory.return_value = custom_traj
        executor = TrajectoryExecutor(budget=budget, telemetry=telemetry, model=mock_model)
        task = self._create_task("custom_01")
        res = executor.execute("dummy/adapter", task)
        assert res.patch == "diff patch"
        assert res.total_tokens == 120

    def test_execute_budget_exceeded_sets_truncation(self) -> None:
        """Verify exceeding tool budget marks had_truncation True and drops patch."""
        budget = ExecutionBudget(overrides={"max_tool_calls": 1})
        telemetry = MagicMock(spec=TelemetryLogger)
        executor = TrajectoryExecutor(budget=budget, telemetry=telemetry)
        task = self._create_task("task_over_budget", patch=self._SAMPLE_PATCH)
        res = executor.execute("dummy/adapter", task)
        assert res.had_truncation is True
        assert res.patch is None

    def test_handle_nudge_formats_message(self) -> None:
        """Verify _handle_nudge formats reminder message with reason."""
        budget = ExecutionBudget()
        telemetry = MagicMock(spec=TelemetryLogger)
        executor = TrajectoryExecutor(budget=budget, telemetry=telemetry)
        task = self._create_task("task_nudge")
        traj = executor._build_synthetic_trajectory(task)
        nudge = executor._handle_nudge(traj, "approaching tool limit")
        assert "approaching tool limit" in nudge
        assert "submit_patch" in nudge

    def test_execute_with_plain_callable(self) -> None:
        """Verify executor with plain callable function."""
        budget = ExecutionBudget()
        telemetry = MagicMock(spec=TelemetryLogger)
        custom_traj = Trajectory(
            instance_id="call_01",
            repo="tiangolo/fastapi",
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=50,
            num_tool_calls=1,
            num_files_changed=1,
            patch="diff",
            had_truncation=False,
            elapsed_seconds=0.2,
        )
        executor = TrajectoryExecutor(
            budget=budget, telemetry=telemetry, model=lambda t: custom_traj
        )
        task = self._create_task("call_01")
        res = executor.execute("adapter", task)
        assert res.patch == "diff"

    def test_execute_time_budget_exceeded(self) -> None:
        """Verify exceeding time budget marks truncation."""
        budget = ExecutionBudget(overrides={"max_time_minutes": 0.00001})
        budget._max_time_seconds = 0.0001
        telemetry = MagicMock(spec=TelemetryLogger)
        executor = TrajectoryExecutor(budget=budget, telemetry=telemetry)
        task = self._create_task("time_task", patch=self._SAMPLE_PATCH)
        res = executor.execute("adapter", task)
        assert res.had_truncation is True

    def _create_task(self, instance_id: str, patch: str = "") -> Task:
        """Helper to create test Task."""
        return Task(
            instance_id=instance_id,
            repo="tiangolo/fastapi",
            base_commit=self._MOCK_COMMIT,
            problem_statement="Description",
            hints_text="Hints",
            patch=patch,
            test_patch="",
            created_at=self._MOCK_CREATED,
        )
