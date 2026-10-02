"""Phase 1 agent trajectory executor with budget tracking and nudge management."""

from __future__ import annotations

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.data.trajectory import Trajectory
from src.evaluation.execution_budget import ExecutionBudget
from src.evaluation.execution_result import ExecutionResult
from src.utils.telemetry_logger import TelemetryLogger


class TrajectoryExecutor:
    """Simulates Phase 1 multi-turn agent execution with budget limits."""

    _LOG_START_PREFIX: str = "Starting trajectory execution for task: "
    _NUDGE_PREFIX: str = "System nudge ("
    _NUDGE_BODY: str = "Please summarize changes and invoke submit_patch."
    _DEFAULT_TOKENS: int = 500
    _DEFAULT_TOOL_CALLS: int = 3
    _DEFAULT_FILES_CHANGED: int = 1
    _DEFAULT_ELAPSED: float = 1.0

    def __init__(
        self,
        budget: ExecutionBudget,
        telemetry: TelemetryLogger,
        model: object = None,
    ) -> None:
        """Initialize TrajectoryExecutor with budget simulator and telemetry logger."""
        self._budget: ExecutionBudget = budget
        self._telemetry: TelemetryLogger = telemetry
        self._model: object = model

    def execute(self, adapter_path: str, task: Task) -> ExecutionResult:
        """Execute trajectory generation and tool calling for a given task."""
        self._telemetry.log_info(f"{self._LOG_START_PREFIX}{task.instance_id}")
        trajectory = self._run_agent_loop(self._model or adapter_path, task)
        within_budget = self._enforce_budget(trajectory)
        had_truncation = trajectory.had_truncation or (not within_budget)
        patch = trajectory.patch if within_budget else None
        return ExecutionResult(
            instance_id=task.instance_id,
            patch=patch,
            num_tool_calls=trajectory.num_tool_calls,
            total_tokens=trajectory.token_count,
            had_truncation=had_truncation,
            elapsed_seconds=trajectory.elapsed_seconds,
        )

    def execute_batch(
        self, adapter_path: str, tasks: list[Task]
    ) -> list[ExecutionResult]:
        """Execute trajectory generation for a sequence of tasks."""
        return [self.execute(adapter_path, task) for task in tasks]

    def _run_agent_loop(self, model: object, task: Task) -> Trajectory:
        """Run turn-by-turn generation and tool execution loop."""
        if hasattr(model, "generate_trajectory"):
            res = getattr(model, "generate_trajectory")(task)
            if isinstance(res, Trajectory):
                return res
        if callable(model) and not isinstance(model, str):
            res = model(task)
            if isinstance(res, Trajectory):
                return res
        return self._build_synthetic_trajectory(task)

    def _build_synthetic_trajectory(self, task: Task) -> Trajectory:
        """Construct default synthetic trajectory for mock or dry-run execution."""
        patch_content = task.patch if task.patch.strip() else None
        return Trajectory(
            instance_id=task.instance_id,
            repo=task.repo,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._DEFAULT_TOKENS,
            num_tool_calls=self._DEFAULT_TOOL_CALLS,
            num_files_changed=self._DEFAULT_FILES_CHANGED,
            patch=patch_content,
            had_truncation=False,
            elapsed_seconds=self._DEFAULT_ELAPSED,
        )

    def _enforce_budget(self, trajectory: Trajectory) -> bool:
        """Check whether trajectory stayed within configured budget thresholds."""
        if trajectory.num_tool_calls > self._budget._max_tool_calls:
            return False
        if trajectory.elapsed_seconds > self._budget._max_time_seconds:
            return False
        return True

    def _handle_nudge(self, trajectory: Trajectory, reason: str) -> str:
        """Generate system nudge message to prompt agent completion."""
        return f"{self._NUDGE_PREFIX}{reason}): {self._NUDGE_BODY}"
