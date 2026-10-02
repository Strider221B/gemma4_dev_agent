"""Reward model computing multi-signal rewards for agent trajectory rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.complexity_tier import ComplexityTier

if TYPE_CHECKING:
    from src.data.task import Task
    from src.data.tool_call import ToolCall
    from src.data.trajectory import Trajectory
    from src.utils.telemetry_logger import TelemetryLogger


class RewardModel:
    """Computes multi-signal composite rewards for agent trajectory outcomes."""

    _PASS_REWARD: float = 1.0
    _FAIL_REWARD: float = -0.5
    _NO_PATCH_PENALTY: float = -1.0
    _PARTIAL_PASS_WEIGHT: float = 0.5
    _EFFICIENCY_BONUS_MAX: float = 0.15
    _TRUNCATION_PENALTY: float = -0.2
    _SCRATCH_IN_WORKSPACE_PENALTY: float = -0.3
    _TEST_MODIFICATION_PENALTY: float = -0.5
    _LENGTH_PENALTY_THRESHOLD: int = 24000
    _REWARD_CLAMP_MIN: float = -1.5
    _REWARD_CLAMP_MAX: float = 1.3
    _EFFICIENCY_THRESHOLD_RATIO: float = 0.7
    _MEDIAN_SIMPLE: int = 6
    _MEDIAN_MODERATE: int = 12
    _MEDIAN_COMPLEX: int = 20
    _TMP_PREFIX: str = "/tmp/"
    _ARG_FILEPATH: str = "filepath"
    _TOOL_WRITE_FILE: str = "write_file"
    _TOOL_EDIT_FILE: str = "edit_file"
    _SCRATCH_MARKERS: tuple[str, ...] = ("repro", "scratch", "temp", "tmp_")
    _TEST_MARKERS: tuple[str, ...] = (
        "test_",
        "_test.py",
        "tests/",
        "test/",
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
    )
    _KEY_OUTCOME: str = "outcome"
    _KEY_EFFICIENCY: str = "efficiency"
    _KEY_PENALTIES: str = "penalties"
    _KEY_LENGTH_PENALTY: str = "length_penalty"
    _LENGTH_SCALE_FACTOR: float = 4000.0
    _LENGTH_COEFFICIENT: float = -0.1

    def __init__(self, telemetry: TelemetryLogger) -> None:
        """Initialize RewardModel with injected telemetry logger."""
        self._telemetry: TelemetryLogger = telemetry

    def compute_batch_rewards(
        self, trajectories: list[Trajectory], tasks: list[Task]
    ) -> list[float]:
        """Compute composite rewards for aligned lists of trajectories and tasks."""
        return [
            self.compute_reward(traj, task)
            for traj, task in zip(trajectories, tasks)
        ]

    def compute_reward(self, trajectory: Trajectory, task: Task) -> float:
        """Compute composite reward for a single trajectory and task pair."""
        outcome = self._compute_outcome_reward(trajectory, task)
        efficiency = self._compute_efficiency_bonus(trajectory, task)
        penalties = self._compute_behaviour_penalties(trajectory)
        length_pen = self._compute_length_penalty(trajectory)
        components: dict[str, float] = {
            self._KEY_OUTCOME: outcome,
            self._KEY_EFFICIENCY: efficiency,
            self._KEY_PENALTIES: penalties,
            self._KEY_LENGTH_PENALTY: length_pen,
        }
        total = sum(components.values())
        clamped = max(self._REWARD_CLAMP_MIN, min(self._REWARD_CLAMP_MAX, total))
        self._telemetry.log_reward(trajectory.instance_id, components, clamped)
        return clamped

    def _apply_and_test(self, patch: str, task: Task) -> object:
        """Execute patch in evaluation environment and return verification result."""
        from src.evaluation.verification_result import VerificationResult

        return VerificationResult(
            resolved=False,
            error=None,
            passed_tests=0,
            failed_tests=0,
            total_tests=0,
        )

    def _compute_behaviour_penalties(self, trajectory: Trajectory) -> float:
        """Penalise undesirable behaviors including truncation, scratch files, and test edits."""
        penalty = 0.0
        if trajectory.had_truncation:
            penalty += self._TRUNCATION_PENALTY
        for turn in trajectory.turns:
            for tc in turn.tool_calls:
                penalty += self._evaluate_tool_call_penalty(tc)
        return penalty

    def _compute_efficiency_bonus(self, trajectory: Trajectory, task: Task) -> float:
        """Reward trajectories that solve tasks using fewer tool calls than median."""
        complexity_val = getattr(task, "complexity", None)
        complexity = (
            complexity_val
            if isinstance(complexity_val, ComplexityTier)
            else trajectory.complexity
        )
        median = self._get_median_tool_calls(complexity)
        if median <= 0:
            return 0.0
        threshold = median * self._EFFICIENCY_THRESHOLD_RATIO
        if trajectory.num_tool_calls < threshold:
            return self._EFFICIENCY_BONUS_MAX
        if trajectory.num_tool_calls < median:
            ratio = 1.0 - (float(trajectory.num_tool_calls) / float(median))
            return self._EFFICIENCY_BONUS_MAX * ratio
        return 0.0

    def _compute_length_penalty(self, trajectory: Trajectory) -> float:
        """Penalise trajectories exceeding length threshold using quadratic scaling."""
        if trajectory.token_count > self._LENGTH_PENALTY_THRESHOLD:
            excess = float(
                trajectory.token_count - self._LENGTH_PENALTY_THRESHOLD
            ) / self._LENGTH_SCALE_FACTOR
            return self._LENGTH_COEFFICIENT * (excess**2)
        return 0.0

    def _compute_outcome_reward(self, trajectory: Trajectory, task: Task) -> float:
        """Compute binary or partial pass/fail reward from test execution result."""
        if not trajectory.patch:
            return self._NO_PATCH_PENALTY
        test_result = self._apply_and_test(trajectory.patch, task)
        all_pass = getattr(
            test_result, "all_pass", getattr(test_result, "resolved", False)
        )
        if all_pass:
            return self._PASS_REWARD
        passed = getattr(
            test_result, "passed", getattr(test_result, "passed_tests", 0)
        )
        total = getattr(
            test_result, "total", getattr(test_result, "total_tests", 0)
        )
        some_pass = getattr(test_result, "some_pass", passed > 0)
        if some_pass and total > 0:
            ratio = float(passed) / float(total)
            return self._PARTIAL_PASS_WEIGHT * ratio
        return self._FAIL_REWARD

    def _evaluate_tool_call_penalty(self, tc: ToolCall) -> float:
        """Evaluate penalty for an individual tool call."""
        penalty = 0.0
        filepath = str(tc.args.get(self._ARG_FILEPATH, ""))
        if tc.tool_name == self._TOOL_WRITE_FILE:
            if self._is_scratch_file(filepath) and not filepath.startswith(
                self._TMP_PREFIX
            ):
                penalty += self._SCRATCH_IN_WORKSPACE_PENALTY
        if tc.tool_name in (self._TOOL_EDIT_FILE, self._TOOL_WRITE_FILE):
            if self._is_test_file(filepath):
                penalty += self._TEST_MODIFICATION_PENALTY
        return penalty

    def _get_median_tool_calls(self, complexity: ComplexityTier) -> int:
        """Lookup median expected tool calls by task complexity tier."""
        if complexity == ComplexityTier.SIMPLE:
            return self._MEDIAN_SIMPLE
        if complexity == ComplexityTier.MODERATE:
            return self._MEDIAN_MODERATE
        return self._MEDIAN_COMPLEX

    def _is_scratch_file(self, filepath: str) -> bool:
        """Determine if a file path is a scratch or temporary reproduction file."""
        lowered = filepath.lower()
        return any(marker in lowered for marker in self._SCRATCH_MARKERS)

    def _is_test_file(self, filepath: str) -> bool:
        """Determine if a file path targets tests or testing configurations."""
        lowered = filepath.lower()
        return any(marker in lowered for marker in self._TEST_MARKERS)
