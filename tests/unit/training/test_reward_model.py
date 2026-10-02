"""Unit tests for RewardModel evaluating multi-signal trajectory rewards."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.turn import Turn
from src.training.reward_model import RewardModel
from src.utils.telemetry_logger import TelemetryLogger


class TestRewardModel:
    """Test suite verifying RewardModel reward calculation and penalties."""

    _INSTANCE_ID: str = "test_task_001"
    _REPO: str = "google/gemma4"
    _COMMIT: str = "a1b2c3d"
    _PROBLEM: str = "Fix array index out of bounds in parser"
    _HINTS: str = "Check boundary conditions in parse_tree"
    _PATCH: str = "diff --git a/parser.py b/parser.py\n+ return None"
    _TEST_PATCH: str = "diff --git a/test_parser.py b/test_parser.py"
    _CREATED_AT: str = "2026-10-01T00:00:00Z"
    _HIGH_TOKEN_COUNT: int = 28000
    _NORMAL_TOKEN_COUNT: int = 5000
    _SCRATCH_WORKSPACE_PATH: str = "repro_script.py"
    _SCRATCH_TMP_PATH: str = "/tmp/repro_script.py"
    _TEST_FILE_PATH: str = "tests/test_parser.py"
    _SRC_FILE_PATH: str = "src/parser.py"

    @pytest.fixture
    def sample_task(self) -> Task:
        """Create sample Task object for tests."""
        return Task(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            base_commit=self._COMMIT,
            problem_statement=self._PROBLEM,
            hints_text=self._HINTS,
            patch=self._PATCH,
            test_patch=self._TEST_PATCH,
            created_at=self._CREATED_AT,
        )

    @pytest.fixture
    def mock_telemetry(self) -> MagicMock:
        """Create mock TelemetryLogger fixture."""
        return MagicMock(spec=TelemetryLogger)

    def test_compute_reward_full_pass_returns_positive(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify reward is positive and near 1.0 when tests completely pass."""
        reward_model = RewardModel(mock_telemetry)
        test_result = MagicMock(all_pass=True, passed=5, total=5)
        reward_model._apply_and_test = MagicMock(return_value=test_result)  # type: ignore[method-assign]
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=3,
            num_files_changed=1,
            patch=self._PATCH,
        )
        reward = reward_model.compute_reward(trajectory, sample_task)
        assert reward > 0.8
        mock_telemetry.log_reward.assert_called_once()

    def test_compute_reward_no_patch_returns_negative(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify reward applies penalty when no patch is provided."""
        reward_model = RewardModel(mock_telemetry)
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=2,
            num_files_changed=0,
            patch=None,
        )
        reward = reward_model.compute_reward(trajectory, sample_task)
        assert reward <= -0.85

    def test_compute_reward_partial_pass_returns_partial_credit(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify partial reward is awarded proportionally when some tests pass."""
        reward_model = RewardModel(mock_telemetry)
        test_result = MagicMock(all_pass=False, some_pass=True, passed=2, total=4)
        reward_model._apply_and_test = MagicMock(return_value=test_result)  # type: ignore[method-assign]
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=6,
            num_files_changed=1,
            patch=self._PATCH,
        )
        reward = reward_model.compute_reward(trajectory, sample_task)
        assert 0.0 < reward < 0.5

    def test_efficiency_bonus_few_calls(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify maximum efficiency bonus is granted when tool calls are under threshold."""
        reward_model = RewardModel(mock_telemetry)
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=2,
            num_files_changed=1,
            patch=self._PATCH,
        )
        bonus = reward_model._compute_efficiency_bonus(trajectory, sample_task)
        assert bonus == pytest.approx(0.15)

    def test_efficiency_bonus_many_calls(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify efficiency bonus is 0 when tool calls meet or exceed median."""
        reward_model = RewardModel(mock_telemetry)
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=10,
            num_files_changed=1,
            patch=self._PATCH,
        )
        bonus = reward_model._compute_efficiency_bonus(trajectory, sample_task)
        assert bonus == 0.0

    def test_truncation_penalty_applied(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify truncation penalty is assessed when trajectory flag is set."""
        reward_model = RewardModel(mock_telemetry)
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=2,
            num_files_changed=1,
            patch=self._PATCH,
            had_truncation=True,
        )
        penalty = reward_model._compute_behaviour_penalties(trajectory)
        assert penalty <= -0.2

    def test_scratch_in_workspace_penalty(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify penalty is applied when scratch files are saved in workspace root."""
        reward_model = RewardModel(mock_telemetry)
        turn = Turn(
            role="model",
            tool_calls=[
                ToolCall(
                    tool_name="write_file",
                    args={"filepath": self._SCRATCH_WORKSPACE_PATH},
                )
            ],
        )
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[turn],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=1,
            num_files_changed=1,
            patch=self._PATCH,
        )
        penalty = reward_model._compute_behaviour_penalties(trajectory)
        assert penalty == pytest.approx(-0.3)

    def test_test_modification_penalty(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify penalty is applied when agent modifies test files."""
        reward_model = RewardModel(mock_telemetry)
        turn = Turn(
            role="model",
            tool_calls=[
                ToolCall(
                    tool_name="edit_file",
                    args={"filepath": self._TEST_FILE_PATH},
                )
            ],
        )
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[turn],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=1,
            num_files_changed=1,
            patch=self._PATCH,
        )
        penalty = reward_model._compute_behaviour_penalties(trajectory)
        assert penalty == pytest.approx(-0.5)

    def test_length_penalty_under_threshold_is_zero(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify length penalty is zero when token count is under threshold."""
        reward_model = RewardModel(mock_telemetry)
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=20000,
            num_tool_calls=2,
            num_files_changed=1,
            patch=self._PATCH,
        )
        penalty = reward_model._compute_length_penalty(trajectory)
        assert penalty == 0.0

    def test_length_penalty_over_threshold_is_negative(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify quadratic length penalty is negative when exceeding threshold."""
        reward_model = RewardModel(mock_telemetry)
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._HIGH_TOKEN_COUNT,
            num_tool_calls=2,
            num_files_changed=1,
            patch=self._PATCH,
        )
        penalty = reward_model._compute_length_penalty(trajectory)
        assert penalty < 0.0

    def test_reward_is_clamped(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify total reward is clamped within [-1.5, 1.3]."""
        reward_model = RewardModel(mock_telemetry)
        test_result = MagicMock(all_pass=False, some_pass=False, passed=0, total=5)
        reward_model._apply_and_test = MagicMock(return_value=test_result)  # type: ignore[method-assign]
        bad_turn = Turn(
            role="model",
            tool_calls=[
                ToolCall(
                    tool_name="write_file",
                    args={"filepath": self._SCRATCH_WORKSPACE_PATH},
                ),
                ToolCall(
                    tool_name="edit_file",
                    args={"filepath": self._TEST_FILE_PATH},
                ),
            ],
        )
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[bad_turn],
            token_count=40000,
            num_tool_calls=20,
            num_files_changed=2,
            patch=self._PATCH,
            had_truncation=True,
        )
        reward = reward_model.compute_reward(trajectory, sample_task)
        assert reward >= -1.5

    def test_compute_batch_rewards(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify compute_batch_rewards computes reward list for batch."""
        reward_model = RewardModel(mock_telemetry)
        reward_model.compute_reward = MagicMock(return_value=0.75)  # type: ignore[method-assign]
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=2,
            num_files_changed=1,
            patch=self._PATCH,
        )
        rewards = reward_model.compute_batch_rewards([trajectory], [sample_task])
        assert rewards == [0.75]

    def test_apply_and_test_default_implementation(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify _apply_and_test returns unverified result by default."""
        reward_model = RewardModel(mock_telemetry)
        result = reward_model._apply_and_test(self._PATCH, sample_task)
        assert getattr(result, "resolved", False) is False

    def test_efficiency_bonus_linear_interpolation(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify linear interpolation bonus when tool calls are between threshold and median."""
        reward_model = RewardModel(mock_telemetry)
        # For SIMPLE, median = 6, threshold = 6 * 0.7 = 4.2. num_tool_calls = 5 is in [4.2, 6)
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=5,
            num_files_changed=1,
            patch=self._PATCH,
        )
        bonus = reward_model._compute_efficiency_bonus(trajectory, sample_task)
        assert 0.0 < bonus < 0.15

    def test_get_median_tool_calls_moderate_and_complex(
        self, mock_telemetry: MagicMock
    ) -> None:
        """Verify median tool call lookups for MODERATE and COMPLEX tiers."""
        reward_model = RewardModel(mock_telemetry)
        assert reward_model._get_median_tool_calls(ComplexityTier.MODERATE) == 12
        assert reward_model._get_median_tool_calls(ComplexityTier.COMPLEX) == 20

    def test_scratch_in_tmp_no_penalty(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify scratch files saved under /tmp/ incur no penalty."""
        reward_model = RewardModel(mock_telemetry)
        turn = Turn(
            role="model",
            tool_calls=[
                ToolCall(
                    tool_name="write_file",
                    args={"filepath": self._SCRATCH_TMP_PATH},
                )
            ],
        )
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[turn],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=1,
            num_files_changed=1,
            patch=self._PATCH,
        )
        penalty = reward_model._compute_behaviour_penalties(trajectory)
        assert penalty == 0.0

    def test_compute_outcome_reward_fail_returns_negative(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify _compute_outcome_reward returns _FAIL_REWARD when all tests fail."""
        reward_model = RewardModel(mock_telemetry)
        test_result = MagicMock(all_pass=False, some_pass=False, passed=0, total=5)
        reward_model._apply_and_test = MagicMock(return_value=test_result)  # type: ignore[method-assign]
        trajectory = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=self._NORMAL_TOKEN_COUNT,
            num_tool_calls=2,
            num_files_changed=1,
            patch=self._PATCH,
        )
        reward = reward_model._compute_outcome_reward(trajectory, sample_task)
        assert reward == -0.5
