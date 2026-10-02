"""Unit tests for PreferencePairGenerator constructing DPO preference pairs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.data.trajectory import Trajectory
from src.data.turn import Turn
from src.training.preference_pair_generator import PreferencePairGenerator
from src.training.reward_model import RewardModel
from src.utils.telemetry_logger import TelemetryLogger


class TestPreferencePairGenerator:
    """Test suite verifying preference pair creation and threshold filtering."""

    _INSTANCE_ID: str = "pair_task_001"
    _REPO: str = "google/gemma4"
    _COMMIT: str = "abc1234"
    _PROBLEM: str = "Handle NoneType error in dispatch"
    _HINTS: str = "Check argument validation"
    _PATCH: str = "diff --git a/dispatch.py b/dispatch.py\n+ if arg is None: return"
    _TEST_PATCH: str = "diff --git a/test_dispatch.py b/test_dispatch.py"
    _CREATED_AT: str = "2026-10-01T00:00:00Z"
    _SCORE_HIGH: float = 0.9
    _SCORE_LOW: float = 0.2
    _SCORE_SIMILAR: float = 0.8

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
    def mock_reward_model(self) -> MagicMock:
        """Create mock RewardModel fixture."""
        return MagicMock(spec=RewardModel)

    @pytest.fixture
    def mock_telemetry(self) -> MagicMock:
        """Create mock TelemetryLogger fixture."""
        return MagicMock(spec=TelemetryLogger)

    def test_generate_pairs_returns_list(
        self,
        sample_task: Task,
        mock_reward_model: MagicMock,
        mock_telemetry: MagicMock,
    ) -> None:
        """Verify generate_pairs produces list of preference dictionaries."""
        generator = PreferencePairGenerator(mock_reward_model, mock_telemetry)
        traj_good = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[Turn(role="model", thought="Good logic", text="Applied fix")],
            token_count=1000,
            num_tool_calls=1,
            num_files_changed=1,
            patch=self._PATCH,
        )
        traj_bad = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[Turn(role="model", text="Failed")],
            token_count=1000,
            num_tool_calls=1,
            num_files_changed=0,
            patch=None,
        )
        generator._generate_completions = MagicMock(return_value=[traj_good, traj_bad])  # type: ignore[method-assign]
        mock_reward_model.compute_reward.side_effect = [
            self._SCORE_HIGH,
            self._SCORE_LOW,
        ]
        pairs = generator.generate_pairs([sample_task], None, None, n_per_task=2)
        assert len(pairs) == 1
        assert pairs[0]["instance_id"] == self._INSTANCE_ID
        assert "prompt" in pairs[0]

    def test_generate_pairs_chosen_score_higher_than_rejected(
        self,
        sample_task: Task,
        mock_reward_model: MagicMock,
        mock_telemetry: MagicMock,
    ) -> None:
        """Verify chosen score strictly exceeds rejected score in each generated pair."""
        generator = PreferencePairGenerator(mock_reward_model, mock_telemetry)
        traj_good = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=1000,
            num_tool_calls=1,
            num_files_changed=1,
            patch=self._PATCH,
        )
        traj_bad = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=1000,
            num_tool_calls=1,
            num_files_changed=0,
            patch=None,
        )
        generator._generate_completions = MagicMock(return_value=[traj_good, traj_bad])  # type: ignore[method-assign]
        mock_reward_model.compute_reward.side_effect = [
            self._SCORE_HIGH,
            self._SCORE_LOW,
        ]
        pairs = generator.generate_pairs([sample_task], None, None, n_per_task=2)
        assert float(pairs[0]["chosen_score"]) > float(pairs[0]["rejected_score"])

    def test_generate_pairs_filters_small_differences(
        self,
        sample_task: Task,
        mock_reward_model: MagicMock,
        mock_telemetry: MagicMock,
    ) -> None:
        """Verify pairs with score gap below min threshold are discarded."""
        generator = PreferencePairGenerator(mock_reward_model, mock_telemetry)
        traj1 = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=1000,
            num_tool_calls=1,
            num_files_changed=1,
            patch=self._PATCH,
        )
        traj2 = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=1000,
            num_tool_calls=1,
            num_files_changed=1,
            patch=self._PATCH,
        )
        generator._generate_completions = MagicMock(return_value=[traj1, traj2])  # type: ignore[method-assign]
        mock_reward_model.compute_reward.side_effect = [
            self._SCORE_HIGH,
            self._SCORE_SIMILAR,
        ]
        pairs = generator.generate_pairs([sample_task], None, None, n_per_task=2)
        assert len(pairs) == 0

    def test_format_trajectory_with_formatted_text_attribute(
        self, mock_reward_model: MagicMock, mock_telemetry: MagicMock
    ) -> None:
        """Verify format_trajectory honors pre-existing formatted_text property."""
        generator = PreferencePairGenerator(mock_reward_model, mock_telemetry)
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=1000,
            num_tool_calls=0,
            num_files_changed=0,
        )
        traj.formatted_text = "Custom text representation"  # type: ignore[attr-defined]
        formatted = generator._format_trajectory(traj)
        assert formatted == "Custom text representation"

    def test_format_trajectory_with_patch_fallback(
        self, mock_reward_model: MagicMock, mock_telemetry: MagicMock
    ) -> None:
        """Verify format_trajectory falls back to patch when turns are empty."""
        generator = PreferencePairGenerator(mock_reward_model, mock_telemetry)
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=1000,
            num_tool_calls=0,
            num_files_changed=1,
            patch=self._PATCH,
        )
        formatted = generator._format_trajectory(traj)
        assert formatted == self._PATCH

    def test_format_trajectory_with_tool_result(
        self, mock_reward_model: MagicMock, mock_telemetry: MagicMock
    ) -> None:
        """Verify format_trajectory includes tool_result content from turn."""
        generator = PreferencePairGenerator(mock_reward_model, mock_telemetry)
        turn = Turn(role="user", tool_result="Test result output")
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[turn],
            token_count=1000,
            num_tool_calls=0,
            num_files_changed=0,
        )
        formatted = generator._format_trajectory(traj)
        assert "Test result output" in formatted

    def test_generate_completions_delegates_to_rollout_generator(
        self,
        sample_task: Task,
        mock_reward_model: MagicMock,
        mock_telemetry: MagicMock,
    ) -> None:
        """Verify _generate_completions delegates rollout creation to RolloutGenerator."""
        generator = PreferencePairGenerator(mock_reward_model, mock_telemetry)
        mock_model = MagicMock(return_value="output")
        mock_tokenizer = MagicMock()
        completions = generator._generate_completions(
            mock_model, mock_tokenizer, sample_task, n=1
        )
        assert len(completions) == 1
        assert completions[0].instance_id == self._INSTANCE_ID
