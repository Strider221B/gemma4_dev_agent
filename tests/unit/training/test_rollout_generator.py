"""Unit tests for RolloutGenerator simulating trajectory rollouts for RL training."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.task import Task
from src.data.tool_call import ToolCall
from src.training.rollout_generator import RolloutGenerator
from src.utils.telemetry_logger import TelemetryLogger


class TestRolloutGenerator:
    """Test suite verifying RolloutGenerator parsing, simulation, and rollout generation."""

    _INSTANCE_ID: str = "rollout_task_001"
    _REPO: str = "google/gemma4"
    _COMMIT: str = "commit_123"
    _PROBLEM: str = "Fix memory leak in cache manager"
    _HINTS: str = "Examine LRU cache eviction logic"
    _PATCH: str = "diff --git a/cache.py b/cache.py\n+ del self._cache[key]"
    _TEST_PATCH: str = "diff --git a/test_cache.py b/test_cache.py"
    _CREATED_AT: str = "2026-10-01T00:00:00Z"
    _MODEL_OUTPUT_DIFF: str = (
        "<|thought|>\nThinking about cache eviction.\n<|/thought|>\n"
        "diff --git a/cache.py b/cache.py\n+ del key"
    )
    _MODEL_OUTPUT_TAG: str = (
        "<|thought|>\nLet us patch.\n<|/thought|>\n"
        "<patch>\ndiff --git a/c.py b/c.py\n</patch>"
    )
    _MODEL_OUTPUT_TOOL_CALL: str = (
        '<|tool_call|>{"tool_name": "submit_patch", '
        '"args": {"patch": "diff --git a/c.py b/c.py"}}<|/tool_call|>'
    )
    _MALFORMED_TOOL_CALL: str = "<|tool_call|>{invalid json}<|/tool_call|>"

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

    def test_generate_rollouts_returns_trajectories(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify generate_rollouts produces requested count of rollouts."""
        generator = RolloutGenerator(mock_telemetry)
        mock_model = MagicMock(return_value=self._MODEL_OUTPUT_DIFF)
        mock_tokenizer = MagicMock()
        rollouts = generator.generate_rollouts(
            mock_model, mock_tokenizer, sample_task, num_generations=3
        )
        assert len(rollouts) == 3
        assert rollouts[0].instance_id == self._INSTANCE_ID
        mock_telemetry.log_info.assert_called_once()

    def test_execute_trajectory_with_tool_calls_and_patch(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify execute_trajectory parses submit_patch tool call and produces patch."""
        generator = RolloutGenerator(mock_telemetry)
        traj = generator._execute_trajectory(self._MODEL_OUTPUT_TOOL_CALL, sample_task)
        assert traj.patch is not None
        assert "diff --git" in traj.patch
        assert traj.num_tool_calls == 1
        assert len(traj.turns) == 3

    def test_execute_trajectory_with_thought_and_diff(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify thought tags and unified diff regex are parsed into Trajectory."""
        generator = RolloutGenerator(mock_telemetry)
        traj = generator._execute_trajectory(self._MODEL_OUTPUT_DIFF, sample_task)
        assert traj.patch is not None
        assert traj.turns[1].thought == "Thinking about cache eviction."

    def test_execute_trajectory_with_patch_tag(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify patch enclosed in <patch> tags is extracted correctly."""
        generator = RolloutGenerator(mock_telemetry)
        traj = generator._execute_trajectory(self._MODEL_OUTPUT_TAG, sample_task)
        assert traj.patch == "diff --git a/c.py b/c.py"

    def test_simulate_tool_execution_all_tools(
        self, mock_telemetry: MagicMock
    ) -> None:
        """Verify simulated results for all known and unknown tools."""
        generator = RolloutGenerator(mock_telemetry)
        tools = [
            ("run_command", "Command executed"),
            ("read_file", "File content loaded"),
            ("edit_file", "File edited successfully"),
            ("write_file", "File written successfully"),
            ("list_dir", "Directory entries listed"),
            ("submit_patch", "Patch submitted"),
            ("unknown_tool", "Tool executed: unknown_tool"),
        ]
        for tool_name, expected_substring in tools:
            res = generator._simulate_tool_execution(
                ToolCall(tool_name=tool_name, args={})
            )
            assert expected_substring in res

    def test_sample_model_output_branches(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify sample_model_output handles generate_text, callable, str, and object."""
        generator = RolloutGenerator(mock_telemetry)
        mock_with_generate = MagicMock(generate_text=lambda t: "from_generate")
        assert (
            generator._sample_model_output(mock_with_generate, None, sample_task)
            == "from_generate"
        )
        assert (
            generator._sample_model_output("raw_string", None, sample_task)
            == "raw_string"
        )

        class DummyModel:
            output: str = "from_output"

        assert (
            generator._sample_model_output(DummyModel(), None, sample_task)
            == "from_output"
        )
        assert generator._sample_model_output(object(), None, sample_task) == ""

    def test_malformed_tool_call_handled_gracefully(
        self, sample_task: Task, mock_telemetry: MagicMock
    ) -> None:
        """Verify malformed JSON inside tool_call tag does not raise error."""
        generator = RolloutGenerator(mock_telemetry)
        traj = generator._execute_trajectory(self._MALFORMED_TOOL_CALL, sample_task)
        assert traj.num_tool_calls == 0
