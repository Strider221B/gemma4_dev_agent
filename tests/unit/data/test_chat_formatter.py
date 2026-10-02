"""Unit tests for ChatFormatter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.config.data_paths_config import DataPathsConfig
from src.data.chat_formatter import ChatFormatter
from src.data.complexity_tier import ComplexityTier
from src.data.ingestor import DataIngestor
from src.data.mock_data_factory import MockDataFactory
from src.data.patch_parser import PatchParser
from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.trajectory_synthesiser import TrajectorySynthesiser
from src.data.turn import Turn
from src.utils.token_counter import TokenCounter


class TestChatFormatter:
    """Test suite for ChatFormatter class."""

    _MOCK_TASK_ID: str = "mock_task_001"
    _MOCK_REPO: str = "test/repo"
    _ROLE_USER: str = "user"
    _ROLE_MODEL: str = "model"
    _SAMPLE_PROMPT: str = "Please fix the issue in app.py"
    _SAMPLE_THOUGHT: str = "Analyzing the codebase structure."
    _SAMPLE_TEXT: str = "Searching for relevant symbols."
    _SAMPLE_TOOL_NAME: str = "search_similar_code"
    _SAMPLE_QUERY_KEY: str = "query"
    _SAMPLE_QUERY_VAL: str = "solve_bug"
    _SAMPLE_RESULT: str = '{"status": "ok", "matches": [1, 2]}'
    _SAMPLE_TOKEN_COUNT: int = 42
    _CUSTOM_MAX_TOKENS: int = 16384
    _BOS: str = "<bos>"
    _START_TURN: str = "<start_of_turn>"
    _END_TURN: str = "<end_of_turn>"
    _TOOL_OPEN: str = "<|tool_call|>"
    _TOOL_CLOSE: str = "<|/tool_call|>"
    _THOUGHT_OPEN: str = "<|thought|>"
    _THOUGHT_CLOSE: str = "<|/thought|>"
    _KEY_ROLE: str = "role"
    _KEY_CONTENT: str = "content"

    def test_format_trajectory_contains_bos_token(self) -> None:
        """Verify formatted trajectory string begins with <bos> token."""
        formatter = self._create_formatter()
        trajectory = self._create_sample_trajectory()
        formatted = formatter.format_trajectory(trajectory)
        assert formatted.startswith(self._BOS)

    def test_format_trajectory_contains_start_end_turn_tokens(self) -> None:
        """Verify formatted trajectory contains turn delimiters with roles."""
        formatter = self._create_formatter()
        trajectory = self._create_sample_trajectory()
        formatted = formatter.format_trajectory(trajectory)
        assert f"{self._START_TURN}{self._ROLE_USER}" in formatted
        assert f"{self._START_TURN}{self._ROLE_MODEL}" in formatted
        assert self._END_TURN in formatted

    def test_format_trajectory_wraps_thought_in_tags(self) -> None:
        """Verify model thought is wrapped in Gemma thought delimiters."""
        formatter = self._create_formatter()
        trajectory = self._create_sample_trajectory()
        formatted = formatter.format_trajectory(trajectory)
        assert f"{self._THOUGHT_OPEN}\n{self._SAMPLE_THOUGHT}\n{self._THOUGHT_CLOSE}" in formatted

    def test_format_trajectory_wraps_tool_call_in_tags(self) -> None:
        """Verify tool calls are wrapped in Gemma tool_call delimiters."""
        formatter = self._create_formatter()
        trajectory = self._create_sample_trajectory()
        formatted = formatter.format_trajectory(trajectory)
        assert self._TOOL_OPEN in formatted
        assert self._TOOL_CLOSE in formatted
        assert self._SAMPLE_TOOL_NAME in formatted

    def test_format_model_turn_with_thought_and_tool_call(self) -> None:
        """Verify model turn containing thought, text, and tool call formats properly."""
        formatter = self._create_formatter()
        tool_call = ToolCall(
            tool_name=self._SAMPLE_TOOL_NAME,
            args={self._SAMPLE_QUERY_KEY: self._SAMPLE_QUERY_VAL},
        )
        turn = Turn(
            role=self._ROLE_MODEL,
            thought=self._SAMPLE_THOUGHT,
            text=self._SAMPLE_TEXT,
            tool_calls=[tool_call],
        )
        formatted = formatter._format_model_turn(turn)
        assert self._THOUGHT_OPEN in formatted
        assert self._SAMPLE_THOUGHT in formatted
        assert self._SAMPLE_TEXT in formatted
        assert self._TOOL_OPEN in formatted
        assert self._SAMPLE_TOOL_NAME in formatted

    def test_format_model_turn_without_thought(self) -> None:
        """Verify model turn with no thought block omits thought tags."""
        formatter = self._create_formatter()
        turn = Turn(role=self._ROLE_MODEL, text=self._SAMPLE_TEXT)
        formatted = formatter._format_model_turn(turn)
        assert self._THOUGHT_OPEN not in formatted
        assert self._SAMPLE_TEXT in formatted

    def test_format_user_turn_with_tool_result(self) -> None:
        """Verify user turn holding tool execution output formats as result content."""
        formatter = self._create_formatter()
        turn = Turn(role=self._ROLE_USER, tool_result=self._SAMPLE_RESULT)
        formatted = formatter._format_user_turn(turn)
        assert formatted == self._SAMPLE_RESULT

    def test_format_user_turn_with_text(self) -> None:
        """Verify user turn containing text prompt formats as expected."""
        formatter = self._create_formatter()
        turn = Turn(role=self._ROLE_USER, text=self._SAMPLE_PROMPT)
        formatted = formatter._format_user_turn(turn)
        assert formatted == self._SAMPLE_PROMPT

    def test_format_tool_call_produces_valid_json(self) -> None:
        """Verify tool call block produces parseable JSON matching the payload."""
        formatter = self._create_formatter()
        tool_call = ToolCall(
            tool_name=self._SAMPLE_TOOL_NAME,
            args={self._SAMPLE_QUERY_KEY: self._SAMPLE_QUERY_VAL},
        )
        formatted = formatter._format_tool_call(tool_call)
        assert formatted.startswith(f"{self._TOOL_OPEN}\n")
        assert formatted.endswith(f"\n{self._TOOL_CLOSE}")
        inner_json = formatted.replace(self._TOOL_OPEN, "").replace(self._TOOL_CLOSE, "").strip()
        parsed = json.loads(inner_json)
        assert parsed["tool_name"] == self._SAMPLE_TOOL_NAME
        assert parsed["args"][self._SAMPLE_QUERY_KEY] == self._SAMPLE_QUERY_VAL

    def test_format_messages_returns_list_of_dicts(self) -> None:
        """Verify format_messages outputs list of dicts with role and content keys."""
        formatter = self._create_formatter()
        trajectory = self._create_sample_trajectory()
        messages = formatter.format_messages(trajectory.turns)
        assert isinstance(messages, list)
        assert len(messages) == len(trajectory.turns)
        for msg in messages:
            assert self._KEY_ROLE in msg
            assert self._KEY_CONTENT in msg

    def test_format_messages_alternates_roles(self) -> None:
        """Verify format_messages preserves expected alternating roles."""
        formatter = self._create_formatter()
        trajectory = self._create_sample_trajectory()
        messages = formatter.format_messages(trajectory.turns)
        assert messages[0][self._KEY_ROLE] == self._ROLE_USER
        assert messages[1][self._KEY_ROLE] == self._ROLE_MODEL
        assert messages[2][self._KEY_ROLE] == self._ROLE_USER

    def test_count_tokens_positive(self) -> None:
        """Verify count_tokens returns positive count for non-empty text."""
        formatter = self._create_formatter()
        count = formatter.count_tokens(self._SAMPLE_PROMPT)
        assert count > 0

    def test_count_tokens_delegates_to_token_counter(self) -> None:
        """Verify count_tokens delegates directly to the injected TokenCounter."""
        mock_counter = MagicMock(spec=TokenCounter)
        mock_counter.count_tokens.return_value = self._SAMPLE_TOKEN_COUNT
        formatter = ChatFormatter(token_counter=mock_counter)
        result = formatter.count_tokens(self._SAMPLE_PROMPT)
        assert result == self._SAMPLE_TOKEN_COUNT
        mock_counter.count_tokens.assert_called_once_with(self._SAMPLE_PROMPT)

    def test_format_trajectory_empty_turns(self) -> None:
        """Verify trajectory with zero turns returns only the BOS token."""
        formatter = self._create_formatter()
        trajectory = Trajectory(
            instance_id=self._MOCK_TASK_ID,
            repo=self._MOCK_REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[],
            token_count=0,
            num_tool_calls=0,
            num_files_changed=0,
        )
        assert formatter.format_trajectory(trajectory) == self._BOS

    def test_format_user_turn_with_text_and_tool_result(self) -> None:
        """Verify user turn containing both text and tool_result formats both."""
        formatter = self._create_formatter()
        turn = Turn(
            role=self._ROLE_USER,
            text=self._SAMPLE_PROMPT,
            tool_result=self._SAMPLE_RESULT,
        )
        formatted = formatter._format_user_turn(turn)
        assert self._SAMPLE_PROMPT in formatted
        assert self._SAMPLE_RESULT in formatted

    def test_format_user_turn_empty(self) -> None:
        """Verify user turn without text or tool_result returns empty string."""
        formatter = self._create_formatter()
        turn = Turn(role=self._ROLE_USER)
        assert formatter._format_user_turn(turn) == ""

    def test_format_model_turn_multiple_tool_calls(self) -> None:
        """Verify model turn with multiple tool calls contains all call blocks."""
        formatter = self._create_formatter()
        tc1 = ToolCall(tool_name="tool_one", args={"k": 1})
        tc2 = ToolCall(tool_name="tool_two", args={"k": 2})
        turn = Turn(role=self._ROLE_MODEL, tool_calls=[tc1, tc2])
        formatted = formatter._format_model_turn(turn)
        assert "tool_one" in formatted
        assert "tool_two" in formatted
        assert formatted.count(self._TOOL_OPEN) == 2

    def test_max_tokens_property(self) -> None:
        """Verify max_tokens property returns configured value."""
        counter = TokenCounter()
        formatter = ChatFormatter(token_counter=counter, max_tokens=self._CUSTOM_MAX_TOKENS)
        assert formatter.max_tokens == self._CUSTOM_MAX_TOKENS

    def test_mock_data_factory_pipeline_formatting(self) -> None:
        """Verify end-to-end integration: synthesise mock task and format trajectory."""
        task = MockDataFactory().create_mock_task()
        counter = TokenCounter()
        paths = DataPathsConfig(tasks_path="", graphs_dir="", embeddings_dir="")
        synthesiser = TrajectorySynthesiser(
            ingestor=DataIngestor(paths),
            patch_parser=PatchParser(),
            token_counter=counter,
            mock_mode=True,
        )
        trajectory = synthesiser.synthesise(task)
        formatter = ChatFormatter(token_counter=counter)
        formatted = formatter.format_trajectory(trajectory)
        messages = formatter.format_messages(trajectory.turns)

        assert formatted.startswith(self._BOS)
        assert f"{self._START_TURN}{self._ROLE_USER}" in formatted
        assert f"{self._START_TURN}{self._ROLE_MODEL}" in formatted
        assert self._TOOL_OPEN in formatted
        assert "submit_patch" in formatted
        assert len(messages) > 0
        assert messages[0][self._KEY_ROLE] == self._ROLE_USER

    def _create_formatter(self) -> ChatFormatter:
        """Instantiate ChatFormatter with a real TokenCounter."""
        return ChatFormatter(token_counter=TokenCounter())

    def _create_sample_trajectory(self) -> Trajectory:
        """Build a deterministic 3-turn sample trajectory for formatting tests."""
        turn1 = Turn(role=self._ROLE_USER, text=self._SAMPLE_PROMPT)
        turn2 = Turn(
            role=self._ROLE_MODEL,
            thought=self._SAMPLE_THOUGHT,
            text=self._SAMPLE_TEXT,
            tool_calls=[
                ToolCall(
                    tool_name=self._SAMPLE_TOOL_NAME,
                    args={self._SAMPLE_QUERY_KEY: self._SAMPLE_QUERY_VAL},
                )
            ],
        )
        turn3 = Turn(role=self._ROLE_USER, tool_result=self._SAMPLE_RESULT)
        return Trajectory(
            instance_id=self._MOCK_TASK_ID,
            repo=self._MOCK_REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[turn1, turn2, turn3],
            token_count=100,
            num_tool_calls=1,
            num_files_changed=1,
        )
