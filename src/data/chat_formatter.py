"""Chat template formatting for Gemma 4 agent trajectories."""

from __future__ import annotations

import json

from src.data.constants import MAX_TRAJECTORY_TOKENS
from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.turn import Turn
from src.utils.token_counter import TokenCounter


class ChatFormatter:
    """Formats multi-turn trajectories into Gemma 4 chat template strings."""

    _BOS_TOKEN: str = "<bos>"
    _START_OF_TURN: str = "<|turn>"
    _END_OF_TURN: str = "<turn|>"
    _TOOL_CALL_OPEN: str = "<|tool_call>"
    _TOOL_CALL_CLOSE: str = "<tool_call|>"
    _THOUGHT_OPEN: str = "<|channel>thought"
    _THOUGHT_CLOSE: str = "<channel|>"
    _USER_ROLE: str = "user"
    _MODEL_ROLE: str = "model"
    _KEY_ROLE: str = "role"
    _KEY_CONTENT: str = "content"
    _KEY_TOOL_NAME: str = "tool_name"
    _KEY_ARGS: str = "args"
    _NEWLINE: str = "\n"

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = MAX_TRAJECTORY_TOKENS,
    ) -> None:
        """Initialize ChatFormatter with token counter and max token limit."""
        self._token_counter: TokenCounter = token_counter
        self._max_tokens: int = max_tokens

    @property
    def max_tokens(self) -> int:
        """Return the configured maximum token budget."""
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        """Delegate token counting to the token counter."""
        return self._token_counter.count_tokens(text)

    def format_messages(self, turns: list[Turn]) -> list[dict[str, str]]:
        """Convert turns into a list of HuggingFace conversation role-content dicts."""
        messages: list[dict[str, str]] = []
        for turn in turns:
            content = (
                self._format_user_turn(turn)
                if turn.role == self._USER_ROLE
                else self._format_model_turn(turn)
            )
            messages.append({self._KEY_ROLE: turn.role, self._KEY_CONTENT: content})
        return messages

    def format_trajectory(self, trajectory: Trajectory) -> str:
        """Convert a complete trajectory into a Gemma 4 formatted chat string."""
        return self._apply_chat_template(trajectory.turns)

    def _apply_chat_template(self, turns: list[Turn]) -> str:
        """Assemble all turns with BOS token and start/end turn delimiters."""
        formatted_turns: list[str] = []
        for turn in turns:
            content = (
                self._format_user_turn(turn)
                if turn.role == self._USER_ROLE
                else self._format_model_turn(turn)
            )
            turn_block = (
                f"{self._START_OF_TURN}{turn.role}\n{content}\n{self._END_OF_TURN}"
                if content
                else f"{self._START_OF_TURN}{turn.role}\n{self._END_OF_TURN}"
            )
            formatted_turns.append(turn_block)

        if not formatted_turns:
            return self._BOS_TOKEN

        return f"{self._BOS_TOKEN}{self._NEWLINE.join(formatted_turns)}\n"

    def _format_model_turn(self, turn: Turn) -> str:
        """Format model turn with thought blocks, response text, and tool calls."""
        parts: list[str] = []
        if turn.thought:
            parts.append(self._format_thought_block(turn.thought))
        if turn.text:
            parts.append(turn.text.strip())
        for tool_call in turn.tool_calls:
            parts.append(self._format_tool_call(tool_call))
        return self._NEWLINE.join(parts)

    def _format_thought_block(self, thought: str) -> str:
        """Wrap reasoning text within Gemma thought tags."""
        return f"{self._THOUGHT_OPEN}\n{thought.strip()}\n{self._THOUGHT_CLOSE}"

    def _format_tool_call(self, tool_call: ToolCall) -> str:
        """Format a tool invocation into a Gemma tool_call JSON block."""
        payload = {
            self._KEY_TOOL_NAME: tool_call.tool_name,
            self._KEY_ARGS: tool_call.args,
        }
        json_str = json.dumps(payload)
        return f"{self._TOOL_CALL_OPEN}\n{json_str}\n{self._TOOL_CALL_CLOSE}"

    def _format_tool_result(self, result: str) -> str:
        """Format tool result string as user turn content."""
        return result.strip()

    def _format_user_turn(self, turn: Turn) -> str:
        """Format user turn content from text or tool result."""
        parts: list[str] = []
        if turn.text:
            parts.append(turn.text.strip())
        if turn.tool_result is not None:
            parts.append(self._format_tool_result(turn.tool_result))
        return self._NEWLINE.join(parts)
