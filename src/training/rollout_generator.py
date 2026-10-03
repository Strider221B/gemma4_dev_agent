"""Rollout generator simulating agent execution loops for RL training."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from src.data.complexity_tier import ComplexityTier
from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.turn import Turn

if TYPE_CHECKING:
    from src.data.task import Task
    from src.utils.telemetry_logger import TelemetryLogger


class RolloutGenerator:
    """Generates trajectory rollouts from model generations with simulated tool execution."""

    _ROLE_USER: str = "user"
    _ROLE_MODEL: str = "model"
    _TOOL_RUN_COMMAND: str = "run_command"
    _TOOL_READ_FILE: str = "read_file"
    _TOOL_EDIT_FILE: str = "edit_file"
    _TOOL_WRITE_FILE: str = "write_file"
    _TOOL_LIST_DIR: str = "list_dir"
    _TOOL_SUBMIT_PATCH: str = "submit_patch"
    _ARG_PATCH: str = "patch"
    _ARG_FILEPATH: str = "filepath"
    _RESULT_CMD: str = "Command executed successfully with return code 0."
    _RESULT_READ: str = "File content loaded."
    _RESULT_EDIT: str = "File edited successfully."
    _RESULT_WRITE: str = "File written successfully."
    _RESULT_LIST: str = "Directory entries listed."
    _RESULT_SUBMIT: str = "Patch submitted."
    _RESULT_DEFAULT_PREFIX: str = "Tool executed: "
    _THOUGHT_PATTERN: str = (
        r"(?:<\|channel>thought|<\|thought\|>)(.*?)(?:<channel\|>|<\|/thought\|>)"
    )
    _TOOL_CALL_PATTERN: str = (
        r"(?:<\|tool_call>|<\|tool_call\|>)(.*?)(?:<tool_call\|>|<\|/tool_call\|>)"
    )
    _PATCH_TAG_PATTERN: str = r"<patch>(.*?)</patch>"
    _DIFF_PATTERN: str = r"(diff --git.*?)(?=\Z)"
    _CHARS_PER_TOKEN: int = 4
    _JOIN_DELIMITER: str = "\n"

    def __init__(self, telemetry: TelemetryLogger) -> None:
        """Initialize RolloutGenerator with injected telemetry logger."""
        self._telemetry: TelemetryLogger = telemetry

    def generate_rollouts(
        self, model: object, tokenizer: object, task: Task, num_generations: int
    ) -> list[Trajectory]:
        """Generate multiple trajectory rollouts for a specified task."""
        self._telemetry.log_info(
            f"Generating {num_generations} rollouts for task {task.instance_id}"
        )
        rollouts: list[Trajectory] = []
        for _ in range(num_generations):
            output_text = self._sample_model_output(model, tokenizer, task)
            trajectory = self._execute_trajectory(output_text, task)
            rollouts.append(trajectory)
        return rollouts

    def _execute_trajectory(self, model_output: str, task: Task) -> Trajectory:
        """Parse raw model output and execute tool steps into a Trajectory."""
        thought = self._extract_thought(model_output)
        tool_calls = self._extract_tool_calls(model_output)
        patch = self._extract_patch(model_output, tool_calls)
        turns: list[Turn] = [Turn(role=self._ROLE_USER, text=task.problem_statement)]
        turns.append(
            Turn(
                role=self._ROLE_MODEL,
                thought=thought,
                text=model_output,
                tool_calls=tool_calls,
            )
        )
        if tool_calls:
            results = [self._simulate_tool_execution(tc) for tc in tool_calls]
            turns.append(
                Turn(role=self._ROLE_USER, tool_result=self._JOIN_DELIMITER.join(results))
            )
        token_count = max(1, len(model_output) // self._CHARS_PER_TOKEN)
        complexity = getattr(task, "complexity", ComplexityTier.SIMPLE)
        return Trajectory(
            instance_id=task.instance_id,
            repo=task.repo,
            complexity=complexity,
            turns=turns,
            token_count=token_count,
            num_tool_calls=len(tool_calls),
            num_files_changed=1 if patch else 0,
            patch=patch,
        )

    def _extract_patch(
        self, text: str, tool_calls: list[ToolCall]
    ) -> str | None:
        """Extract unified diff patch from tool calls or formatted text tags."""
        for tc in tool_calls:
            if tc.tool_name == self._TOOL_SUBMIT_PATCH:
                arg_val = tc.args.get(self._ARG_PATCH)
                if arg_val:
                    return str(arg_val)
        tag_match = re.search(self._PATCH_TAG_PATTERN, text, re.DOTALL)
        if tag_match:
            return tag_match.group(1).strip()
        diff_match = re.search(self._DIFF_PATTERN, text, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        return None

    def _extract_thought(self, text: str) -> str | None:
        """Extract reasoning thought trace from Gemma formatted thought tags."""
        match = re.search(self._THOUGHT_PATTERN, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract and deserialize all tool call payloads from model text."""
        tool_calls: list[ToolCall] = []
        matches = re.findall(self._TOOL_CALL_PATTERN, text, re.DOTALL)
        for match_str in matches:
            try:
                data = json.loads(match_str.strip())
                if isinstance(data, dict) and "tool_name" in data:
                    args = data.get("args", {})
                    tool_calls.append(
                        ToolCall(
                            tool_name=str(data["tool_name"]),
                            args=args if isinstance(args, dict) else {},
                        )
                    )
            except Exception:
                continue
        return tool_calls

    def _sample_model_output(
        self, model: object, tokenizer: object, task: Task
    ) -> str:
        """Query model or invocation callable to produce text output."""
        if hasattr(model, "generate_text"):
            return str(getattr(model, "generate_text")(task))
        if callable(model):
            return str(model(task))
        if isinstance(model, str):
            return model
        return str(getattr(model, "output", ""))

    def _simulate_tool_execution(self, tool_call: ToolCall) -> str:
        """Simulate execution output for a specified tool call."""
        name = tool_call.tool_name
        if name == self._TOOL_RUN_COMMAND:
            return self._RESULT_CMD
        if name == self._TOOL_READ_FILE:
            return self._RESULT_READ
        if name == self._TOOL_EDIT_FILE:
            return self._RESULT_EDIT
        if name == self._TOOL_WRITE_FILE:
            return self._RESULT_WRITE
        if name == self._TOOL_LIST_DIR:
            return self._RESULT_LIST
        if name == self._TOOL_SUBMIT_PATCH:
            return self._RESULT_SUBMIT
        return f"{self._RESULT_DEFAULT_PREFIX}{name}"
