"""Trajectory augmentation module for diversifying agent trajectories."""

from __future__ import annotations

import copy
import random
from typing import Callable

from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.turn import Turn


class TrajectoryAugmentor:
    """Diversifies agent trajectories via various training augmentation strategies."""

    _MAX_AUGMENTATIONS_PER_TRAJECTORY: int = 3
    _WINDOW_DELTA: int = 10
    _ROLE_USER: str = "user"
    _ROLE_MODEL: str = "model"
    _TOOL_SEARCH: str = "search_similar_code"
    _TOOL_NEIGHBORS: str = "get_code_neighbors"
    _TOOL_READ_FILE: str = "read_file"
    _TOOL_EDIT_FILE: str = "edit_file"
    _TOOL_RUN_CMD: str = "run_command"
    _TOOL_WRITE_FILE: str = "write_file"
    _TOOL_GET_STATUS: str = "get_status"
    _ARG_FILEPATH: str = "filepath"
    _ARG_START_LINE: str = "start_line"
    _ARG_END_LINE: str = "end_line"
    _ARG_OLD_STRING: str = "old_string"
    _ARG_NEW_STRING: str = "new_string"
    _ARG_COMMAND: str = "command"
    _ARG_CONTENT: str = "content"
    _ARG_QUERY: str = "query"
    _DEFAULT_QUERY: str = "symbol"
    _BAD_OLD_STRING: str = "INVALID_OLD_STRING_FOR_ERROR_RECOVERY\n"
    _REPRO_PATH: str = "/tmp/repro.py"
    _REPRO_CONTENT: str = "# Reproduction script\nassert True\n"
    _REPRO_RUN_CMD: str = "python /tmp/repro.py"
    _CLEANUP_CMD: str = "rm -f /tmp/repro.py"
    _GREP_CMD: str = "grep -rn '{query}' ."
    _FAILED_EDIT_RESULT: str = '{"status": "error", "message": "old_string not found"}'
    _OK_STATUS_RESULT: str = (
        '{"status": "ok", "remaining_time_min": 45.0, "tool_calls_remaining": 85}'
    )
    _WRITE_OK_RESULT: str = '{"status": "ok", "bytes_written": 32}'
    _REPRO_RUN_RESULT: str = '{"status": "ok", "returncode": 0, "output": ""}'
    _CLEANUP_RESULT: str = '{"status": "ok", "returncode": 0, "output": ""}'
    _GREP_RESULT: str = (
        '{"status": "ok", "returncode": 0, "output": "src/utils.py:10: def foo():"}'
    )
    _ERR_RECOVERY_THOUGHT: str = "Attempting to apply patch."
    _ERR_CORRECTION_THOUGHT: str = (
        "The edit failed. Re-reading context and correcting edit pattern."
    )
    _ERR_CORRECTION_TEXT: str = "Correcting the patch."
    _SCRATCH_THOUGHT: str = "Writing reproduction script to verify the issue."
    _REPRO_EVAL_THOUGHT: str = "Executing reproduction script."
    _CLEANUP_THOUGHT: str = "Cleaning up temporary reproduction files."
    _STATUS_THOUGHT: str = "Checking current execution budget."
    _EMPTY_STR: str = ""

    def __init__(self, seed: int = 42) -> None:
        """Initialize TrajectoryAugmentor with random seed."""
        self._seed: int = seed
        self._rng: random.Random = random.Random(seed)

    def augment(self, trajectory: Trajectory) -> list[Trajectory]:
        """Produce augmented variants of a single trajectory."""
        strategies: list[Callable[[Trajectory], Trajectory]] = [
            self._permute_tool_order,
            self._inject_error_recovery,
            self._vary_read_windows,
            self._toggle_graph_vs_grep,
            self._add_scratchpad_discipline,
            self._add_budget_checks,
        ]
        augmented: list[Trajectory] = []
        selected = self._rng.sample(
            strategies,
            k=min(len(strategies), self._MAX_AUGMENTATIONS_PER_TRAJECTORY),
        )
        for strat in selected:
            augmented.append(strat(trajectory))
        return augmented

    def _permute_tool_order(self, trajectory: Trajectory) -> Trajectory:
        """Swap order of navigation tool calls."""
        clone = self._clone_trajectory(trajectory)
        nav_indices = self._find_navigation_turn_indices(clone.turns)
        if len(nav_indices) >= 2:
            idx1, idx2 = nav_indices[0], nav_indices[1]
            pair1 = clone.turns[idx1 : idx1 + 2]
            pair2 = clone.turns[idx2 : idx2 + 2]
            clone.turns[idx1 : idx1 + 2] = pair2
            clone.turns[idx2 : idx2 + 2] = pair1
        return clone

    def _inject_error_recovery(self, trajectory: Trajectory) -> Trajectory:
        """Insert a failed edit_file call followed by correction turns."""
        clone = self._clone_trajectory(trajectory)
        for idx, turn in enumerate(clone.turns):
            for tc in turn.tool_calls:
                if tc.tool_name == self._TOOL_EDIT_FILE:
                    clone.turns[idx:idx] = self._build_recovery_turns(tc)
                    clone.num_tool_calls = sum(
                        len(t.tool_calls) for t in clone.turns
                    )
                    return clone
        return clone

    def _vary_read_windows(self, trajectory: Trajectory) -> Trajectory:
        """Change start_line and end_line in read_file calls."""
        clone = self._clone_trajectory(trajectory)
        for turn in clone.turns:
            for idx, tc in enumerate(turn.tool_calls):
                if tc.tool_name == self._TOOL_READ_FILE:
                    turn.tool_calls[idx] = self._expand_read_tool_call(tc)
        return clone

    def _toggle_graph_vs_grep(self, trajectory: Trajectory) -> Trajectory:
        """Replace search_similar_code with run_command grep or vice versa."""
        clone = self._clone_trajectory(trajectory)
        for idx, turn in enumerate(clone.turns):
            for c_idx, tc in enumerate(turn.tool_calls):
                if tc.tool_name == self._TOOL_SEARCH:
                    self._apply_grep_replacement(clone, idx, c_idx, tc)
                    return clone
        return clone

    def _add_scratchpad_discipline(self, trajectory: Trajectory) -> Trajectory:
        """Insert a write_file to /tmp/repro.py, run_command, and cleanup."""
        clone = self._clone_trajectory(trajectory)
        insert_idx = min(2, len(clone.turns))
        clone.turns[insert_idx:insert_idx] = self._build_scratchpad_turns()
        clone.num_tool_calls = sum(len(t.tool_calls) for t in clone.turns)
        return clone

    def _add_budget_checks(self, trajectory: Trajectory) -> Trajectory:
        """Insert get_status() calls at 1/3 and 2/3 points of trajectory."""
        clone = self._clone_trajectory(trajectory)
        if len(clone.turns) < 3:
            return clone
        idx1 = len(clone.turns) // 3
        idx2 = (2 * len(clone.turns)) // 3
        clone.turns[idx2:idx2] = self._build_status_turns()
        clone.turns[idx1:idx1] = self._build_status_turns()
        clone.num_tool_calls = sum(len(t.tool_calls) for t in clone.turns)
        return clone

    def _clone_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Deep copy helper preserving trajectory attributes."""
        clone = copy.deepcopy(trajectory)
        clone.num_tool_calls = sum(len(t.tool_calls) for t in clone.turns)
        return clone

    def _find_navigation_turn_indices(self, turns: list[Turn]) -> list[int]:
        """Find starting turn indices of navigation tool calls."""
        nav_tools = (
            self._TOOL_SEARCH,
            self._TOOL_NEIGHBORS,
            self._TOOL_READ_FILE,
        )
        indices: list[int] = []
        for i, turn in enumerate(turns[:-1]):
            if turn.role == self._ROLE_MODEL and any(
                tc.tool_name in nav_tools for tc in turn.tool_calls
            ):
                indices.append(i)
        return indices

    def _build_recovery_turns(self, tc: ToolCall) -> list[Turn]:
        """Create failed edit turn, failure result, and correction thought."""
        path = str(tc.args.get(self._ARG_FILEPATH, self._EMPTY_STR))
        failed_args: dict[str, object] = {
            self._ARG_FILEPATH: path,
            self._ARG_OLD_STRING: self._BAD_OLD_STRING,
            self._ARG_NEW_STRING: self._EMPTY_STR,
        }
        failed_turn = Turn(
            role=self._ROLE_MODEL,
            thought=self._ERR_RECOVERY_THOUGHT,
            tool_calls=[ToolCall(self._TOOL_EDIT_FILE, failed_args)],
        )
        err_user_turn = Turn(
            role=self._ROLE_USER,
            tool_result=self._FAILED_EDIT_RESULT,
        )
        thought_turn = Turn(
            role=self._ROLE_MODEL,
            thought=self._ERR_CORRECTION_THOUGHT,
            text=self._ERR_CORRECTION_TEXT,
        )
        return [failed_turn, err_user_turn, thought_turn]

    def _expand_read_tool_call(self, tc: ToolCall) -> ToolCall:
        """Expand read_file line window bounds."""
        start = max(
            1,
            int(str(tc.args.get(self._ARG_START_LINE, 1))) - self._WINDOW_DELTA,
        )
        end = int(str(tc.args.get(self._ARG_END_LINE, 50))) + self._WINDOW_DELTA
        new_args = {
            **tc.args,
            self._ARG_START_LINE: start,
            self._ARG_END_LINE: end,
        }
        return ToolCall(tool_name=tc.tool_name, args=new_args)

    def _apply_grep_replacement(
        self, clone: Trajectory, idx: int, c_idx: int, tc: ToolCall
    ) -> None:
        """Replace search_similar_code with run_command grep invocation."""
        query = str(tc.args.get(self._ARG_QUERY, self._DEFAULT_QUERY))
        cmd = self._GREP_CMD.format(query=query)
        clone.turns[idx].tool_calls[c_idx] = ToolCall(
            self._TOOL_RUN_CMD, {self._ARG_COMMAND: cmd}
        )
        if idx + 1 < len(clone.turns) and clone.turns[idx + 1].role == self._ROLE_USER:
            clone.turns[idx + 1] = Turn(
                role=self._ROLE_USER,
                tool_result=self._GREP_RESULT,
            )

    def _build_scratchpad_turns(self) -> list[Turn]:
        """Build sequence of turns creating, testing, and cleaning repro file."""
        write_call = ToolCall(
            self._TOOL_WRITE_FILE,
            {
                self._ARG_FILEPATH: self._REPRO_PATH,
                self._ARG_CONTENT: self._REPRO_CONTENT,
            },
        )
        t1 = Turn(
            role=self._ROLE_MODEL,
            thought=self._SCRATCH_THOUGHT,
            tool_calls=[write_call],
        )
        t2 = Turn(role=self._ROLE_USER, tool_result=self._WRITE_OK_RESULT)
        run_call = ToolCall(
            self._TOOL_RUN_CMD, {self._ARG_COMMAND: self._REPRO_RUN_CMD}
        )
        t3 = Turn(
            role=self._ROLE_MODEL,
            thought=self._REPRO_EVAL_THOUGHT,
            tool_calls=[run_call],
        )
        t4 = Turn(role=self._ROLE_USER, tool_result=self._REPRO_RUN_RESULT)
        clean_call = ToolCall(
            self._TOOL_RUN_CMD, {self._ARG_COMMAND: self._CLEANUP_CMD}
        )
        t5 = Turn(
            role=self._ROLE_MODEL,
            thought=self._CLEANUP_THOUGHT,
            tool_calls=[clean_call],
        )
        t6 = Turn(role=self._ROLE_USER, tool_result=self._CLEANUP_RESULT)
        return [t1, t2, t3, t4, t5, t6]

    def _build_status_turns(self) -> list[Turn]:
        """Build status inquiry turn and harness response turn."""
        call = ToolCall(self._TOOL_GET_STATUS, {})
        t1 = Turn(
            role=self._ROLE_MODEL,
            thought=self._STATUS_THOUGHT,
            tool_calls=[call],
        )
        t2 = Turn(role=self._ROLE_USER, tool_result=self._OK_STATUS_RESULT)
        return [t1, t2]
