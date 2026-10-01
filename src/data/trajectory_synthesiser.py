"""Trajectory synthesiser converting problem and patch into agent trajectories."""

from __future__ import annotations

import json
import re

from src.data.complexity_tier import ComplexityTier
from src.data.file_change import FileChange
from src.data.hunk import Hunk
from src.data.ingestor import DataIngestor
from src.data.patch_parser import PatchParser
from src.data.task import Task
from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.turn import Turn
from src.utils.token_counter import TokenCounter


class TrajectorySynthesiser:
    """Synthesises multi-turn agent interaction trajectories from benchmark tasks."""

    _MAX_TRAJECTORY_TOKENS: int = 28672
    _READ_CONTEXT_LINES: int = 30
    _MAX_OLD_STRING_LINES: int = 15
    _DEFAULT_QUERY_K: int = 5
    _DEFAULT_SCORE: float = 0.95
    _ROLE_USER: str = "user"
    _ROLE_MODEL: str = "model"
    _TOOL_SEARCH: str = "search_similar_code"
    _TOOL_NEIGHBORS: str = "get_code_neighbors"
    _TOOL_READ_FILE: str = "read_file"
    _TOOL_EDIT_FILE: str = "edit_file"
    _TOOL_RUN_COMMAND: str = "run_command"
    _TOOL_SUBMIT_PATCH: str = "submit_patch"
    _KEY_STATUS: str = "status"
    _KEY_OK: str = "ok"
    _KEY_QUERY: str = "query"
    _KEY_RESULTS: str = "results"
    _KEY_NODE_NAME: str = "node_name"
    _KEY_FILEPATH: str = "filepath"
    _KEY_SCORE: str = "score"
    _KEY_COUNT: str = "count"
    _KEY_NODE: str = "node"
    _KEY_NEIGHBORS: str = "neighbors"
    _KEY_SOURCE: str = "source"
    _KEY_TARGET: str = "target"
    _KEY_TYPE: str = "type"
    _KEY_START_LINE: str = "start_line"
    _KEY_END_LINE: str = "end_line"
    _KEY_CONTENT: str = "content"
    _KEY_OCCURRENCES: str = "occurrences"
    _KEY_STRATEGY: str = "strategy"
    _KEY_RETURNCODE: str = "returncode"
    _KEY_OUTPUT: str = "output"
    _KEY_OLD_STRING: str = "old_string"
    _KEY_NEW_STRING: str = "new_string"
    _KEY_COMMAND: str = "command"
    _KEY_MESSAGE: str = "message"
    _DEFAULT_SYMBOL: str = "add_numbers"
    _DEFAULT_FILEPATH: str = "utils.py"
    _DEFAULT_KEYWORD: str = "issue"
    _PYTEST_CMD: str = "python -m pytest -q"
    _PYTEST_OUTPUT: str = "1 passed in 0.05s"
    _PATCH_SUBMITTED_MSG: str = "Patch submitted successfully."
    _MSG_SEARCHING_CODE: str = (
        "Let me search for the relevant code in the repository."
    )
    _MSG_EXAMINING_GRAPH: str = (
        "Examining the call graph for this function."
    )
    _ROOT_CAUSE_THOUGHT: str = (
        "Root cause identified. Preparing patch modifications."
    )
    _ROOT_CAUSE_TEXT: str = "Root cause identified. Proceeding to fix."
    _STOPWORDS: tuple[str, ...] = (
        "the", "and", "that", "this", "with", "from", "for", "have", "were",
    )

    def __init__(
        self,
        ingestor: DataIngestor,
        patch_parser: PatchParser,
        token_counter: TokenCounter,
        mock_mode: bool = False,
    ) -> None:
        """Initialize TrajectorySynthesiser with required components."""
        self._ingestor: DataIngestor = ingestor
        self._patch_parser: PatchParser = patch_parser
        self._token_counter: TokenCounter = token_counter
        self._mock_mode: bool = mock_mode

    def synthesise(self, task: Task) -> Trajectory:
        """Convert a single task into a multi-turn gold trajectory."""
        file_changes = self._patch_parser.parse(task.patch)
        graph = (
            None
            if self._mock_mode
            else self._ingestor.load_graph(task.instance_id)
        )
        symbols = self._extract_symbols(file_changes, graph)
        turns: list[Turn] = [self._build_initial_prompt(task)]
        turns.extend(self._synthesise_navigation(task, graph, symbols))
        turns.extend(self._synthesise_diagnosis(task, file_changes))
        turns.extend(self._synthesise_patches(task, file_changes))
        turns.extend(self._synthesise_verification(task))
        turns.extend(self._synthesise_submission(task))
        return Trajectory(
            instance_id=task.instance_id,
            repo=task.repo,
            complexity=self._classify(task),
            turns=turns,
            token_count=self._count_tokens(turns),
            num_tool_calls=sum(len(t.tool_calls) for t in turns),
            num_files_changed=len(file_changes),
        )

    def synthesise_all(self, tasks: list[Task]) -> list[Trajectory]:
        """Batch synthesize tasks and filter out over-budget trajectories."""
        trajectories: list[Trajectory] = []
        for task in tasks:
            trajectory = self.synthesise(task)
            if trajectory.token_count <= self._MAX_TRAJECTORY_TOKENS:
                trajectories.append(trajectory)
        return trajectories

    def _build_initial_prompt(self, task: Task) -> Turn:
        """Build the initial user turn containing instructions and budget."""
        prompt_text = (
            f"You are evaluating a software engineering task for repository "
            f"{task.repo}.\n\n"
            f"Problem Statement:\n{task.problem_statement}\n\n"
            f"## Hints:\n{task.hints_text}\n\n"
            f"## Task Budget (Session terminates when any budget is exhausted)\n"
            f"- Time allowance: 60.0 minutes\n"
            f"- Tool calls allowance: 100 calls\n\n"
            f"## Execution Environment Rules\n"
            f"- Single command timeout: 300 seconds\n"
            f"- Command output limit: 5000 characters\n"
            f"- File view limit: 150 lines per read_file call\n"
        )
        return Turn(role=self._ROLE_USER, text=prompt_text)

    def _synthesise_navigation(
        self, task: Task, graph: object, symbols: list[str]
    ) -> list[Turn]:
        """Create turns for search_similar_code and get_code_neighbors."""
        symbol = symbols[0] if symbols else self._DEFAULT_SYMBOL
        keywords = self._extract_keywords(task.problem_statement)
        thought_search = (
            f"The problem mentions {keywords}. Searching for {symbol}."
        )
        t1 = Turn(
            role=self._ROLE_MODEL,
            thought=thought_search,
            text=self._MSG_SEARCHING_CODE,
            tool_calls=[
                ToolCall(
                    self._TOOL_SEARCH,
                    {self._KEY_QUERY: symbol, "k": self._DEFAULT_QUERY_K},
                )
            ],
        )
        res_search = self._simulate_search(symbol, graph)
        t2 = Turn(role=self._ROLE_USER, tool_result=json.dumps(res_search))
        t3 = Turn(
            role=self._ROLE_MODEL,
            thought=f"Found {symbol}. Examining code neighbors.",
            text=self._MSG_EXAMINING_GRAPH,
            tool_calls=[
                ToolCall(
                    self._TOOL_NEIGHBORS,
                    {self._KEY_NODE: symbol},
                )
            ],
        )
        res_nbr = self._simulate_neighbors(symbol, graph)
        t4 = Turn(role=self._ROLE_USER, tool_result=json.dumps(res_nbr))
        return [t1, t2, t3, t4]

    def _synthesise_diagnosis(
        self, task: Task, file_changes: list[FileChange]
    ) -> list[Turn]:
        """Create turns for read_file calls and root cause diagnosis."""
        turns: list[Turn] = []
        for fc in file_changes:
            start_line, end_line = self._compute_read_range(fc)
            read_args = {
                self._KEY_FILEPATH: fc.filepath,
                self._KEY_START_LINE: start_line,
                self._KEY_END_LINE: end_line,
            }
            turns.append(
                Turn(
                    role=self._ROLE_MODEL,
                    thought=f"Reading {fc.filepath} around line {start_line}.",
                    text=f"Reading `{fc.filepath}`.",
                    tool_calls=[ToolCall(self._TOOL_READ_FILE, read_args)],
                )
            )
            read_res = self._simulate_read_file(
                fc.filepath, start_line, end_line
            )
            turns.append(
                Turn(role=self._ROLE_USER, tool_result=json.dumps(read_res))
            )
        turns.append(
            Turn(
                role=self._ROLE_MODEL,
                thought=self._ROOT_CAUSE_THOUGHT,
                text=self._ROOT_CAUSE_TEXT,
            )
        )
        return turns

    def _synthesise_patches(
        self, task: Task, file_changes: list[FileChange]
    ) -> list[Turn]:
        """Create edit_file tool call turns for each hunk."""
        turns: list[Turn] = []
        for fc in file_changes:
            for hunk in fc.hunks:
                turns.extend(self._create_hunk_edit_turns(fc.filepath, hunk))
        return turns

    def _synthesise_verification(self, task: Task) -> list[Turn]:
        """Create verification turn running test suite via run_command."""
        cmd_args: dict[str, object] = {self._KEY_COMMAND: self._PYTEST_CMD}
        t1 = Turn(
            role=self._ROLE_MODEL,
            thought="Running test suite to verify the applied fix.",
            text="Running verification tests.",
            tool_calls=[ToolCall(self._TOOL_RUN_COMMAND, cmd_args)],
        )
        res = self._simulate_pytest_result(task)
        t2 = Turn(role=self._ROLE_USER, tool_result=json.dumps(res))
        return [t1, t2]

    def _synthesise_submission(self, task: Task) -> list[Turn]:
        """Create final submit_patch turn and acknowledgement."""
        t1 = Turn(
            role=self._ROLE_MODEL,
            thought="All verification tests pass. Submitting completed patch.",
            text="All tests passed. Submitting patch.",
            tool_calls=[ToolCall(self._TOOL_SUBMIT_PATCH, {})],
        )
        submit_res = {
            self._KEY_STATUS: self._KEY_OK,
            self._KEY_MESSAGE: self._PATCH_SUBMITTED_MSG,
        }
        t2 = Turn(role=self._ROLE_USER, tool_result=json.dumps(submit_res))
        return [t1, t2]

    def _extract_symbols(
        self, file_changes: list[FileChange], graph: object
    ) -> list[str]:
        """Extract symbol identifiers from file hunks or code graph."""
        symbols: list[str] = []
        pattern = re.compile(r"^\s*(?:def|class)\s+([A-Za-z0-9_]+)")
        for fc in file_changes:
            for hunk in fc.hunks:
                for line in hunk.old_lines + hunk.new_lines:
                    match = pattern.match(line)
                    if match and match.group(1) not in symbols:
                        symbols.append(match.group(1))
        if not symbols and graph is not None:
            symbols.extend(self._extract_symbols_from_graph(graph))
        if not symbols:
            symbols.append(self._DEFAULT_SYMBOL)
        return symbols

    def _extract_keywords(self, problem_statement: str) -> str:
        """Extract key terms from problem statement."""
        words = re.findall(r"\b[A-Za-z_]{4,}\b", problem_statement.lower())
        filtered = [w for w in words if w not in self._STOPWORDS]
        if not filtered:
            return self._DEFAULT_KEYWORD
        return ", ".join(filtered[:3])

    def _simulate_search(self, query: str, graph: object) -> dict[str, object]:
        """Simulate search_similar_code tool result."""
        return {
            self._KEY_STATUS: self._KEY_OK,
            self._KEY_QUERY: query,
            self._KEY_RESULTS: [
                {
                    self._KEY_NODE_NAME: query,
                    self._KEY_FILEPATH: self._DEFAULT_FILEPATH,
                    self._KEY_SCORE: self._DEFAULT_SCORE,
                }
            ],
            self._KEY_COUNT: 1,
        }

    def _simulate_neighbors(
        self, node: str, graph: object
    ) -> dict[str, object]:
        """Simulate get_code_neighbors tool result."""
        return {
            self._KEY_STATUS: self._KEY_OK,
            self._KEY_NODE: node,
            self._KEY_NEIGHBORS: [
                {
                    self._KEY_SOURCE: node,
                    self._KEY_TARGET: f"{node}_caller",
                    self._KEY_TYPE: "calls",
                }
            ],
            self._KEY_COUNT: 1,
        }

    def _simulate_read_file(
        self, filepath: str, start_line: int, end_line: int
    ) -> dict[str, object]:
        """Simulate read_file tool result."""
        return {
            self._KEY_STATUS: self._KEY_OK,
            self._KEY_FILEPATH: filepath,
            self._KEY_START_LINE: start_line,
            self._KEY_END_LINE: end_line,
            self._KEY_CONTENT: (
                f"# Content for {filepath} lines {start_line}-{end_line}"
            ),
        }

    def _simulate_edit_result(self, filepath: str) -> dict[str, object]:
        """Simulate edit_file success result."""
        return {
            self._KEY_STATUS: self._KEY_OK,
            self._KEY_FILEPATH: filepath,
            self._KEY_OCCURRENCES: 1,
            self._KEY_STRATEGY: "exact",
        }

    def _simulate_pytest_result(self, task: Task) -> dict[str, object]:
        """Simulate pytest pass result."""
        return {
            self._KEY_STATUS: self._KEY_OK,
            self._KEY_RETURNCODE: 0,
            self._KEY_OUTPUT: self._PYTEST_OUTPUT,
        }

    def _split_hunk(self, hunk: Hunk) -> tuple[str, str]:
        """Split a large hunk into a smaller old_string and new_string."""
        old_sub = hunk.old_lines[: self._MAX_OLD_STRING_LINES]
        new_sub = hunk.new_lines[: self._MAX_OLD_STRING_LINES]
        return ("\n".join(old_sub), "\n".join(new_sub))

    def _classify(self, task: Task) -> ComplexityTier:
        """Classify task complexity tier."""
        return self._ingestor.classify_complexity(task)

    def _count_tokens(self, turns: list[Turn]) -> int:
        """Calculate token count across all turns."""
        tokens = 0
        for turn in turns:
            tokens += self._count_turn_tokens(turn)
        return tokens

    def _compute_read_range(self, fc: FileChange) -> tuple[int, int]:
        """Compute context start and end line for read_file."""
        start = 1
        end = 50
        if fc.hunks:
            first_hunk = fc.hunks[0]
            start = max(1, first_hunk.old_start - self._READ_CONTEXT_LINES)
            end = (
                first_hunk.old_start
                + max(1, first_hunk.old_count)
                + self._READ_CONTEXT_LINES
            )
        return (start, end)

    def _create_hunk_edit_turns(
        self, filepath: str, hunk: Hunk
    ) -> list[Turn]:
        """Build model edit_file turn and user ok response for a hunk."""
        old_str = "\n".join(hunk.old_lines)
        new_str = "\n".join(hunk.new_lines)
        if len(hunk.old_lines) > self._MAX_OLD_STRING_LINES:
            old_str, new_str = self._split_hunk(hunk)
        edit_args: dict[str, object] = {
            self._KEY_FILEPATH: filepath,
            self._KEY_OLD_STRING: old_str,
            self._KEY_NEW_STRING: new_str,
        }
        model_turn = Turn(
            role=self._ROLE_MODEL,
            thought=f"Fixing {filepath} at line {hunk.old_start}.",
            text=f"Applying patch to `{filepath}`.",
            tool_calls=[ToolCall(self._TOOL_EDIT_FILE, edit_args)],
        )
        res = self._simulate_edit_result(filepath)
        user_turn = Turn(role=self._ROLE_USER, tool_result=json.dumps(res))
        return [model_turn, user_turn]

    def _extract_symbols_from_graph(self, graph: object) -> list[str]:
        """Extract node names from code graph structure."""
        if isinstance(graph, dict) and "nodes" in graph:
            nodes = graph.get("nodes", [])
            if isinstance(nodes, list):
                return [
                    str(n.get("name", ""))
                    for n in nodes
                    if isinstance(n, dict) and "name" in n
                ]
        return []

    def _count_turn_tokens(self, turn: Turn) -> int:
        """Calculate token count for a single turn."""
        parts = [
            turn.role,
            turn.thought or "",
            turn.text or "",
            turn.tool_result or "",
        ]
        for tc in turn.tool_calls:
            parts.append(tc.tool_name)
            parts.append(json.dumps(tc.args))
        return self._token_counter.count_tokens(" ".join(parts))
