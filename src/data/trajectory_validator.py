"""Validator for synthesised and augmented agent trajectories."""

from __future__ import annotations

from src.data.trajectory import Trajectory
from src.data.validation_result import ValidationResult
from src.utils.token_counter import TokenCounter


class TrajectoryValidator:
    """Validates synthesised trajectories for training readiness and safety."""

    _MAX_TOKENS: int = 28672
    _MAX_TOOL_CALLS: int = 80
    _TOOL_EDIT_FILE: str = "edit_file"
    _TOOL_WRITE_FILE: str = "write_file"
    _TOOL_SUBMIT_PATCH: str = "submit_patch"
    _ARG_FILEPATH: str = "filepath"
    _ARG_OLD_STRING: str = "old_string"
    _TMP_PREFIX: str = "/tmp/"
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
    _WARN_SUBMIT_PATCH: str = "submit_patch is not the last tool call"
    _EMPTY_STR: str = ""

    def __init__(self, token_counter: TokenCounter) -> None:
        """Initialize validator with injected token counter."""
        self._token_counter: TokenCounter = token_counter

    def validate(self, trajectory: Trajectory) -> ValidationResult:
        """Run all validation checks and return a ValidationResult."""
        errors: list[str] = []
        warnings: list[str] = []
        errors.extend(self._check_token_budget(trajectory))
        errors.extend(self._check_edit_file_validity(trajectory))
        errors.extend(self._check_no_test_modifications(trajectory))
        errors.extend(self._check_scratch_file_paths(trajectory))
        warnings.extend(self._check_tool_call_budget(trajectory))
        warnings.extend(self._check_submit_patch_is_last(trajectory))
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_token_budget(self, trajectory: Trajectory) -> list[str]:
        """Check whether total trajectory tokens exceed the max limit."""
        if trajectory.token_count > self._MAX_TOKENS:
            return [
                f"Token count {trajectory.token_count} exceeds {self._MAX_TOKENS}"
            ]
        return []

    def _check_tool_call_budget(self, trajectory: Trajectory) -> list[str]:
        """Check whether total tool calls count exceeds recommended threshold."""
        if trajectory.num_tool_calls > self._MAX_TOOL_CALLS:
            return [
                f"Tool calls {trajectory.num_tool_calls} exceeds recommended "
                f"{self._MAX_TOOL_CALLS}"
            ]
        return []

    def _check_edit_file_validity(self, trajectory: Trajectory) -> list[str]:
        """Verify edit_file tool calls have a valid old_string argument."""
        errors: list[str] = []
        for turn in trajectory.turns:
            for tc in turn.tool_calls:
                if tc.tool_name == self._TOOL_EDIT_FILE:
                    old_str = str(tc.args.get(self._ARG_OLD_STRING, self._EMPTY_STR))
                    if not old_str:
                        path = str(tc.args.get(self._ARG_FILEPATH, self._EMPTY_STR))
                        errors.append(f"edit_file missing old_string for {path}")
        return errors

    def _check_no_test_modifications(self, trajectory: Trajectory) -> list[str]:
        """Verify no test or configuration files are modified."""
        errors: list[str] = []
        for turn in trajectory.turns:
            for tc in turn.tool_calls:
                if tc.tool_name in (self._TOOL_EDIT_FILE, self._TOOL_WRITE_FILE):
                    path = str(tc.args.get(self._ARG_FILEPATH, self._EMPTY_STR))
                    if self._is_test_file(path):
                        errors.append(f"Modifies test file: {path}")
        return errors

    def _check_scratch_file_paths(self, trajectory: Trajectory) -> list[str]:
        """Verify temporary/scratch files are created exclusively under /tmp/."""
        errors: list[str] = []
        for turn in trajectory.turns:
            for tc in turn.tool_calls:
                if tc.tool_name == self._TOOL_WRITE_FILE:
                    path = str(tc.args.get(self._ARG_FILEPATH, self._EMPTY_STR))
                    if self._is_scratch_file(path) and not path.startswith(
                        self._TMP_PREFIX
                    ):
                        errors.append(f"Scratch file in workspace: {path}")
        return errors

    def _check_submit_patch_is_last(self, trajectory: Trajectory) -> list[str]:
        """Verify submit_patch is the final tool call in the trajectory."""
        all_calls = [tc for turn in trajectory.turns for tc in turn.tool_calls]
        if not all_calls or all_calls[-1].tool_name != self._TOOL_SUBMIT_PATCH:
            return [self._WARN_SUBMIT_PATCH]
        return []

    def _is_test_file(self, filepath: str) -> bool:
        """Determine if a file path belongs to tests or test configs."""
        lowered = filepath.lower()
        return any(marker in lowered for marker in self._TEST_MARKERS)

    def _is_scratch_file(self, filepath: str) -> bool:
        """Determine if a file path is a scratch or reproduction file."""
        lowered = filepath.lower()
        return any(marker in lowered for marker in self._SCRATCH_MARKERS)
