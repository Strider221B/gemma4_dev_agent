"""Unit tests for TrajectoryValidator."""

from __future__ import annotations

from src.data.complexity_tier import ComplexityTier
from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.trajectory_validator import TrajectoryValidator
from src.data.turn import Turn
from src.utils.token_counter import TokenCounter


class TestTrajectoryValidator:
    """Test suite for TrajectoryValidator class."""

    _INSTANCE_ID: str = "mock_001"
    _REPO_NAME: str = "mock/repo"
    _SRC_FILE: str = "src/utils.py"
    _TEST_FILE: str = "tests/test_utils.py"
    _SCRATCH_FILE_WS: str = "repro.py"
    _SCRATCH_FILE_TMP: str = "/tmp/repro.py"
    _OLD_CODE: str = "def add(): return 1"
    _NEW_CODE: str = "def add(): return 2"
    _OVER_BUDGET_TOKENS: int = 30000
    _NORMAL_TOKENS: int = 500
    _HIGH_TOOL_CALLS: int = 85
    _TOOL_EDIT: str = "edit_file"
    _TOOL_WRITE: str = "write_file"
    _TOOL_SUBMIT: str = "submit_patch"
    _ROLE_USER: str = "user"
    _ROLE_MODEL: str = "model"

    def test_validate_valid_trajectory_returns_valid(self) -> None:
        """Verify that a compliant trajectory returns valid with no errors."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        result = validator.validate(traj)
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_validate_over_token_budget_returns_invalid(self) -> None:
        """Verify trajectory exceeding max tokens fails validation."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        traj.token_count = self._OVER_BUDGET_TOKENS
        result = validator.validate(traj)
        assert result.valid is False
        assert any("Token count" in err for err in result.errors)

    def test_validate_tool_call_budget_warns(self) -> None:
        """Verify exceeding tool call recommendation produces a warning."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        traj.num_tool_calls = self._HIGH_TOOL_CALLS
        result = validator.validate(traj)
        assert result.valid is True
        assert any("Tool calls" in warn for warn in result.warnings)

    def test_validate_missing_old_string_returns_invalid(self) -> None:
        """Verify edit_file with empty old_string produces validation error."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        bad_edit = ToolCall(
            self._TOOL_EDIT, {"filepath": self._SRC_FILE, "old_string": ""}
        )
        traj.turns[1] = Turn(role=self._ROLE_MODEL, tool_calls=[bad_edit])
        result = validator.validate(traj)
        assert result.valid is False
        assert any("missing old_string" in err for err in result.errors)

    def test_validate_test_file_modification_returns_invalid(self) -> None:
        """Verify trajectory editing test files produces validation error."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        test_edit = ToolCall(
            self._TOOL_EDIT,
            {"filepath": self._TEST_FILE, "old_string": "x", "new_string": "y"},
        )
        traj.turns[1] = Turn(role=self._ROLE_MODEL, tool_calls=[test_edit])
        result = validator.validate(traj)
        assert result.valid is False
        assert any("Modifies test file" in err for err in result.errors)

    def test_validate_scratch_in_workspace_returns_invalid(self) -> None:
        """Verify writing scratch files inside workspace produces error."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        write_call = ToolCall(
            self._TOOL_WRITE,
            {"filepath": self._SCRATCH_FILE_WS, "content": "print(1)"},
        )
        traj.turns.insert(
            1, Turn(role=self._ROLE_MODEL, tool_calls=[write_call])
        )
        result = validator.validate(traj)
        assert result.valid is False
        assert any("Scratch file in workspace" in err for err in result.errors)

    def test_validate_scratch_in_tmp_is_valid(self) -> None:
        """Verify writing scratch files in /tmp/ is accepted."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        write_call = ToolCall(
            self._TOOL_WRITE,
            {"filepath": self._SCRATCH_FILE_TMP, "content": "print(1)"},
        )
        traj.turns.insert(
            1, Turn(role=self._ROLE_MODEL, tool_calls=[write_call])
        )
        result = validator.validate(traj)
        assert result.valid is True

    def test_validate_no_submit_patch_warns(self) -> None:
        """Verify missing submit_patch as last tool call produces a warning."""
        validator = TrajectoryValidator(TokenCounter())
        traj = self._build_valid_trajectory()
        traj.turns = traj.turns[:3]
        result = validator.validate(traj)
        assert result.valid is True
        assert any(
            "submit_patch is not the last" in warn for warn in result.warnings
        )

    def _build_valid_trajectory(self) -> Trajectory:
        """Construct a minimal valid Trajectory instance."""
        edit_args: dict[str, object] = {
            "filepath": self._SRC_FILE,
            "old_string": self._OLD_CODE,
            "new_string": self._NEW_CODE,
        }
        turns = [
            Turn(role=self._ROLE_USER, text="Problem statement"),
            Turn(
                role=self._ROLE_MODEL,
                tool_calls=[ToolCall(self._TOOL_EDIT, edit_args)],
            ),
            Turn(role=self._ROLE_USER, tool_result='{"status": "ok"}'),
            Turn(
                role=self._ROLE_MODEL,
                tool_calls=[ToolCall(self._TOOL_SUBMIT, {})],
            ),
            Turn(role=self._ROLE_USER, tool_result='{"status": "ok"}'),
        ]
        return Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO_NAME,
            complexity=ComplexityTier.SIMPLE,
            turns=turns,
            token_count=self._NORMAL_TOKENS,
            num_tool_calls=2,
            num_files_changed=1,
        )
