"""Unit tests for Trajectory data model."""

from src.data.complexity_tier import ComplexityTier
from src.data.trajectory import Trajectory
from src.data.turn import Turn


class TestTrajectory:
    """Test suite for Trajectory dataclass."""

    _INSTANCE_ID: str = "repo__task-1"
    _REPO: str = "owner/repo"
    _ROLE_USER: str = "user"
    _TOKEN_COUNT: int = 1500
    _NEW_TOKEN_COUNT: int = 2500
    _NUM_TOOL_CALLS: int = 4
    _NUM_FILES_CHANGED: int = 2
    _SAMPLE_PATCH: str = "diff --git a/a.py b/a.py"
    _ELAPSED_SECONDS: float = 12.5

    def _create_trajectory(self) -> Trajectory:
        """Helper to create a standard Trajectory instance."""
        return Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.MODERATE,
            turns=[Turn(role=self._ROLE_USER)],
            token_count=self._TOKEN_COUNT,
            num_tool_calls=self._NUM_TOOL_CALLS,
            num_files_changed=self._NUM_FILES_CHANGED,
        )

    def test_trajectory_creation(self) -> None:
        """Verify Trajectory initialization with explicit fields."""
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.MODERATE,
            turns=[Turn(role=self._ROLE_USER)],
            token_count=self._TOKEN_COUNT,
            num_tool_calls=self._NUM_TOOL_CALLS,
            num_files_changed=self._NUM_FILES_CHANGED,
            patch=self._SAMPLE_PATCH,
            had_truncation=True,
            elapsed_seconds=self._ELAPSED_SECONDS,
        )
        assert traj.instance_id == self._INSTANCE_ID
        assert traj.repo == self._REPO
        assert traj.complexity == ComplexityTier.MODERATE
        assert len(traj.turns) == 1
        assert traj.token_count == self._TOKEN_COUNT
        assert traj.num_tool_calls == self._NUM_TOOL_CALLS
        assert traj.num_files_changed == self._NUM_FILES_CHANGED
        assert traj.patch == self._SAMPLE_PATCH
        assert traj.had_truncation is True
        assert traj.elapsed_seconds == self._ELAPSED_SECONDS

    def test_trajectory_default_values(self) -> None:
        """Verify default values on Trajectory optional attributes."""
        traj = self._create_trajectory()
        assert traj.patch is None
        assert traj.had_truncation is False
        assert traj.elapsed_seconds == 0.0

    def test_trajectory_mutable(self) -> None:
        """Verify Trajectory is mutable so fields like token_count can be updated."""
        traj = self._create_trajectory()
        traj.token_count = self._NEW_TOKEN_COUNT
        assert traj.token_count == self._NEW_TOKEN_COUNT
