"""Unit tests for TrajectoryAugmentor."""

from __future__ import annotations

from src.data.complexity_tier import ComplexityTier
from src.data.tool_call import ToolCall
from src.data.trajectory import Trajectory
from src.data.trajectory_augmentor import TrajectoryAugmentor
from src.data.turn import Turn


class TestTrajectoryAugmentor:
    """Test suite for TrajectoryAugmentor class."""

    _INSTANCE_ID: str = "mock_001"
    _REPO: str = "mock/repo"
    _FILEPATH: str = "src/utils.py"
    _TOOL_SEARCH: str = "search_similar_code"
    _TOOL_NEIGHBORS: str = "get_code_neighbors"
    _TOOL_READ: str = "read_file"
    _TOOL_EDIT: str = "edit_file"
    _TOOL_SUBMIT: str = "submit_patch"
    _TOOL_STATUS: str = "get_status"
    _TOOL_RUN: str = "run_command"
    _ROLE_USER: str = "user"
    _ROLE_MODEL: str = "model"
    _SEED: int = 42

    def test_augment_returns_list_of_trajectories(self) -> None:
        """Verify augment produces non-empty list of trajectories."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        results = augmentor.augment(traj)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_augment_preserves_instance_id(self) -> None:
        """Verify all augmented trajectories preserve the original instance_id."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        results = augmentor.augment(traj)
        for augmented in results:
            assert augmented.instance_id == traj.instance_id
            assert augmented.repo == traj.repo

    def test_permute_tool_order_changes_tool_sequence(self) -> None:
        """Verify navigation tool calls are permuted."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        permuted = augmentor._permute_tool_order(traj)
        first_tool_orig = traj.turns[1].tool_calls[0].tool_name
        first_tool_perm = permuted.turns[1].tool_calls[0].tool_name
        assert first_tool_orig == self._TOOL_SEARCH
        assert first_tool_perm == self._TOOL_NEIGHBORS

    def test_inject_error_recovery_adds_failed_edit(self) -> None:
        """Verify error recovery strategy injects failed edit and correction."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        with_recovery = augmentor._inject_error_recovery(traj)
        tools = [
            tc.tool_name
            for turn in with_recovery.turns
            for tc in turn.tool_calls
        ]
        assert tools.count(self._TOOL_EDIT) == 2
        assert any(
            turn.tool_result and "error" in turn.tool_result
            for turn in with_recovery.turns
        )

    def test_add_budget_checks_inserts_get_status(self) -> None:
        """Verify budget checks strategy inserts get_status tool calls."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        with_budget = augmentor._add_budget_checks(traj)
        status_calls = [
            tc
            for turn in with_budget.turns
            for tc in turn.tool_calls
            if tc.tool_name == self._TOOL_STATUS
        ]
        assert len(status_calls) == 2

    def test_augment_respects_max_augmentations(self) -> None:
        """Verify returned list never exceeds maximum augmentations limit."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        results = augmentor.augment(traj)
        assert len(results) <= 3

    def test_vary_read_windows_expands_bounds(self) -> None:
        """Verify vary read windows strategy adjusts line ranges."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        varied = augmentor._vary_read_windows(traj)
        read_calls = [
            tc
            for turn in varied.turns
            for tc in turn.tool_calls
            if tc.tool_name == self._TOOL_READ
        ]
        assert len(read_calls) > 0
        assert read_calls[0].args["start_line"] == 1
        assert int(str(read_calls[0].args["end_line"])) > 50

    def test_toggle_graph_vs_grep_replaces_search(self) -> None:
        """Verify search_similar_code is toggled into grep run_command."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        toggled = augmentor._toggle_graph_vs_grep(traj)
        tools = [
            tc.tool_name for turn in toggled.turns for tc in turn.tool_calls
        ]
        assert self._TOOL_SEARCH not in tools
        assert self._TOOL_RUN in tools

    def test_add_scratchpad_discipline_inserts_repro_turns(self) -> None:
        """Verify scratchpad strategy inserts reproduction and cleanup turns."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = self._build_sample_trajectory()
        scratch_traj = augmentor._add_scratchpad_discipline(traj)
        write_calls = [
            tc
            for turn in scratch_traj.turns
            for tc in turn.tool_calls
            if tc.tool_name == "write_file"
        ]
        assert len(write_calls) == 1
        assert "/tmp/repro.py" in str(write_calls[0].args.get("filepath"))

    def test_permute_tool_order_insufficient_navigation_calls(self) -> None:
        """Verify permute tool order returns unchanged trajectory when <2 nav calls."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[Turn(role=self._ROLE_USER, text="hi")],
            token_count=10,
            num_tool_calls=0,
            num_files_changed=0,
        )
        permuted = augmentor._permute_tool_order(traj)
        assert len(permuted.turns) == 1

    def test_inject_error_recovery_no_edit_file(self) -> None:
        """Verify error recovery returns unchanged clone if no edit_file call exists."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[Turn(role=self._ROLE_USER, text="hi")],
            token_count=10,
            num_tool_calls=0,
            num_files_changed=0,
        )
        recovered = augmentor._inject_error_recovery(traj)
        assert len(recovered.turns) == 1

    def test_toggle_graph_vs_grep_no_search_tool(self) -> None:
        """Verify toggle graph vs grep returns unchanged clone if no search call."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[Turn(role=self._ROLE_USER, text="hi")],
            token_count=10,
            num_tool_calls=0,
            num_files_changed=0,
        )
        toggled = augmentor._toggle_graph_vs_grep(traj)
        assert len(toggled.turns) == 1

    def test_add_budget_checks_short_trajectory(self) -> None:
        """Verify budget checks strategy returns clone when turns < 3."""
        augmentor = TrajectoryAugmentor(seed=self._SEED)
        traj = Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=[Turn(role=self._ROLE_USER, text="hi")],
            token_count=10,
            num_tool_calls=0,
            num_files_changed=0,
        )
        checked = augmentor._add_budget_checks(traj)
        assert len(checked.turns) == 1

    def _build_sample_trajectory(self) -> Trajectory:
        """Construct multi-turn trajectory with navigation, patch, and submit."""
        search_call = ToolCall(self._TOOL_SEARCH, {"query": "foo", "k": 5})
        nbr_call = ToolCall(self._TOOL_NEIGHBORS, {"node": "foo"})
        read_call = ToolCall(
            self._TOOL_READ,
            {"filepath": self._FILEPATH, "start_line": 10, "end_line": 50},
        )
        edit_call = ToolCall(
            self._TOOL_EDIT,
            {"filepath": self._FILEPATH, "old_string": "a", "new_string": "b"},
        )
        submit_call = ToolCall(self._TOOL_SUBMIT, {})
        turns = [
            Turn(role=self._ROLE_USER, text="Problem description"),
            Turn(role=self._ROLE_MODEL, tool_calls=[search_call]),
            Turn(role=self._ROLE_USER, tool_result='{"status": "ok"}'),
            Turn(role=self._ROLE_MODEL, tool_calls=[nbr_call]),
            Turn(role=self._ROLE_USER, tool_result='{"status": "ok"}'),
            Turn(role=self._ROLE_MODEL, tool_calls=[read_call]),
            Turn(role=self._ROLE_USER, tool_result='{"status": "ok"}'),
            Turn(role=self._ROLE_MODEL, tool_calls=[edit_call]),
            Turn(role=self._ROLE_USER, tool_result='{"status": "ok"}'),
            Turn(role=self._ROLE_MODEL, tool_calls=[submit_call]),
            Turn(role=self._ROLE_USER, tool_result='{"status": "ok"}'),
        ]
        return Trajectory(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            complexity=ComplexityTier.SIMPLE,
            turns=turns,
            token_count=1000,
            num_tool_calls=5,
            num_files_changed=1,
        )
