"""Unit tests for PeerSolution data model."""

from src.evaluation.peer_solution import PeerSolution


class TestPeerSolution:
    """Test suite for PeerSolution dataclass."""

    _NOTEBOOK: str = "competitor_nb.ipynb"
    _URL: str = "https://kaggle.com/competitor/nb"
    _AUTHOR: str = "peer_user"
    _LB_SCORE: float = 0.58
    _DATE: str = "2026-10-01"
    _PATH: str = "/tmp/nb.ipynb"
    _NEW_STATUS: str = "inspected"
    _NEW_CV_SCORE: float = 0.52

    def _create_solution(self) -> PeerSolution:
        """Helper to create a standard PeerSolution instance."""
        return PeerSolution(
            notebook_name=self._NOTEBOOK,
            kaggle_url=self._URL,
            author=self._AUTHOR,
            lb_score=self._LB_SCORE,
            capture_date=self._DATE,
            local_path=self._PATH,
        )

    def test_peer_solution_creation_and_defaults(self) -> None:
        """Verify default values on PeerSolution optional attributes."""
        solution = self._create_solution()
        assert solution.notebook_name == self._NOTEBOOK
        assert solution.kaggle_url == self._URL
        assert solution.author == self._AUTHOR
        assert solution.lb_score == self._LB_SCORE
        assert solution.capture_date == self._DATE
        assert solution.local_path == self._PATH
        assert solution.analysis_status == "pending"
        assert solution.prompt_strategy is None
        assert solution.agent_architecture is None
        assert solution.adapter_details is None
        assert solution.key_techniques == []
        assert solution.cv_score is None
        assert solution.cv_lb_gap is None
        assert solution.leak_detected is None

    def test_peer_solution_mutable(self) -> None:
        """Verify PeerSolution fields can be updated during analysis."""
        solution = self._create_solution()
        solution.analysis_status = self._NEW_STATUS
        solution.cv_score = self._NEW_CV_SCORE
        assert solution.analysis_status == self._NEW_STATUS
        assert solution.cv_score == self._NEW_CV_SCORE
