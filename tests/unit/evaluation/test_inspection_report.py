"""Unit tests for InspectionReport data model."""

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.inspection_check import InspectionCheck
from src.evaluation.inspection_report import InspectionReport
from src.evaluation.peer_solution import PeerSolution


class TestInspectionReport:
    """Test suite for InspectionReport frozen dataclass."""

    _NOTEBOOK: str = "peer.ipynb"
    _URL: str = "https://kaggle.com/nb"
    _AUTHOR: str = "author"
    _LB_SCORE: float = 0.55
    _DATE: str = "2026-10-01"
    _PATH: str = "/tmp/peer.ipynb"
    _CHECK_NAME: str = "leak_check"
    _DEFAULT_RISK: str = "low"
    _CUSTOM_RISK: str = "critical"
    _MODIFIED_RISK: str = "medium"
    _ARCH: str = "SingleAgent"
    _PROMPT: str = "You are a SWE agent"
    _TRAINING: str = "SFT 3 epochs"
    _NOVEL: str = "AST search"

    def _create_solution(self) -> PeerSolution:
        """Helper to create a standard PeerSolution."""
        return PeerSolution(
            notebook_name=self._NOTEBOOK,
            kaggle_url=self._URL,
            author=self._AUTHOR,
            lb_score=self._LB_SCORE,
            capture_date=self._DATE,
            local_path=self._PATH,
        )

    def test_inspection_report_defaults(self) -> None:
        """Verify default values on InspectionReport optional fields."""
        solution = self._create_solution()
        check = InspectionCheck(name=self._CHECK_NAME, passed=True, findings=[])
        report = InspectionReport(solution=solution, checks=[check])

        assert report.solution == solution
        assert len(report.checks) == 1
        assert report.architecture is None
        assert report.prompts == []
        assert report.training is None
        assert report.novel_techniques == []
        assert report.risk_level == self._DEFAULT_RISK

    def test_inspection_report_custom_fields(self) -> None:
        """Verify InspectionReport with explicit values for optional fields."""
        solution = self._create_solution()
        check = InspectionCheck(name=self._CHECK_NAME, passed=False, findings=[])
        report = InspectionReport(
            solution=solution,
            checks=[check],
            architecture=self._ARCH,
            prompts=[self._PROMPT],
            training=self._TRAINING,
            novel_techniques=[self._NOVEL],
            risk_level=self._CUSTOM_RISK,
        )
        assert report.architecture == self._ARCH
        assert report.prompts == [self._PROMPT]
        assert report.training == self._TRAINING
        assert report.novel_techniques == [self._NOVEL]
        assert report.risk_level == self._CUSTOM_RISK

    def test_inspection_report_is_frozen(self) -> None:
        """Verify modifying field on InspectionReport raises FrozenInstanceError."""
        solution = self._create_solution()
        check = InspectionCheck(name=self._CHECK_NAME, passed=True, findings=[])
        report = InspectionReport(solution=solution, checks=[check])
        with pytest.raises(FrozenInstanceError):
            report.risk_level = self._MODIFIED_RISK
