"""Unit tests for PeerAnalyser inspecting competitor notebooks and validation."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.evaluation.adversarial_result import AdversarialResult
from src.evaluation.cv_evaluator import CVEvaluator
from src.evaluation.cv_report import CVReport
from src.evaluation.cv_result import CVResult
from src.evaluation.cv_splitter import CVSplitter
from src.evaluation.inspection_check import InspectionCheck
from src.evaluation.inspection_report import InspectionReport
from src.evaluation.notebook_parser import NotebookParser
from src.evaluation.peer_analyser import PeerAnalyser
from src.evaluation.peer_solution import PeerSolution
from src.evaluation.task_result import TaskResult
from src.utils.telemetry_logger import TelemetryLogger


class TestPeerAnalyser:
    """Test suite covering static inspection, adversarial CV, significance, and decision logic."""

    def _create_analyser(
        self,
        cv_evaluator: MagicMock | None = None,
        splitter: MagicMock | None = None,
        notebook_parser: MagicMock | None = None,
        telemetry: MagicMock | None = None,
        baseline_adapter: str | None = None,
        quick_task_ids: list[str] | None = None,
    ) -> PeerAnalyser:
        """Create a PeerAnalyser with default or customized mocks."""
        cv_eval = cv_evaluator or MagicMock(spec=CVEvaluator)
        split = splitter or MagicMock(spec=CVSplitter)
        parser = notebook_parser or MagicMock(spec=NotebookParser)
        telem = telemetry or MagicMock(spec=TelemetryLogger)
        return PeerAnalyser(
            cv_evaluator=cv_eval,
            splitter=split,
            notebook_parser=parser,
            telemetry=telem,
            baseline_adapter_path=baseline_adapter,
            quick_eval_task_ids=quick_task_ids,
        )

    def _create_sample_solution(self, local_path: str = "sample.ipynb") -> PeerSolution:
        """Create a default sample PeerSolution for tests."""
        return PeerSolution(
            notebook_name="peer_v1",
            kaggle_url="https://kaggle.com/code/user/peer",
            author="competitor",
            lb_score=0.40,
            capture_date="2026-09-29",
            local_path=local_path,
        )

    # ------------------ Step 2: Code Inspection Tests ------------------

    def test_scan_for_leakage_clean_code(self) -> None:
        """Verify leakage check passes when code does not contain leakage keywords."""
        analyser = self._create_analyser()
        cells = [{"index": 0, "source": "import numpy as np\ndef solve(x): return x + 1"}]
        check = analyser._scan_for_leakage(cells)
        assert check.passed is True
        assert len(check.findings) == 0
        assert check.severity == "info"

    def test_scan_for_leakage_detects_test_patch(self) -> None:
        """Verify leakage scan flags critical finding when test_patch is used in agent logic."""
        analyser = self._create_analyser()
        cells = [{"index": 1, "source": "agent_test_patch = apply(test_patch)"}]
        check = analyser._scan_for_leakage(cells)
        assert check.passed is False
        assert any("CRITICAL" in f for f in check.findings)
        assert check.severity == "critical"

    def test_scan_for_leakage_detects_hardcoded_ids(self) -> None:
        """Verify leakage scan flags suspicious finding when more than 5 instance IDs are found."""
        analyser = self._create_analyser()
        code = (
            'tasks = ["fastapi_1", "fastapi_2", "fastapi_3", '
            '"requests_4", "requests_5", "httpx_6"]'
        )
        check = analyser._scan_for_leakage([{"index": 2, "source": code}])
        assert check.passed is False
        assert any("SUSPICIOUS" in f for f in check.findings)

    def test_scan_for_overfitting_clean(self) -> None:
        """Verify overfitting scan passes with clean code."""
        analyser = self._create_analyser()
        cells = [{"index": 0, "source": "model.train()\nloss.backward()"}]
        check = analyser._scan_for_overfitting(cells)
        assert check.passed is True
        assert len(check.findings) == 0

    def test_scan_for_overfitting_detects_manual_patches(self) -> None:
        """Verify overfitting scan detects hardcoded manual patches dictionary."""
        analyser = self._create_analyser()
        cells = [{"index": 1, "source": 'patch_map = {"issue_1": "diff..."}'}]
        check = analyser._scan_for_overfitting(cells)
        assert check.passed is False
        assert check.severity == "critical"
        assert any("Manual patch dictionary detected" in f for f in check.findings)

    def test_scan_for_overfitting_detects_low_epochs(self) -> None:
        """Verify overfitting scan detects suspicious single-epoch training configuration."""
        analyser = self._create_analyser()
        cells = [{"index": 1, "source": "num_train_epochs = 1"}]
        check = analyser._scan_for_overfitting(cells)
        assert check.passed is True  # Passed because not critical
        assert any("Only 1 training epoch" in f for f in check.findings)

    def test_inspect_returns_report(self) -> None:
        """Verify full inspection generates complete InspectionReport and updates solution."""
        parser = MagicMock(spec=NotebookParser)
        parser.load_notebook.return_value = {"cells": []}
        parser.extract_code_cells.return_value = [
            {"index": 0, "source": 'prompt = "System prompt here"\nagenttool = 1'},
            {"index": 1, "source": 'grpo_training = True\ntechnique = "reflection"'},
        ]
        analyser = self._create_analyser(notebook_parser=parser)
        solution = self._create_sample_solution()
        report = analyser.inspect(solution)
        assert isinstance(report, InspectionReport)
        assert report.architecture == "multi_agent"
        assert "System prompt here" in report.prompts
        assert report.training == "grpo_rl"
        assert "reflection" in report.novel_techniques
        assert solution.analysis_status == "inspected"

    def test_assess_risk_critical(self) -> None:
        """Verify risk assessment outputs critical when findings contain critical tag."""
        analyser = self._create_analyser()
        checks = [InspectionCheck(name="leak", passed=False, findings=["CRITICAL: test leak"])]
        assert analyser._assess_risk(checks) == "critical"

    def test_assess_risk_clean(self) -> None:
        """Verify risk assessment outputs clean when checks have no findings."""
        analyser = self._create_analyser()
        checks = [InspectionCheck(name="leak", passed=True, findings=[])]
        assert analyser._assess_risk(checks) == "clean"

    def test_assess_risk_medium_and_low(self) -> None:
        """Verify risk assessment produces medium for suspicious and low for warnings."""
        analyser = self._create_analyser()
        checks_med = [InspectionCheck(name="c", passed=False, findings=["SUSPICIOUS: count"])]
        assert analyser._assess_risk(checks_med) == "medium"
        checks_low = [InspectionCheck(name="c", passed=False, findings=["Minor keyword"])]
        assert analyser._assess_risk(checks_low) == "low"

    # ------------------ Step 3: Adversarial CV Tests ------------------

    def test_adversarial_cv_with_precomputed_results(self) -> None:
        """Verify adversarial_cv_test properly integrates precomputed CV report against baseline."""
        peer_tasks = [
            TaskResult("t1", "fastapi", ComplexityTier.SIMPLE, True, 10, 2, 50, False, 1.0)
        ]
        base_tasks = [
            TaskResult("t1", "fastapi", ComplexityTier.SIMPLE, False, 10, 2, 50, False, 1.0)
        ]
        peer_fold = CVResult(
            fold_idx=0,
            val_repo="fastapi",
            resolution_rate=0.45,
            resolved_count=9,
            total_count=20,
            per_task_results=peer_tasks,
        )
        base_fold = CVResult(
            fold_idx=0,
            val_repo="fastapi",
            resolution_rate=0.35,
            resolved_count=7,
            total_count=20,
            per_task_results=base_tasks,
        )
        peer_report = CVReport(
            "peer_v1", "ts", [peer_fold], 0.45, {"fastapi": 0.45}, {}, "none", []
        )
        base_report = CVReport(
            "baseline", "ts", [base_fold], 0.35, {"fastapi": 0.35}, {}, "none", []
        )
        cv_eval = MagicMock(spec=CVEvaluator)
        cv_eval.evaluate_all_folds.return_value = base_report
        analyser = self._create_analyser(cv_evaluator=cv_eval)
        solution = self._create_sample_solution()
        adv_res = analyser.adversarial_cv_test(solution, tasks=[], precomputed_cv=peer_report)
        assert adv_res.peer_cv_score == 0.45
        assert adv_res.baseline_cv_score == 0.35
        assert round(adv_res.improvement, 2) == 0.10
        assert "fastapi" in adv_res.per_fold_comparison

    def test_adversarial_cv_runs_live_cv_when_no_precomputed(self) -> None:
        """Verify adversarial_cv_test triggers evaluate_all_folds when precomputed is None."""
        fold = CVResult(0, "requests", 0.5, 5, 10, per_task_results=[])
        report = CVReport("v", "ts", [fold], 0.5, {}, {}, "none", [])
        cv_eval = MagicMock(spec=CVEvaluator)
        cv_eval.evaluate_all_folds.return_value = report
        analyser = self._create_analyser(cv_evaluator=cv_eval)
        solution = self._create_sample_solution()
        adv_res = analyser.adversarial_cv_test(solution, tasks=[])
        assert adv_res.peer_cv_score == 0.5
        assert cv_eval.evaluate_all_folds.call_count == 2

    def test_bootstrap_significance_test_significant(self) -> None:
        """Verify bootstrap significance test identifies marked improvement as significant."""
        analyser = self._create_analyser()
        treatment = [
            TaskResult(f"id_{i}", "repo", ComplexityTier.SIMPLE, True, 10, 2, 50, False, 1.0)
            for i in range(25)
        ]
        control = [
            TaskResult(f"id_{i}", "repo", ComplexityTier.SIMPLE, False, 10, 2, 50, False, 1.0)
            for i in range(25)
        ]
        is_sig = analyser._bootstrap_significance_test(treatment, control, confidence=0.95)
        assert is_sig is True

    def test_bootstrap_significance_test_not_significant(self) -> None:
        """Verify bootstrap test detects no significant difference between identical sets."""
        analyser = self._create_analyser()
        treatment = [
            TaskResult(f"id_{i}", "repo", ComplexityTier.SIMPLE, i % 2 == 0, 10, 2, 50, False, 1.0)
            for i in range(20)
        ]
        control = [
            TaskResult(f"id_{i}", "repo", ComplexityTier.SIMPLE, i % 2 == 0, 10, 2, 50, False, 1.0)
            for i in range(20)
        ]
        is_sig = analyser._bootstrap_significance_test(treatment, control, confidence=0.95)
        assert is_sig is False

    def test_bootstrap_significance_test_empty_or_worse(self) -> None:
        """Verify bootstrap test returns False when inputs are empty or treatment is worse."""
        analyser = self._create_analyser()
        assert analyser._bootstrap_significance_test([], []) is False
        treatment = [TaskResult("1", "repo", ComplexityTier.SIMPLE, False, 0, 0, 0, False, 0.0)]
        control = [TaskResult("1", "repo", ComplexityTier.SIMPLE, True, 0, 0, 0, False, 0.0)]
        assert analyser._bootstrap_significance_test(treatment, control) is False

    # ------------------ Step 4: Decision Matrix Tests ------------------

    def test_make_decision_reject_critical_risk(self) -> None:
        """Verify critical risk forces immediate rejection regardless of metrics."""
        analyser = self._create_analyser()
        sol = self._create_sample_solution()
        report = InspectionReport(solution=sol, checks=[], risk_level="critical")
        adv = AdversarialResult(0.6, 0.3, 0.3, 0.05, True, {})
        decision = analyser.make_decision(report, adv)
        assert decision.action == "REJECT"
        assert decision.confidence == "high"

    def test_make_decision_reject_high_gap(self) -> None:
        """Verify CV-LB gap exceeding 20% forces rejection due to test-set overfitting risk."""
        analyser = self._create_analyser()
        sol = self._create_sample_solution()
        report = InspectionReport(solution=sol, checks=[], risk_level="clean")
        adv = AdversarialResult(0.5, 0.4, 0.10, 0.25, True, {})
        decision = analyser.make_decision(report, adv)
        assert decision.action == "REJECT"
        assert "exceeds 20% threshold" in decision.reason

    def test_make_decision_adopt_large_improvement(self) -> None:
        """Verify ADOPT when improvement > 5%, significant, gap < 10%, >= 3 folds improve."""
        analyser = self._create_analyser()
        sol = self._create_sample_solution()
        report = InspectionReport(
            solution=sol, checks=[], risk_level="clean", novel_techniques=["reflection"]
        )
        per_fold = {
            "repo1": {"delta": 0.06},
            "repo2": {"delta": 0.08},
            "repo3": {"delta": 0.05},
            "repo4": {"delta": -0.01},
        }
        adv = AdversarialResult(0.55, 0.48, 0.07, 0.04, True, per_fold)
        decision = analyser.make_decision(report, adv)
        assert decision.action == "ADOPT"
        assert decision.confidence == "high"
        assert decision.components_to_adopt == ["reflection"]

    def test_make_decision_partial_adopt(self) -> None:
        """Verify PARTIAL_ADOPT when improvement 2-5% and significant."""
        analyser = self._create_analyser()
        sol = self._create_sample_solution()
        report = InspectionReport(
            solution=sol, checks=[], risk_level="low", novel_techniques=["context_pruning"]
        )
        adv = AdversarialResult(0.44, 0.41, 0.03, 0.12, True, {})
        decision = analyser.make_decision(report, adv)
        assert decision.action == "PARTIAL_ADOPT"
        assert decision.confidence == "medium"
        assert decision.components_to_adopt == ["context_pruning"]

    def test_make_decision_reject_insufficient_improvement(self) -> None:
        """Verify REJECT when improvement is below 2% threshold."""
        analyser = self._create_analyser()
        sol = self._create_sample_solution()
        report = InspectionReport(solution=sol, checks=[], risk_level="clean")
        adv = AdversarialResult(0.40, 0.395, 0.005, 0.05, False, {})
        decision = analyser.make_decision(report, adv)
        assert decision.action == "REJECT"
        assert "Insufficient improvement" in decision.reason

    def test_isolate_beneficial_techniques_with_quick_eval(self) -> None:
        """Verify _isolate_beneficial_techniques evaluates isolated techniques when tasks exist."""
        cv_eval = MagicMock(spec=CVEvaluator)
        cv_eval.quick_evaluate.side_effect = [
            CVResult(-1, "m", 0.60, 6, 10, []),  # technique result (+20% delta)
            CVResult(-1, "m", 0.40, 4, 10, []),  # baseline result
        ]
        sample_task = Task(
            "t1", "repo", "commit", "problem", "hints", "patch", "test_patch", "2026-09-01"
        )
        analyser = self._create_analyser(cv_evaluator=cv_eval, quick_task_ids=["t1"])
        analyser._tasks = [sample_task]
        adv = AdversarialResult(0.45, 0.40, 0.05, 0.05, True, {})
        beneficial = analyser._isolate_beneficial_techniques(["reflection"], adv)
        assert beneficial == ["reflection"]

    def test_isolate_beneficial_techniques_skips_unimproved(self) -> None:
        """Verify _isolate_beneficial_techniques excludes techniques with delta <= 2%."""
        cv_eval = MagicMock(spec=CVEvaluator)
        cv_eval.quick_evaluate.side_effect = [
            CVResult(-1, "m", 0.41, 4, 10, []),  # technique result (+1% delta)
            CVResult(-1, "m", 0.40, 4, 10, []),  # baseline result
        ]
        sample_task = Task("t1", "r", "c", "p", "h", "pt", "tp", "2026-09-01")
        analyser = self._create_analyser(cv_evaluator=cv_eval, quick_task_ids=["t1"])
        analyser._tasks = [sample_task]
        adv = AdversarialResult(0.45, 0.40, 0.05, 0.05, True, {})
        beneficial = analyser._isolate_beneficial_techniques(["low_gain_tech"], adv)
        assert beneficial == []

    def test_isolate_beneficial_techniques_fallback_when_no_tasks(self) -> None:
        """Verify _isolate_beneficial_techniques returns techniques as-is when tasks unavailable."""
        analyser = self._create_analyser()
        adv = AdversarialResult(0.45, 0.40, 0.05, 0.05, True, {})
        assert analyser._isolate_beneficial_techniques(["tot"], adv) == ["tot"]

    def test_make_decision_reject_inconsistent_folds(self) -> None:
        """Verify REJECT when CV improvement is high but improves < 3 folds."""
        analyser = self._create_analyser()
        sol = self._create_sample_solution()
        report = InspectionReport(solution=sol, checks=[], risk_level="clean")
        per_fold = {
            "r1": {"delta": 0.15},
            "r2": {"delta": -0.05},
            "r3": {"delta": -0.02},
            "r4": {"delta": -0.01},
        }
        adv = AdversarialResult(0.60, 0.50, 0.10, 0.05, True, per_fold)
        decision = analyser.make_decision(report, adv)
        assert decision.action == "REJECT"

    def test_extract_architecture_and_training_variants(self) -> None:
        """Verify architecture and training strategy detection across patterns."""
        analyser = self._create_analyser()
        react_cells = [{"source": "def react_agent_loop(): pass"}]
        assert analyser._extract_architecture(react_cells) == "single_agent_react"
        std_cells = [{"source": "x = 10"}]
        assert analyser._extract_architecture(std_cells) == "standard_agent"

        assert analyser._extract_training_strategy([{"source": "dpo_trainer = 1"}]) == "dpo_rl"
        assert analyser._extract_training_strategy([{"source": "lora_config = 1"}]) == "sft_lora"
        sup_strat = analyser._extract_training_strategy([{"source": "def train(): pass"}])
        assert sup_strat == "supervised_training"
        assert analyser._extract_training_strategy([{"source": "pass"}]) == "prompt_only"

    def test_identify_novel_techniques_variants(self) -> None:
        """Verify detection of various novel techniques."""
        analyser = self._create_analyser()
        code = "tot = True; dynamic_budget = 10; pruning = True; best_of_n = 5; subagent = True"
        techs = analyser._identify_novel_techniques([{"source": code}])
        assert "tree_of_thought" in techs
        assert "dynamic_budgeting" in techs
        assert "context_pruning" in techs
        assert "best_of_n_sampling" in techs
        assert "multi_agent_delegation" in techs
