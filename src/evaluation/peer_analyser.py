"""Adversarial evaluation and competitive inspection for peer solutions."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.data.task import Task
from src.evaluation.adversarial_result import AdversarialResult
from src.evaluation.cv_evaluator import CVEvaluator
from src.evaluation.cv_report import CVReport
from src.evaluation.cv_splitter import CVSplitter
from src.evaluation.decision import Decision
from src.evaluation.inspection_check import InspectionCheck
from src.evaluation.inspection_report import InspectionReport
from src.evaluation.notebook_parser import NotebookParser
from src.evaluation.peer_solution import PeerSolution
from src.evaluation.task_result import TaskResult
from src.utils.telemetry_logger import TelemetryLogger


class PeerAnalyser:
    """Performs static code inspection, adversarial CV validation, and adoption decision."""

    _LEAKAGE_KEYWORDS: list[str] = [
        "test_patch", "instance_id", "solution", "answer",
        "ground_truth", "gold_patch", "reference_patch",
    ]
    _OVERFITTING_INDICATORS: list[str] = [
        "specific task", "hardcoded", "manual fix", "if instance_id ==", "special case",
    ]
    _ADOPT_IMPROVEMENT_THRESHOLD: float = 0.05
    _PARTIAL_IMPROVEMENT_THRESHOLD: float = 0.02
    _MAX_CV_LB_GAP: float = 0.20
    _ADOPT_CV_LB_GAP_THRESHOLD: float = 0.10
    _MIN_IMPROVING_FOLDS: int = 3
    _SUSPICIOUS_INSTANCE_ID_COUNT: int = 5
    _BOOTSTRAP_SAMPLES: int = 10000
    _BOOTSTRAP_SEED: int = 42
    _DEFAULT_CONFIDENCE: float = 0.95

    _SEVERITY_CRITICAL: str = "critical"
    _SEVERITY_WARNING: str = "warning"
    _SEVERITY_INFO: str = "info"
    _RISK_CLEAN: str = "clean"
    _RISK_LOW: str = "low"
    _RISK_MEDIUM: str = "medium"
    _RISK_CRITICAL: str = "critical"
    _ACTION_ADOPT: str = "ADOPT"
    _ACTION_PARTIAL: str = "PARTIAL_ADOPT"
    _ACTION_REJECT: str = "REJECT"
    _CONFIDENCE_HIGH: str = "high"
    _CONFIDENCE_MEDIUM: str = "medium"

    _CHECK_LEAKAGE: str = "data_leakage"
    _CHECK_OVERFITTING: str = "overfitting_patterns"
    _STATUS_INSPECTED: str = "inspected"
    _VERSION_BASELINE: str = "baseline"
    _VERSION_PEER_PREFIX: str = "peer_"

    _KEY_SOURCE: str = "source"
    _KEY_INDEX: str = "index"
    _KEY_PEER: str = "peer"
    _KEY_BASELINE: str = "baseline"
    _KEY_DELTA: str = "delta"
    _CRITICAL_TAG: str = "CRITICAL"
    _SUSPICIOUS_TAG: str = "SUSPICIOUS"

    _INSTANCE_ID_PATTERN: str = r'["\'](?:fastapi|rich|requests|httpx)_\d+["\']'
    _EPOCH_PATTERN: str = r'num_(?:train_)?epochs\s*[=:]\s*(\d+)'
    _SUSPICIOUS_EPOCH_MSG: str = "SUSPICIOUS: Only 1 training epoch — likely prompt-only submission"
    _REASON_CRITICAL_RISK: str = "Critical leakage or overfitting detected"

    def __init__(
        self,
        cv_evaluator: CVEvaluator,
        splitter: CVSplitter,
        notebook_parser: NotebookParser,
        telemetry: TelemetryLogger,
        baseline_adapter_path: str | None = None,
        quick_eval_task_ids: list[str] | None = None,
    ) -> None:
        """Initialize the peer solution analyser with evaluation dependencies."""
        self._cv_evaluator: CVEvaluator = cv_evaluator
        self._splitter: CVSplitter = splitter
        self._notebook_parser: NotebookParser = notebook_parser
        self._telemetry: TelemetryLogger = telemetry
        self._baseline_adapter_path: str | None = baseline_adapter_path
        self._quick_eval_task_ids: list[str] | None = quick_eval_task_ids
        self._tasks: list[Task] = []

    def inspect(self, solution: PeerSolution) -> InspectionReport:
        """Perform static inspection of competitor notebook code and architecture."""
        nb_dict = self._notebook_parser.load_notebook(solution.local_path)
        cells = self._notebook_parser.extract_code_cells(nb_dict)
        leakage_check = self._scan_for_leakage(cells)
        overfitting_check = self._scan_for_overfitting(cells)
        checks = [leakage_check, overfitting_check]
        arch = self._extract_architecture(cells)
        prompts = self._extract_prompts(cells)
        training = self._extract_training_strategy(cells)
        techniques = self._identify_novel_techniques(cells)
        risk = self._assess_risk(checks)
        self._update_solution_metadata(solution, checks, arch, prompts, training, techniques)
        return InspectionReport(
            solution=solution,
            checks=checks,
            architecture=arch,
            prompts=prompts,
            training=training,
            novel_techniques=techniques,
            risk_level=risk,
        )

    def adversarial_cv_test(
        self,
        solution: PeerSolution,
        tasks: list[Task],
        precomputed_cv: CVReport | None = None,
    ) -> AdversarialResult:
        """Run the peer solution through cross-validation against our baseline."""
        self._tasks = tasks
        peer_cv = self._obtain_peer_cv_result(solution, tasks, precomputed_cv)
        base_cv = self._cv_evaluator.evaluate_all_folds(
            adapter_path=self._baseline_adapter_path or "",
            tasks=tasks,
            version=self._VERSION_BASELINE,
        )
        improvement = peer_cv.aggregate_resolution_rate - base_cv.aggregate_resolution_rate
        cv_lb_gap = abs(peer_cv.aggregate_resolution_rate - solution.lb_score)
        cv_tasks = [t for f in peer_cv.fold_results for t in f.per_task_results]
        base_tasks = [t for f in base_cv.fold_results for t in f.per_task_results]
        is_sig = self._bootstrap_significance_test(
            cv_tasks, base_tasks, confidence=self._DEFAULT_CONFIDENCE
        )
        per_fold = self._build_per_fold_comparison(peer_cv, base_cv)
        return AdversarialResult(
            peer_cv_score=peer_cv.aggregate_resolution_rate,
            baseline_cv_score=base_cv.aggregate_resolution_rate,
            improvement=improvement,
            cv_lb_gap=cv_lb_gap,
            is_significant=is_sig,
            per_fold_comparison=per_fold,
        )

    def make_decision(
        self, report: InspectionReport, adversarial: AdversarialResult
    ) -> Decision:
        """Apply decision matrix to determine adoption strategy."""
        if report.risk_level == self._RISK_CRITICAL:
            return Decision(
                action=self._ACTION_REJECT,
                reason=self._REASON_CRITICAL_RISK,
                confidence=self._CONFIDENCE_HIGH,
            )
        if adversarial.cv_lb_gap > self._MAX_CV_LB_GAP:
            gap_str = f"{adversarial.cv_lb_gap:.1%}"
            reason_msg = (
                f"CV-LB gap {gap_str} exceeds 20% threshold — likely overfit to public test set"
            )
            return Decision(
                action=self._ACTION_REJECT,
                reason=reason_msg,
                confidence=self._CONFIDENCE_HIGH,
            )
        adopt = self._check_adopt_criteria(report, adversarial)
        if adopt is not None:
            return adopt
        partial = self._check_partial_adopt_criteria(report, adversarial)
        if partial is not None:
            return partial
        imp_str = f"+{adversarial.improvement:.1%}"
        return Decision(
            action=self._ACTION_REJECT,
            reason=f"Insufficient improvement ({imp_str}) or inconsistent across folds",
            confidence=self._CONFIDENCE_MEDIUM,
        )

    def _update_solution_metadata(
        self,
        solution: PeerSolution,
        checks: list[InspectionCheck],
        arch: str,
        prompts: list[str],
        training: str,
        techniques: list[str],
    ) -> None:
        """Update the mutable metadata fields of the inspected PeerSolution."""
        solution.analysis_status = self._STATUS_INSPECTED
        solution.agent_architecture = arch
        solution.prompt_strategy = "\n".join(prompts) if prompts else None
        solution.adapter_details = training
        solution.key_techniques = techniques
        leak_check = next((c for c in checks if c.name == self._CHECK_LEAKAGE), None)
        solution.leak_detected = not leak_check.passed if leak_check else False

    def _scan_for_leakage(
        self, code_cells: list[dict[str, Any]]
    ) -> InspectionCheck:
        """Detect potential data leakage in code cells."""
        findings: list[str] = []
        for cell in code_cells:
            code = str(cell.get(self._KEY_SOURCE, ""))
            idx = int(str(cell.get(self._KEY_INDEX, 0)))
            self._check_cell_leakage(code, idx, findings)
        severity = self._SEVERITY_CRITICAL if any(
            self._CRITICAL_TAG in f for f in findings
        ) else self._SEVERITY_WARNING
        passed = len(findings) == 0
        return InspectionCheck(
            name=self._CHECK_LEAKAGE,
            passed=passed,
            findings=findings,
            severity=self._SEVERITY_INFO if passed else severity,
        )

    def _check_cell_leakage(
        self, code: str, idx: int, findings: list[str]
    ) -> None:
        """Check a single code cell for data leakage indicators."""
        code_lower = code.lower()
        for kw in self._LEAKAGE_KEYWORDS:
            if kw in code_lower:
                findings.append(f"Leakage keyword '{kw}' found in cell {idx}")
        if "test_patch" in code and ("agent" in code or "inference" in code):
            findings.append("CRITICAL: test_patch accessed during inference pipeline")
        matches = re.findall(self._INSTANCE_ID_PATTERN, code)
        if len(matches) > self._SUSPICIOUS_INSTANCE_ID_COUNT:
            findings.append(f"SUSPICIOUS: {len(matches)} hardcoded instance_ids found")
        if ("/test_" in code or "test_patch" in code) and ("apply" in code or "patch" in code):
            findings.append("WARNING: Possible test patch application in agent logic")

    def _scan_for_overfitting(
        self, code_cells: list[dict[str, Any]]
    ) -> InspectionCheck:
        """Detect patterns indicative of public LB overfitting."""
        findings: list[str] = []
        for cell in code_cells:
            code = str(cell.get(self._KEY_SOURCE, ""))
            idx = int(str(cell.get(self._KEY_INDEX, 0)))
            self._check_cell_overfitting(code, idx, findings)
        has_critical = any(self._CRITICAL_TAG in f for f in findings)
        severity = self._SEVERITY_CRITICAL if has_critical else self._SEVERITY_WARNING
        passed = not has_critical
        return InspectionCheck(
            name=self._CHECK_OVERFITTING,
            passed=passed,
            findings=findings,
            severity=self._SEVERITY_INFO if not findings else severity,
        )

    def _check_cell_overfitting(
        self, code: str, idx: int, findings: list[str]
    ) -> None:
        """Check a single code cell for overfitting patterns."""
        code_lower = code.lower()
        for indicator in self._OVERFITTING_INDICATORS:
            if indicator in code_lower:
                findings.append(f"Overfitting indicator '{indicator}' in cell {idx}")
        if "patch_map" in code or "manual_patches" in code:
            findings.append("CRITICAL: Manual patch dictionary detected")
        epoch_match = re.search(self._EPOCH_PATTERN, code)
        if epoch_match and int(epoch_match.group(1)) <= 1:
            findings.append(self._SUSPICIOUS_EPOCH_MSG)

    def _extract_architecture(self, code_cells: list[dict[str, Any]]) -> str:
        """Identify agent architecture pattern from code cells."""
        full_code = " ".join(str(c.get(self._KEY_SOURCE, "")).lower() for c in code_cells)
        if any(w in full_code for w in ("agenttool", "delegat", "multi_agent", "coordinator")):
            return "multi_agent"
        if any(w in full_code for w in ("react", "tool_call", "agent")):
            return "single_agent_react"
        return "standard_agent"

    def _extract_prompts(self, code_cells: list[dict[str, Any]]) -> list[str]:
        """Extract prompt templates or system prompt definitions from code."""
        prompts: list[str] = []
        pattern = r'(?:system_prompt|prompt)\s*=\s*["\']{1,3}([\s\S]*?)["\']{1,3}'
        for cell in code_cells:
            src = str(cell.get(self._KEY_SOURCE, ""))
            matches = re.findall(pattern, src, flags=re.IGNORECASE)
            for m in matches:
                cleaned = m.strip()
                if cleaned and cleaned not in prompts:
                    prompts.append(cleaned)
        return prompts

    def _extract_training_strategy(self, code_cells: list[dict[str, Any]]) -> str:
        """Identify model training strategy used in notebook."""
        full_code = " ".join(str(c.get(self._KEY_SOURCE, "")).lower() for c in code_cells)
        if "grpo" in full_code:
            return "grpo_rl"
        if "dpo" in full_code:
            return "dpo_rl"
        if "sft" in full_code or "lora" in full_code:
            return "sft_lora"
        if "train" in full_code:
            return "supervised_training"
        return "prompt_only"

    def _identify_novel_techniques(self, code_cells: list[dict[str, Any]]) -> list[str]:
        """Extract innovative techniques identified in the notebook."""
        full_code = " ".join(str(c.get(self._KEY_SOURCE, "")).lower() for c in code_cells)
        techs: list[str] = []
        checks: list[tuple[tuple[str, ...], str]] = [
            (("reflection",), "reflection"),
            (("tree_of_thought", "tot"), "tree_of_thought"),
            (("curriculum",), "curriculum_learning"),
            (("best_of_n", "test_time_compute"), "best_of_n_sampling"),
            (("pruning", "dynamic_context"), "context_pruning"),
            (("subagent", "agent_tool"), "multi_agent_delegation"),
            (("dynamic_budget",), "dynamic_budgeting"),
        ]
        for keywords, label in checks:
            if any(k in full_code for k in keywords) and label not in techs:
                techs.append(label)
        return techs

    def _assess_risk(self, checks: list[InspectionCheck]) -> str:
        """Assess overall risk level based on inspection checks."""
        all_findings = [f for c in checks for f in c.findings]
        if any(self._CRITICAL_TAG in f for f in all_findings):
            return self._RISK_CRITICAL
        if any(self._SUSPICIOUS_TAG in f for f in all_findings):
            return self._RISK_MEDIUM
        if all_findings:
            return self._RISK_LOW
        return self._RISK_CLEAN

    def _check_adopt_criteria(
        self, report: InspectionReport, adversarial: AdversarialResult
    ) -> Decision | None:
        """Check if full adoption criteria are met."""
        if not (
            adversarial.improvement > self._ADOPT_IMPROVEMENT_THRESHOLD
            and adversarial.is_significant
            and adversarial.cv_lb_gap < self._ADOPT_CV_LB_GAP_THRESHOLD
            and report.risk_level in (self._RISK_CLEAN, self._RISK_LOW)
        ):
            return None
        improving_folds = sum(
            1 for v in adversarial.per_fold_comparison.values() if v.get(self._KEY_DELTA, 0.0) > 0.0
        )
        if improving_folds >= self._MIN_IMPROVING_FOLDS:
            reason = (
                f"CV improvement +{adversarial.improvement:.1%}, "
                f"significant, consistent across {improving_folds}/4 folds"
            )
            return Decision(
                action=self._ACTION_ADOPT,
                reason=reason,
                confidence=self._CONFIDENCE_HIGH,
                components_to_adopt=report.novel_techniques,
            )
        return None

    def _check_partial_adopt_criteria(
        self, report: InspectionReport, adversarial: AdversarialResult
    ) -> Decision | None:
        """Check if partial adoption criteria are met."""
        if not (
            adversarial.improvement > self._PARTIAL_IMPROVEMENT_THRESHOLD
            and adversarial.is_significant
            and report.risk_level != self._RISK_CRITICAL
        ):
            return None
        beneficial = self._isolate_beneficial_techniques(
            report.novel_techniques, adversarial
        )
        if beneficial:
            return Decision(
                action=self._ACTION_PARTIAL,
                reason=f"Selective adoption of {len(beneficial)} techniques",
                confidence=self._CONFIDENCE_MEDIUM,
                components_to_adopt=beneficial,
            )
        return None

    def _obtain_peer_cv_result(
        self,
        solution: PeerSolution,
        tasks: list[Task],
        precomputed_cv: CVReport | None,
    ) -> CVReport:
        """Return precomputed CV result or execute CV evaluation."""
        if precomputed_cv is not None:
            return precomputed_cv
        components = self._extract_components(solution)
        merged = self._merge_components(self._baseline_adapter_path, components)
        return self._cv_evaluator.evaluate_all_folds(
            adapter_path=str(merged),
            tasks=tasks,
            version=f"{self._VERSION_PEER_PREFIX}{solution.notebook_name}",
        )

    def _build_per_fold_comparison(
        self, cv_result: CVReport, base_result: CVReport
    ) -> dict[str, dict[str, float]]:
        """Build dictionary comparing peer and baseline resolution per fold."""
        comparison: dict[str, dict[str, float]] = {}
        for peer_f, base_f in zip(cv_result.fold_results, base_result.fold_results):
            repo = peer_f.val_repo
            delta = peer_f.resolution_rate - base_f.resolution_rate
            comparison[repo] = {
                self._KEY_PEER: peer_f.resolution_rate,
                self._KEY_BASELINE: base_f.resolution_rate,
                self._KEY_DELTA: delta,
            }
        return comparison

    def _extract_components(self, solution: PeerSolution) -> dict[str, object]:
        """Extract adoptable components from peer solution."""
        return {
            "adapter_details": solution.adapter_details,
            "prompt_strategy": solution.prompt_strategy,
            "agent_architecture": solution.agent_architecture,
            "key_techniques": list(solution.key_techniques),
        }

    def _merge_components(
        self, baseline: object, components: dict[str, object]
    ) -> object:
        """Merge peer components with baseline configuration."""
        arch = str(components.get("agent_architecture") or "merged")
        return f"{baseline}_{arch}" if baseline else arch

    def _merge_single_technique(self, baseline: object, technique: str) -> str:
        """Create configuration integrating a single isolated technique."""
        return f"{baseline}_{technique}" if baseline else technique

    def _bootstrap_significance_test(
        self,
        treatment: list[TaskResult],
        control: list[TaskResult],
        confidence: float = 0.95,
    ) -> bool:
        """Test if improvement is statistically significant via bootstrap permutation."""
        treatment_resolved = [1.0 if r.resolved else 0.0 for r in treatment]
        control_resolved = [1.0 if r.resolved else 0.0 for r in control]
        if not treatment_resolved or not control_resolved:
            return False
        obs_diff = float(np.mean(treatment_resolved) - np.mean(control_resolved))
        if obs_diff <= 0.0:
            return False
        pooled = np.array(treatment_resolved + control_resolved)
        n_treatment = len(treatment_resolved)
        rng = np.random.RandomState(self._BOOTSTRAP_SEED)
        diffs = self._compute_bootstrap_diffs(
            pooled, n_treatment, rng, self._BOOTSTRAP_SAMPLES
        )
        p_value = float(np.mean(np.array(diffs) >= obs_diff))
        return p_value < (1.0 - confidence)

    def _compute_bootstrap_diffs(
        self,
        pooled: np.ndarray[Any, Any],
        n_treatment: int,
        rng: np.random.RandomState,
        n_samples: int,
    ) -> list[float]:
        """Generate bootstrap permutation differences under null hypothesis."""
        diffs: list[float] = []
        for _ in range(n_samples):
            permuted = rng.permutation(pooled)
            d = float(np.mean(permuted[:n_treatment]) - np.mean(permuted[n_treatment:]))
            diffs.append(d)
        return diffs

    def _isolate_beneficial_techniques(
        self, techniques: list[str], adversarial: AdversarialResult
    ) -> list[str]:
        """Test each technique individually to identify truly beneficial ones."""
        beneficial: list[str] = []
        for technique in techniques:
            if self._tasks and self._quick_eval_task_ids:
                if self._test_single_technique(technique):
                    beneficial.append(technique)
            else:
                beneficial.append(technique)
        return beneficial

    def _test_single_technique(self, technique: str) -> bool:
        """Perform quick evaluation for a single isolated technique."""
        task_ids = self._quick_eval_task_ids or []
        config = self._merge_single_technique(self._baseline_adapter_path, technique)
        q_res = self._cv_evaluator.quick_evaluate(
            adapter_path=config, task_ids=task_ids, tasks=self._tasks
        )
        base_res = self._cv_evaluator.quick_evaluate(
            adapter_path=self._baseline_adapter_path or "",
            task_ids=task_ids,
            tasks=self._tasks,
        )
        delta = q_res.resolution_rate - base_res.resolution_rate
        if delta > self._PARTIAL_IMPROVEMENT_THRESHOLD:
            self._telemetry.log_info(f"Technique '{technique}' provides +{delta:.1%} improvement")
            return True
        self._telemetry.log_info(f"Technique '{technique}' provides only {delta:+.1%} — skipping")
        return False
