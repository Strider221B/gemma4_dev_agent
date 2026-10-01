# Epic 10 — Peer Solution Protocol

> **Runs on:** Server (Kaggle) — **Step 3 (Adversarial Validation) runs on server due to GPU requirements; code is written locally**
> **Depends on:** Epic 9 (CV Evaluator)
> **Estimated effort:** ~2.5 hours (code writing)
> **Goal:** Implement `PeerAnalyser` (4-step adversarial validation: capture → inspect → adversarial CV → decision) and supporting classes.

---

## Pre-Requisites

- Epic 9 is complete (CV Evaluator available for adversarial runs)
- Activate environment: `source ~/python_envs/p313_llm/bin/activate`

---

## Design Reference

- `detailed/05_peer_solution_protocol.md` — Full protocol design

---

## IMPORTANT: Server Execution Note

The user's machine is low-spec. **Step 3 (Adversarial Validation)** involves running full CV evaluations which require GPU inference. The code is written locally, but actual adversarial CV runs must execute on the Kaggle server. Design the code to:
1. Accept a pre-computed CV result (for offline mode) or run CV directly (server mode)
2. Make the `adversarial_cv_test` method accept optional pre-computed results
3. Keep all inspection/decision logic fully local-runnable

---

## Task 10.1: Create `src/evaluation/notebook_parser.py`

**Class:** `NotebookParser`

**Purpose:** Parse Jupyter notebooks to extract code cells for inspection.

**Constructor:** `__init__(self) -> None`

**Public methods:**
- `load_notebook(path: str) -> dict[str, object]` — load `.ipynb` as structured dict
- `extract_code_cells(notebook: dict[str, object]) -> list[dict[str, object]]` — return list of `{index, source}` dicts
- `extract_markdown_cells(notebook: dict[str, object]) -> list[dict[str, object]]`

**Private methods:**
- `_read_file(path: str) -> str` — read file contents

---

## Task 10.2: Create `src/evaluation/peer_analyser.py`

**Class:** `PeerAnalyser`

**Design reference:** `detailed/05_peer_solution_protocol.md` § 4, 5, 6, 7

**Constructor:**
```python
def __init__(
    self,
    cv_evaluator: CVEvaluator,
    splitter: CVSplitter,
    notebook_parser: NotebookParser,
    telemetry: TelemetryLogger,
    baseline_adapter_path: str | None = None,
    quick_eval_task_ids: list[str] | None = None,
) -> None:
```

**Class-level constants:**
- `_LEAKAGE_KEYWORDS: list[str] = ["test_patch", "instance_id", "solution", "answer", "ground_truth", "gold_patch", "reference_patch"]`
- `_OVERFITTING_INDICATORS: list[str] = ["specific task", "hardcoded", "manual fix", "if instance_id ==", "special case"]`
- `_ADOPT_IMPROVEMENT_THRESHOLD: float = 0.05`
- `_PARTIAL_IMPROVEMENT_THRESHOLD: float = 0.02`
- `_MAX_CV_LB_GAP: float = 0.20`
- `_ADOPT_CV_LB_GAP_THRESHOLD: float = 0.10`
- `_MIN_IMPROVING_FOLDS: int = 3`
- `_SUSPICIOUS_INSTANCE_ID_COUNT: int = 5`
- `_BOOTSTRAP_SAMPLES: int = 10000`
- `_BOOTSTRAP_SEED: int = 42`
- `_DEFAULT_CONFIDENCE: float = 0.95`

**Public methods:**
- `inspect(solution: PeerSolution) -> InspectionReport` — Step 2: full code inspection
- `adversarial_cv_test(solution: PeerSolution, tasks: list[Task], precomputed_cv: CVReport | None = None) -> AdversarialResult` — Step 3: CV stress test (runs on server or uses precomputed)
- `make_decision(report: InspectionReport, adversarial: AdversarialResult) -> Decision` — Step 4: adoption decision

**Private methods for Step 2 (Inspection):**
- `_scan_for_leakage(code_cells: list[dict[str, object]]) -> InspectionCheck` — check for leakage keywords
- `_scan_for_overfitting(code_cells: list[dict[str, object]]) -> InspectionCheck` — check for overfitting patterns
- `_extract_architecture(code_cells: list[dict[str, object]]) -> str` — identify agent architecture
- `_extract_prompts(code_cells: list[dict[str, object]]) -> list[str]` — extract system prompts
- `_extract_training_strategy(code_cells: list[dict[str, object]]) -> str` — identify training approach
- `_identify_novel_techniques(code_cells: list[dict[str, object]]) -> list[str]` — list novel techniques
- `_assess_risk(checks: list[InspectionCheck]) -> str` — "clean" / "low" / "medium" / "critical"

**Private methods for Step 3 (Adversarial CV):**
- `_extract_components(solution: PeerSolution) -> dict[str, object]` — extract adoptable components
- `_merge_components(baseline: object, components: dict[str, object]) -> object` — create modified config
- `_bootstrap_significance_test(treatment: list[TaskResult], control: list[TaskResult], confidence: float) -> bool` — bootstrap permutation test

**Private methods for Step 4 (Decision):**
- `_isolate_beneficial_techniques(techniques: list[str], adversarial: AdversarialResult) -> list[str]` — ablation test each technique

---

## Task 10.3: Create `src/evaluation/peer_registry_loader.py`

**Class:** `PeerRegistryLoader`

**Purpose:** Load/save the peer solution registry YAML file.

**Constructor:**
```python
def __init__(self, registry_path: str) -> None:
```

**Public methods:**
- `load_registry() -> list[PeerSolution]` — parse `peer_registry.yaml` into PeerSolution objects
- `save_registry(solutions: list[PeerSolution]) -> None` — write back to YAML
- `add_solution(solution: PeerSolution) -> None` — append a new entry
- `update_status(notebook_name: str, status: str, cv_score: float | None = None, decision: str | None = None) -> None`

**Private methods:**
- `_parse_solution(entry: dict[str, object]) -> PeerSolution`
- `_serialize_solution(solution: PeerSolution) -> dict[str, object]`

---

## Task 10.4: Create `configs/peer_registry.yaml`

Create the initial registry file at `/home/somesh/git_repos/gemma4_dev_agent/configs/peer_registry.yaml` with the template from `detailed/05_peer_solution_protocol.md` § 8.

---

## Task 10.5: Write Tests

### `tests/unit/evaluation/test_notebook_parser.py`
- `test_load_notebook_reads_file` — create a minimal `.ipynb` JSON, load it
- `test_extract_code_cells_returns_list`
- `test_extract_code_cells_correct_count`
- `test_extract_markdown_cells_returns_list`

### `tests/unit/evaluation/test_peer_analyser.py`

**Step 2 tests (all run locally, no GPU needed):**
- `test_scan_for_leakage_clean_code` — no leakage keywords → passed
- `test_scan_for_leakage_detects_test_patch` — code contains "test_patch" in inference context → findings
- `test_scan_for_leakage_detects_hardcoded_ids` — >5 instance IDs → suspicious
- `test_scan_for_overfitting_clean` — no indicators → passed
- `test_scan_for_overfitting_detects_manual_patches` — "patch_map" found → critical
- `test_scan_for_overfitting_detects_low_epochs` — num_epochs=1 with high LB → suspicious
- `test_inspect_returns_report` — full inspection returns InspectionReport
- `test_assess_risk_critical` — critical finding → "critical"
- `test_assess_risk_clean` — no findings → "clean"

**Step 3 tests (mock CV evaluator for local execution):**
- `test_adversarial_cv_with_precomputed_results` — pass precomputed CVReport, verify comparison
- `test_bootstrap_significance_test_significant` — treatment clearly better → True
- `test_bootstrap_significance_test_not_significant` — no difference → False

**Step 4 tests:**
- `test_make_decision_reject_critical_risk` — critical risk → REJECT
- `test_make_decision_reject_high_gap` — CV-LB gap > 20% → REJECT
- `test_make_decision_adopt_large_improvement` — >5%, significant, low gap → ADOPT
- `test_make_decision_partial_adopt` — 2-5%, significant → PARTIAL_ADOPT
- `test_make_decision_reject_insufficient_improvement`

### `tests/unit/evaluation/test_peer_registry_loader.py`
- `test_load_registry_parses_solutions`
- `test_save_registry_writes_yaml`
- `test_add_solution_appends`
- `test_update_status_modifies_entry`

---

## Task 10.6: Run CI

```bash
source ~/python_envs/p313_llm/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `PeerAnalyser.inspect()` detects leakage and overfitting patterns
- [ ] `PeerAnalyser.adversarial_cv_test()` works with both live CV and precomputed results
- [ ] `PeerAnalyser.make_decision()` applies the decision matrix correctly
- [ ] `_bootstrap_significance_test` computes permutation-based p-values
- [ ] `PeerRegistryLoader` loads/saves the registry YAML
- [ ] Step 2 and Step 4 run fully locally (no GPU)
- [ ] Step 3 supports precomputed mode for local testing
- [ ] All tests pass with ≥90% coverage
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/evaluation/notebook_parser.py
src/evaluation/peer_analyser.py
src/evaluation/peer_registry_loader.py
configs/peer_registry.yaml
tests/unit/evaluation/test_notebook_parser.py
tests/unit/evaluation/test_peer_analyser.py
tests/unit/evaluation/test_peer_registry_loader.py
```
