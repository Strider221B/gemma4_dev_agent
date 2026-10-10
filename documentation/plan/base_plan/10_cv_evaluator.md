# Epic 9 — CV Evaluator

> **Runs on:** Server (Kaggle 4×L4 GPUs) — **code is written locally but actual evaluation runs on Kaggle**
> **Depends on:** Epics 6 (Dataset Builder), 7 (SFT Training)
> **Estimated effort:** ~3 hours (code writing)
> **Goal:** Implement the full CV evaluation pipeline: `CVEvaluator`, `TrajectoryExecutor`, `VerificationRunner`, `MockSandbox`, `MetricTracker`, and `ExecutionBudget`.

---

## Pre-Requisites

- Epics 6 and 7 are complete
- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`

---

## Design Reference

- `detailed/04_cv_evaluator.md` — Full CV pipeline design
- `high_level/05_evaluation_framework.md` — Evaluation architecture

---

## Task 9.1: Create `src/evaluation/execution_budget.py`

**Class:** `ExecutionBudget`

**Design reference:** `detailed/04_cv_evaluator.md` § 4.1

**Constructor:**
```python
def __init__(self, overrides: dict[str, object] | None = None) -> None:
```

**Class-level constants:**
- `_DEFAULT_MAX_TOOL_CALLS: int = 100`
- `_DEFAULT_MAX_TIME_MINUTES: float = 60.0`
- `_DEFAULT_MAX_TURNS: int = 500`
- `_DEFAULT_COMMAND_TIMEOUT: int = 300`
- `_MAX_STDOUT_CHARS: int = 5000`
- `_MAX_FILE_LINES: int = 150`
- `_MAX_FILE_CHARS: int = 10000`
- `_MAX_NUDGES: int = 3`
- `_FREE_TOOLS: frozenset[str] = frozenset({"submit_patch", "get_status"})`
- `_LOW_BUDGET_THRESHOLD: int = 10`
- `_LOW_BUDGET_MIN_CALLS: int = 20`

**Public methods:**
- `consume_tool_call(tool_name: str) -> bool` — register call, return False if over budget
- `check_time() -> bool` — True if time budget available
- `should_warn() -> bool` — True if low-budget warning needed
- `get_status() -> dict[str, object]` — harness-compatible status dict

---

## Task 9.2: Create `src/evaluation/mock_sandbox.py`

**Class:** `MockSandbox`

**Design reference:** `high_level/05_evaluation_framework.md` § 5

**Constructor:**
```python
def __init__(self) -> None:
```

**Public methods:**
- `setup_workspace(task: Task) -> str` — create temp workspace from mock data, return path
- `apply_patch(patch: str) -> bool` — apply unified diff to workspace
- `reset_protected_files(task: Task) -> None` — reset test files to baseline
- `run_command(command: str, timeout: int = 300) -> dict[str, object]` — execute command in workspace
- `cleanup() -> None` — remove temp workspace

**Private methods:**
- `_is_protected_file(filepath: str) -> bool` — check test file patterns
- `_create_workspace_from_snapshot(task: Task) -> str` — extract or create workspace
- `_apply_test_patch(test_patch: str) -> bool`

---

## Task 9.3: Create `src/evaluation/trajectory_executor.py`

**Class:** `TrajectoryExecutor`

**Design reference:** `detailed/04_cv_evaluator.md` § 3, `high_level/05_evaluation_framework.md` § 3

**Constructor:**
```python
def __init__(self, budget: ExecutionBudget, telemetry: TelemetryLogger) -> None:
```

**Public methods:**
- `execute(adapter_path: str, task: Task) -> ExecutionResult` — execute agent trajectory (full or dry-run)
- `execute_batch(adapter_path: str, tasks: list[Task]) -> list[ExecutionResult]`

**Private methods:**
- `_run_agent_loop(model: object, task: Task) -> Trajectory` — turn-by-turn generation + tool execution
- `_enforce_budget(trajectory: Trajectory) -> bool` — check all budgets
- `_handle_nudge(trajectory: Trajectory, reason: str) -> str` — generate nudge message

---

## Task 9.4: Create `src/evaluation/verification_runner.py`

**Class:** `VerificationRunner`

**Design reference:** `detailed/04_cv_evaluator.md` § 2, `high_level/05_evaluation_framework.md` § 4

**Constructor:**
```python
def __init__(self, sandbox: MockSandbox, telemetry: TelemetryLogger) -> None:
```

**Public methods:**
- `verify(patch: str, task: Task) -> VerificationResult` — apply patch + run tests in clean sandbox
- `verify_batch(patches: list[str], tasks: list[Task]) -> list[VerificationResult]`

**Private methods:**
- `_apply_patch(sandbox: MockSandbox, patch: str) -> bool` — apply agent patch
- `_reset_protected_files(sandbox: MockSandbox, task: Task) -> None` — replicate harness anti-tampering
- `_run_pytest(sandbox: MockSandbox, task: Task) -> dict[str, object]` — run pytest, parse results
- `_validate_junit_xml(xml_content: str) -> dict[str, int]` — parse JUnit XML for pass/fail counts

---

## Task 9.5: Create `src/evaluation/metric_tracker.py`

**Class:** `MetricTracker`

**Design reference:** `high_level/05_evaluation_framework.md` § 6, `detailed/04_cv_evaluator.md` § 1

**Constructor:**
```python
def __init__(self, gap_threshold: float = 0.15, trend_window: int = 5) -> None:
```

**Class-level constants:**
- `_DEFAULT_GAP_THRESHOLD: float = 0.15`
- `_DEFAULT_TREND_WINDOW: int = 5`

**Public methods:**
- `record(version: str, cv_score: float, lb_score: float | None = None) -> None`
- `detect_trend() -> OverfittingSignal` — analyse trend for overfitting
- `get_history() -> list[dict[str, object]]`
- `export_report() -> str` — JSON report of full history

**Private methods:**
- `_linear_trend(values: list[float]) -> float` — compute linear regression slope
- `_alert_overfitting(entry: dict[str, object]) -> None` — log alert

---

## Task 9.6: Create `src/evaluation/cv_evaluator.py`

**Class:** `CVEvaluator`

**Design reference:** `detailed/04_cv_evaluator.md` § 2

**Constructor:**
```python
def __init__(
    self,
    splitter: CVSplitter,
    executor: TrajectoryExecutor,
    verifier: VerificationRunner,
    tracker: MetricTracker,
    telemetry: TelemetryLogger,
) -> None:
```

**Class-level constants:**
- `_MIN_RESOLUTION_RATE: float = 0.10`

**Public methods:**
- `evaluate_all_folds(adapter_path: str, tasks: list[Task], version: str) -> CVReport`
- `evaluate_fold(adapter_path: str, fold: CVFold, tasks: list[Task]) -> CVResult`
- `quick_evaluate(adapter_path: str, task_ids: list[str], tasks: list[Task]) -> CVResult`

**Private methods:**
- `_aggregate_results(fold_results: list[CVResult], version: str) -> CVReport`
- `_generate_recommendations(rate: float, per_repo: dict[str, float], per_complexity: dict[str, float], fold_results: list[CVResult]) -> list[str]`
- `_compute_tier_rate(results: list[TaskResult], tier: ComplexityTier) -> float`
- `_mean(values: list[float]) -> float` — safe mean (handle empty list)

---

## Task 9.7: Write Tests

### `tests/unit/evaluation/test_execution_budget.py`
- `test_consume_tool_call_increments_counter`
- `test_consume_free_tool_does_not_increment` — `submit_patch`, `get_status`
- `test_consume_over_budget_returns_false`
- `test_should_warn_when_low_budget`
- `test_get_status_returns_dict`

### `tests/unit/evaluation/test_mock_sandbox.py`
- `test_setup_workspace_creates_directory`
- `test_apply_patch_returns_true_for_valid`
- `test_cleanup_removes_workspace`

### `tests/unit/evaluation/test_verification_runner.py`
- `test_verify_with_valid_patch_returns_resolved` — mock sandbox, mock pytest pass
- `test_verify_with_invalid_patch_returns_not_resolved`
- `test_verify_resets_protected_files`

### `tests/unit/evaluation/test_metric_tracker.py`
- `test_record_stores_entry`
- `test_detect_trend_none_when_stable`
- `test_detect_trend_strong_when_cv_up_lb_down`
- `test_detect_trend_moderate_when_cv_grows_faster`
- `test_export_report_returns_json`

### `tests/unit/evaluation/test_cv_evaluator.py`
- `test_evaluate_fold_returns_cv_result` — mock executor + verifier
- `test_evaluate_all_folds_runs_all_folds`
- `test_aggregate_results_computes_weighted_rate`
- `test_generate_recommendations_low_rate`
- `test_generate_recommendations_high_truncation`
- `test_quick_evaluate_subset`

---

## Task 9.8: Run CI

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `CVEvaluator` orchestrates all-fold and single-fold evaluation
- [ ] `TrajectoryExecutor` simulates agent execution with budget enforcement
- [ ] `VerificationRunner` replicates Phase 2 verification
- [ ] `MockSandbox` creates workspace and runs mock tests
- [ ] `MetricTracker` detects overfitting trends
- [ ] `ExecutionBudget` enforces all limits exactly matching the harness
- [ ] All tests pass with ≥90% coverage
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/evaluation/execution_budget.py
src/evaluation/mock_sandbox.py
src/evaluation/trajectory_executor.py
src/evaluation/verification_runner.py
src/evaluation/metric_tracker.py
src/evaluation/cv_evaluator.py
tests/unit/evaluation/test_execution_budget.py
tests/unit/evaluation/test_mock_sandbox.py
tests/unit/evaluation/test_verification_runner.py
tests/unit/evaluation/test_metric_tracker.py
tests/unit/evaluation/test_cv_evaluator.py
```
