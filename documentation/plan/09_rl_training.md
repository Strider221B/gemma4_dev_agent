# Epic 8 — RL Training Pipeline

> **Runs on:** Server (Kaggle 4×L4 GPUs) — **code is written locally but actual training runs on Kaggle**
> **Depends on:** Epic 7 (SFT Training)
> **Estimated effort:** ~3 hours (code writing) + GPU time on Kaggle
> **Goal:** Implement `RLTrainerPipeline` (GRPO + DPO), `RewardModel` (multi-signal reward), `RolloutGenerator`, and `PreferencePairGenerator`.

---

## Pre-Requisites

- Epic 7 is complete
- Activate environment: `source ~/python_envs/p313_llm/bin/activate`
- Training dependencies only on Kaggle — local tests must mock everything

---

## Design Reference

- `detailed/03_rl_training.md` — Full RL pipeline design
- `high_level/04_training_strategy.md` § 3 — RL configuration

---

## Task 8.1: Create `src/training/reward_model.py`

**Class:** `RewardModel`

**Design reference:** `detailed/03_rl_training.md` § 3

**Constructor:**
```python
def __init__(self, telemetry: TelemetryLogger) -> None:
```

**Class-level constants (from design doc):**
- `_PASS_REWARD: float = 1.0`
- `_FAIL_REWARD: float = -0.5`
- `_NO_PATCH_PENALTY: float = -1.0`
- `_PARTIAL_PASS_WEIGHT: float = 0.5`
- `_EFFICIENCY_BONUS_MAX: float = 0.15`
- `_TRUNCATION_PENALTY: float = -0.2`
- `_SCRATCH_IN_WORKSPACE_PENALTY: float = -0.3`
- `_TEST_MODIFICATION_PENALTY: float = -0.5`
- `_LENGTH_PENALTY_THRESHOLD: int = 24000`
- `_REWARD_CLAMP_MIN: float = -1.5`
- `_REWARD_CLAMP_MAX: float = 1.3`
- `_EFFICIENCY_THRESHOLD_RATIO: float = 0.7`

**Public methods:**
- `compute_reward(trajectory: Trajectory, task: Task) -> float` — compute composite reward
- `compute_batch_rewards(trajectories: list[Trajectory], tasks: list[Task]) -> list[float]`

**Private methods:**
- `_compute_outcome_reward(trajectory: Trajectory, task: Task) -> float` — pass/fail from pytest
- `_compute_efficiency_bonus(trajectory: Trajectory, task: Task) -> float` — tool usage efficiency
- `_compute_behaviour_penalties(trajectory: Trajectory) -> float` — scratch/test penalties
- `_compute_length_penalty(trajectory: Trajectory) -> float` — token length regularisation
- `_apply_and_test(patch: str, task: Task) -> object` — apply patch and run tests (mockable)
- `_get_median_tool_calls(complexity: ComplexityTier) -> int` — lookup median by tier
- `_is_test_file(filepath: str) -> bool` — test file pattern check
- `_is_scratch_file(filepath: str) -> bool` — scratch file pattern check

---

## Task 8.2: Create `src/training/rollout_generator.py`

**Class:** `RolloutGenerator`

**Design reference:** `detailed/03_rl_training.md` § 1

**Constructor:**
```python
def __init__(self, telemetry: TelemetryLogger) -> None:
```

**Public methods:**
- `generate_rollouts(model: object, tokenizer: object, task: Task, num_generations: int) -> list[Trajectory]` — generate multiple trajectory rollouts for a task

**Private methods:**
- `_execute_trajectory(model_output: str, task: Task) -> Trajectory` — parse and execute generated trajectory
- `_simulate_tool_execution(tool_call: ToolCall) -> str` — simulate tool result

---

## Task 8.3: Create `src/training/preference_pair_generator.py`

**Class:** `PreferencePairGenerator`

**Design reference:** `detailed/03_rl_training.md` § 4.1

**Constructor:**
```python
def __init__(self, reward_model: RewardModel, telemetry: TelemetryLogger) -> None:
```

**Class-level constants:**
- `_MIN_SCORE_DIFFERENCE: float = 0.3`

**Public methods:**
- `generate_pairs(tasks: list[Task], model: object, tokenizer: object, n_per_task: int = 4) -> list[dict[str, object]]` — generate DPO preference pairs

**Private methods:**
- `_generate_completions(model: object, tokenizer: object, task: Task, n: int) -> list[Trajectory]`
- `_build_prompt(task: Task) -> str`
- `_score_and_rank(completions: list[Trajectory], task: Task) -> list[tuple[Trajectory, float]]`
- `_create_pair(chosen: Trajectory, rejected: Trajectory, task: Task) -> dict[str, object]`

---

## Task 8.4: Create `src/training/rl_early_stopping.py`

**Class:** `RLEarlyStoppingCallback`

**Constructor:**
```python
def __init__(self, min_reward_improvement: float = 0.01, patience: int = 5) -> None:
```

**Public methods:**
- `on_evaluate(args: object, state: object, control: object, metrics: dict[str, object] | None = None, **kwargs: object) -> None` — check reward improvement and set stop flag

---

## Task 8.5: Create `src/training/rl_trainer.py`

**Class:** `RLTrainerPipeline`

**Design reference:** `detailed/03_rl_training.md` § 2

**Constructor:**
```python
def __init__(
    self,
    reward_model: RewardModel,
    checkpoint_mgr: CheckpointManager,
    telemetry: TelemetryLogger,
) -> None:
```

**Class-level constants:**
- `_ADAPTER_SIZE_LIMIT: int = 1_500_000_000`

**Public methods:**
- `run(config: RLConfig, tasks: list[Task]) -> str` — execute RL training, return adapter path

**Private methods:**
- `_load_sft_model(config: RLConfig) -> tuple[object, object]` — load base + SFT adapter merged (lazy import)
- `_merge_and_reapply_lora(model: object, config: RLConfig) -> object` — merge SFT, apply fresh LoRA
- `_build_prompts(tasks: list[Task]) -> object` — build prompts as HF Dataset
- `_train_grpo(model: object, tokenizer: object, prompts: object, config: RLConfig) -> str`
- `_train_dpo(model: object, tokenizer: object, prompts: object, config: RLConfig) -> str`
- `_compute_size(path: str) -> int`

---

## Task 8.6: Write Tests

### `tests/unit/training/test_reward_model.py`
- `test_compute_reward_full_pass_returns_positive` — all tests pass → ~1.0
- `test_compute_reward_no_patch_returns_negative` — no patch → -1.0
- `test_compute_reward_partial_pass_returns_partial_credit`
- `test_efficiency_bonus_few_calls` — below median → positive bonus
- `test_efficiency_bonus_many_calls` — at/above median → 0
- `test_truncation_penalty_applied`
- `test_scratch_in_workspace_penalty`
- `test_test_modification_penalty`
- `test_length_penalty_under_threshold_is_zero`
- `test_length_penalty_over_threshold_is_negative`
- `test_reward_is_clamped`

### `tests/unit/training/test_rl_trainer.py`
- `test_run_grpo_calls_expected_methods` — mock all, verify call sequence
- `test_run_dpo_calls_expected_methods`
- `test_run_invalid_mode_raises_error`
- `test_run_logs_start_and_end`

### `tests/unit/training/test_preference_pair_generator.py`
- `test_generate_pairs_returns_list`
- `test_generate_pairs_chosen_score_higher_than_rejected`
- `test_generate_pairs_filters_small_differences`

### `tests/unit/training/test_rl_early_stopping.py`
- `test_stops_after_patience_exceeded`
- `test_continues_when_improving`

---

## Task 8.7: Run CI

```bash
source ~/python_envs/p313_llm/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `RLTrainerPipeline` supports both GRPO and DPO modes
- [ ] `RewardModel` computes multi-signal reward correctly
- [ ] `PreferencePairGenerator` creates DPO pairs
- [ ] All external libraries lazy-imported and mocked in tests
- [ ] Tests pass with ≥90% coverage
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/training/reward_model.py
src/training/rollout_generator.py
src/training/preference_pair_generator.py
src/training/rl_early_stopping.py
src/training/rl_trainer.py
tests/unit/training/test_reward_model.py
tests/unit/training/test_rl_trainer.py
tests/unit/training/test_preference_pair_generator.py
tests/unit/training/test_rl_early_stopping.py
```
