# Implementation Plan: Gemma 4 ClippableLinear PEFT Support

---

## 1. Objective

Resolve the fatal `ValueError: Target module Gemma4ClippableLinear(...) is not supported` encountered when applying LoRA adapters during Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) on Gemma 4 models. The implementation will introduce dynamic target module resolution that targets the underlying `torch.nn.Linear` within `Gemma4ClippableLinear` while strictly preserving Gemma 4 activation clipping invariants and backwards compatibility.

---

## 2. Tasks & Implementation Steps

### Phase 1: Create `LoRATargetModuleResolver`

- **File**: `src/training/lora_target_resolver.py`
- **Class**: `LoRATargetModuleResolver`
- **Implementation Requirements**:
  1. Adhere to **Strict OOP**: Single class in file matching file name (`lora_target_resolver.py` -> `LoRATargetModuleResolver`).
  2. Class-level private constants:
     - `_CLIPPABLE_CLASS_NAME: str = "Gemma4ClippableLinear"`
     - `_LINEAR_ATTR: str = "linear"`
     - `_DOT_LINEAR: str = ".linear"`
  3. Public method:
     - `resolve(self, model: object, target_modules: list[str]) -> list[str]` (Method length <= 20 lines).
  4. Protected/private helper methods:
     - `_is_clippable_wrapper(self, module: object) -> bool`
     - `_needs_linear_suffix(self, model: object, target: str) -> bool`
     - `_adapt_target_name(self, target: str) -> str`
  5. Logic:
     - Inspect `model.named_modules()` (if `model` has `named_modules`).
     - Check if target module instances match `_CLIPPABLE_CLASS_NAME` or have a child `_LINEAR_ATTR` of linear type.
     - Map wrapped targets to `<target>.linear` (e.g. `q_proj` -> `q_proj.linear`).
     - If module is standard `nn.Linear` or model has no wrapped layers, retain original target name.
     - Deduplicate while preserving order.

### Phase 2: Update `SFTTrainerPipeline`

- **File**: `src/training/sft_trainer.py`
- **Method**: `_apply_lora(self, model: object, config: LoRAConfig) -> object`
- **Implementation Requirements**:
  1. Import `LoRATargetModuleResolver` inside `_apply_lora` or at top-level.
  2. Resolve target modules dynamically:
     ```python
     resolver = LoRATargetModuleResolver()
     resolved_targets = resolver.resolve(model, config.target_modules)
     ```
  3. Pass `target_modules=resolved_targets` to `PeftLoraConfig`.
  4. Keep method body <= 20 lines.

### Phase 3: Update `RLTrainerPipeline`

- **File**: `src/training/rl_trainer.py`
- **Method**: `_merge_and_reapply_lora(self, model: object, config: RLConfig) -> object`
- **Implementation Requirements**:
  1. Import `LoRATargetModuleResolver` inside `_merge_and_reapply_lora` or at top-level.
  2. Resolve target modules dynamically:
     ```python
     resolver = LoRATargetModuleResolver()
     resolved_targets = resolver.resolve(model, list(self._TARGET_MODULES))
     ```
  3. Pass `target_modules=resolved_targets` to `PeftLoraConfig`.
  4. Keep method body <= 20 lines.

### Phase 4: Create Unit Tests for `LoRATargetModuleResolver`

- **File**: `tests/unit/training/test_lora_target_resolver.py`
- **Class**: `TestLoRATargetModuleResolver`
- **Test Scenarios**:
  - `test_resolve_with_clippable_linear_wrappers`: Mock model containing `Gemma4ClippableLinear` instances with child `.linear` modules; assert targets are mapped from `["q_proj", "v_proj"]` to `["q_proj.linear", "v_proj.linear"]`.
  - `test_resolve_with_standard_linear_modules`: Mock model containing direct `nn.Linear` modules; assert targets remain `["q_proj", "v_proj"]`.
  - `test_resolve_with_already_suffixed_targets`: Target list already containing `["q_proj.linear"]`; assert targets are not duplicated into `q_proj.linear.linear`.
  - `test_resolve_with_opaque_or_mock_model`: Model lacking `named_modules`; assert fallback returns copy of original targets without error.
  - `test_resolve_with_empty_target_list`: Empty target list returns empty list.
  - `test_resolve_deduplicates_targets`: Multiple matching occurrences across layers do not duplicate target names.

### Phase 5: Update Existing Trainer Unit Tests

- **Files**:
  - `tests/unit/training/test_sft_trainer.py`
  - `tests/unit/training/test_rl_trainer.py`
- **Requirements**:
  1. Ensure mock pipelines pass `LoRATargetModuleResolver` resolution cleanly.
  2. Add test asserting `_apply_lora` invokes `LoRATargetModuleResolver.resolve` and passes resolved targets to `PeftLoraConfig`.
  3. Add test asserting `_merge_and_reapply_lora` invokes `LoRATargetModuleResolver.resolve`.

### Phase 6: Verification and CI Pipeline Validation

- **Environment**: `source ~/python_envs/p312_kaggle/bin/activate`
- **Checks**:
  1. `ruff check ./src ./tests` (Zero errors, line length 100).
  2. `mypy ./src` (Strict mode, zero errors).
  3. `pytest --cov=src --cov-report=term-missing` (All tests pass, coverage >= 90%).

---

## 3. Verification Checklist (Adspire Coding Standards)

| Check | Requirement |
|---|---|
| Single class in file | `LoRATargetModuleResolver` in `lora_target_resolver.py` |
| File name matches class name | `lora_target_resolver.py` <-> `LoRATargetModuleResolver` |
| Member ordering | Class constants -> `__init__` -> public methods -> protected methods -> private methods |
| Method length limit | Every method body <= 20 lines (excluding signature, docstrings, blanks) |
| File length limit | File <= 500 lines total |
| Type hints | Strict typing on all arguments and return values |
| Docstrings | Google/NumPy style docstring on class and all public/protected methods |
| No magic strings/numbers | All string constants (`_CLIPPABLE_CLASS_NAME`, `_LINEAR_ATTR`, `_DOT_LINEAR`) defined as class-level constants |
| No free-standing functions | All logic encapsulated in classes |
| Test coverage | >= 90% branch and line coverage |
| Local CI passes | `ruff` + `mypy` + `pytest` pass cleanly |
