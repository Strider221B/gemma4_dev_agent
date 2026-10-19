# Implementation Plan: Local Model Support (`gemma-4-e2b-it-qat-w4a16-ct`)

---

## 1. Objective

Establish a lightweight local model test setup using Google's `gemma-4-e2b-it-qat-w4a16-ct` (~2B parameter QAT W4A16 Mixture of Experts model) to validate the training pipeline, library compatibility, quantization lifecycle hooks, and PEFT LoRA adapter wiring locally on a developer workstation. This local test setup eliminates remote trial-and-error debugging on Kaggle GPU clusters and specifically prevents recurring runtime defects such as GitHub Issue #37 (`AttributeError: 'Linear' object has no attribute 'weight'`).

> **CRITICAL ARCHITECTURAL INVARIANT**:
> `gemma-4-e2b-it-qat-w4a16-ct` is a **Mixture of Experts (MoE)** model, whereas the production competition model `gemma-4-31b-it-qat-w4a16-ct` is a **dense** transformer.
> The local model test setup is used **exclusively for verifying syntax, library compatibility, lifecycle hooks, and basic pipeline mechanics**. We do NOT tailor production layer targeting or training configurations to the MoE details of E2B. Local test targeting is restricted to shared attention projection layers (e.g. `q_proj`, `v_proj`).

---

## 2. Implementation Tasks & Phases

### Phase 1: Git Hygiene, Constants & ModelPathResolver

- **Target Files**:
  - `.gitignore`
  - `src/utils/constants.py`
  - `src/utils/model_path_resolver.py`
  - `tests/unit/utils/test_model_path_resolver.py`

- **Step 1.1**: Git Hygiene — Exclude All Model Weights & Checkpoints
  - Update `.gitignore` to explicitly prevent any local models, safetensors, or test artifacts from being tracked or checked into GitHub:
    ```gitignore
    # Large model weights and checkpoints (never commit to git)
    models/
    kaggle_working/
    *.safetensors
    *.bin
    *.pt
    *.pth
    *.onnx
    ```

- **Step 1.2**: Update `src/utils/constants.py`
  - Add class/module constants for the local test model:
    ```python
    LOCAL_TEST_MODEL_NAME: str = "google/gemma-4-e2b-it-qat-w4a16-ct"
    LOCAL_TEST_MODEL_SLUG: str = "gemma-4-e2b-it-qat-w4a16-ct"
    ```

- **Step 1.3**: Update `src/utils/model_path_resolver.py` (`ModelPathResolver`)
  - Add class constants:
    - `_LOCAL_TEST_MODEL_SLUG: ClassVar[str] = "gemma-4-e2b-it-qat-w4a16-ct"`
    - `_ENV_LOCAL_MODEL_PATH: ClassVar[str] = "LOCAL_MODEL_PATH"`
    - `_ENV_GEMMA_E2B_PATH: ClassVar[str] = "GEMMA_E2B_MODEL_PATH"`
  - Add local candidate search paths to `_CANDIDATE_PATHS` or dedicated local resolution method:
    - `models/gemma-4-e2b-it-qat-w4a16-ct`
    - `~/.cache/huggingface/hub/models--google--gemma-4-e2b-it-qat-w4a16-ct`
  - Add helper method:
    - `has_local_model(cls, model_name_or_path: str) -> bool`: Checks if a resolved path exists locally as a directory (used by integration tests for conditional execution).
  - Update `_check_env_paths()` to check `LOCAL_MODEL_PATH` and `GEMMA_E2B_MODEL_PATH`.
  - Update `_is_competition_model()` or add `_is_supported_model()` to recognize both 31B and E2B model slugs and identifiers.
  - Ensure all methods remain ≤ 20 logical lines.

- **Step 1.4**: Unit Tests for Model Path Resolution (Zero Model Weights)
  - Update `tests/unit/utils/test_model_path_resolver.py` to verify:
    - Resolving `google/gemma-4-e2b-it-qat-w4a16-ct` via mocked filesystem directory paths.
    - Resolving `gemma-4-e2b-it-qat-w4a16-ct` via `LOCAL_MODEL_PATH` and `GEMMA_E2B_MODEL_PATH` environment variables.
    - Fallback behavior when local path does not exist.
  - **Invariant**: Uses `unittest.mock.patch("os.path.exists")` — zero physical weights required.

---

### Phase 2: Create Extracted Abstraction & `LocalModelSmokeTester`

- **Target Files**:
  - `src/training/model_verification_runner.py` (`ModelVerificationRunner` Protocol)
  - `src/training/local_model_smoke_tester.py` (`LocalModelSmokeTester`)

- **Step 2.1**: Define Extracted Abstraction Protocol
  - Create `src/training/model_verification_runner.py`:
    ```python
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class ModelVerificationRunner(Protocol):
        """Protocol defining the contract for model verification smoke testers."""

        def run_verification(self) -> dict[str, object]:
            """Execute verification workflow and return status summary."""
            ...
    ```

- **Step 2.2**: Implement `LocalModelSmokeTester`
  - **Class**: `LocalModelSmokeTester(ModelVerificationRunner)`
  - **Implementation Requirements**:
    1. **Strict OOP**: Single class in file matching file name (`local_model_smoke_tester.py` -> `LocalModelSmokeTester`).
    2. **Class-level Private Constants**:
       - `_DEFAULT_MODEL_NAME: ClassVar[str] = "google/gemma-4-e2b-it-qat-w4a16-ct"`
       - `_TARGET_MODULES: ClassVar[tuple[str, ...]] = ("q_proj", "v_proj")`
       - `_DTYPE_BFLOAT16: ClassVar[str] = "bfloat16"`
       - `_DTYPE_FLOAT32: ClassVar[str] = "float32"`
       - `_DEVICE_AUTO: ClassVar[str] = "auto"`
       - `_DEVICE_CPU: ClassVar[str] = "cpu"`
       - `_DEFAULT_LORA_R: ClassVar[int] = 8`
       - `_DEFAULT_LORA_ALPHA: ClassVar[int] = 16`
       - `_DUMMY_SEQ_LEN: ClassVar[int] = 16`
       - `_SUBDIR_TEST_ADAPTER: ClassVar[str] = "test_adapter"`
       - `_KEY_STATUS: ClassVar[str] = "status"`
       - `_KEY_STATUS_SUCCESS: ClassVar[str] = "success"`
       - `_KEY_MODEL_NAME: ClassVar[str] = "model_name"`
       - `_KEY_RESOLVED_PATH: ClassVar[str] = "resolved_path"`
       - `_KEY_FORWARD_PASSED: ClassVar[str] = "forward_passed"`
       - `_KEY_BACKWARD_PASSED: ClassVar[str] = "backward_passed"`
       - `_KEY_ADAPTER_SAVED: ClassVar[str] = "adapter_saved"`
       - `_KEY_ELAPSED_SECONDS: ClassVar[str] = "elapsed_seconds"`
    3. **Constructor Injection**:
       - `__init__(self, model_name: str | None = None, work_dir: str | None = None) -> None`
    4. **Public Methods** (all ≤ 20 lines):
       - `run_verification(self) -> dict[str, object]`: Orchestrates the 7-stage verification workflow and returns a structured summary report.
    5. **Protected/Private Helper Methods** (all ≤ 20 lines):
       - `_resolve_model_path(self) -> str`
       - `_load_model_and_tokenizer(self, resolved_path: str) -> tuple[object, object]`
       - `_build_quantization_config(self, resolved_path: str) -> object | None`
       - `_verify_tokenizer_tokens(self, tokenizer: object) -> None`
       - `_attach_lora_adapters(self, model: object) -> object`
       - `_execute_forward_pass(self, model: object, tokenizer: object) -> object`
       - `_execute_backward_pass(self, outputs: object) -> bool`
       - `_save_adapter_checkpoint(self, model: object) -> str`
       - `_build_summary_report(self, resolved_path: str, adapter_path: str, elapsed: float) -> dict[str, object]`

---

### Phase 3: Verify PEFT & Compressed-Tensors Patching (Issue #37 Guardrail)

- **Target Files**:
  - `src/training/peft_model_patcher.py`
  - `src/training/lora_target_resolver.py`
  - `tests/unit/training/test_peft_model_patcher.py`

- **Step 3.1**: Validate `PEFTModelPatcher` attribute and hook handling
  - Confirm that `peft.tuners.lora.layer.Linear` properly delegates `.weight` and `.bias` to `base_layer`.
  - Validate that when `quantized_forward` executes on an inner or outer linear module, `self.weight` is accessible and does not raise `AttributeError`.
  - Verify that `PEFTModelPatcher.patch_model()` aligns `_old_forward` hooks properly to avoid bypassing LoRA forward logic or executing `quantized_forward` on uninitialized wrapper scopes.

- **Step 3.2**: Unit Tests for Issue #37 Reproduction & Prevention
  - Create a focused test in `tests/unit/training/test_peft_model_patcher.py` simulating a module wrapped by PEFT with `quantized_forward` in its call hierarchy, asserting that accessing `.weight` succeeds and forward pass completes without `AttributeError`.
  - Zero model weights required; uses mock `nn.Module` objects replicating the `quantized_forward` / `Linear` wrapper hierarchy.

---

### Phase 4: Integrate with `LocalPipelineRunner` via Abstraction

- **Target File**: `src/training/local_pipeline_runner.py`
- **Class**: `LocalPipelineRunner`

- **Implementation Requirements**:
  1. Add mode constant:
     - `_MODE_LOCAL_MODEL: ClassVar[str] = "local_model"`
  2. Accept optional `ModelVerificationRunner` in `__init__` for dependency injection:
     - `__init__(self, env_detector: EnvironmentDetector | None = None, work_dir: str | None = None, verification_runner: ModelVerificationRunner | None = None) -> None`
  3. Update CLI argument parser in `main()` to include `local_model`:
     - `choices=[cls._MODE_SMOKE, cls._MODE_MOCK, cls._MODE_FULL, cls._MODE_LOCAL_MODEL]`
  4. Update `run(self, mode: str) -> dict[str, object]`:
     - Add branch: `if mode == self._MODE_LOCAL_MODEL: return self.run_local_model()`
  5. Implement `run_local_model(self) -> dict[str, object]` (≤ 20 lines):
     - Delegates execution to injected or default `ModelVerificationRunner`.
  6. Ensure method length limit (≤ 20 lines) and file length limit (≤ 500 lines) are strictly maintained.

---

### Phase 5: Testing Strategy & Test Suites (Zero Weights for Unit Tests)

- **Unit Tests (`tests/unit/`) — MANDATORY ZERO MODEL WEIGHTS**:
  - `tests/unit/training/test_local_model_smoke_tester.py`:
    - Mocked test suite verifying:
      - Implementation of `ModelVerificationRunner` protocol (`isinstance(tester, ModelVerificationRunner)`).
      - Instantiation with default and custom arguments.
      - Path resolution delegation via `ModelPathResolver`.
      - Tokenizer special tokens verification (passes when present, raises `ValueError` when absent).
      - LoRA attachment with `LoRATargetModuleResolver` and `PEFTModelPatcher`.
      - Forward pass execution and tensor shape assertions.
      - Backward pass gradient verification.
      - Adapter saving and summary generation.
  - `tests/unit/training/test_local_pipeline_runner.py`:
    - Tests dispatching `--mode local_model` to `ModelVerificationRunner` using a mock runner test double.
- **Integration Tests (`tests/integration/`)**:
  - `tests/integration/test_local_model_smoke.py`:
    - Real execution test that runs against `gemma-4-e2b-it-qat-w4a16-ct` **only** if weights exist locally.
    - Decorated with:
      ```python
      @pytest.mark.skipif(
          not ModelPathResolver.has_local_model("google/gemma-4-e2b-it-qat-w4a16-ct"),
          reason="Local Gemma 4 E2B weights not present on disk",
      )
      ```
    - Guarantees GitHub Actions / offline CI runs pass cleanly without attempting model downloads.

---

### Phase 6: Local CI Verification

Run mandatory CI pipeline in `source ~/python_envs/p312_kaggle/bin/activate`:
1. **Stage 1 (Lint)**:
   ```bash
   ruff check ./src ./tests
   ```
   *Requirement*: 0 errors.
2. **Stage 2 (Type Check)**:
   ```bash
   mypy ./src
   ```
   *Requirement*: 0 errors in strict mode.
3. **Stage 3 (Tests & Coverage)**:
   ```bash
   pytest -o addopts=""
   ```
   *Requirement*: 100% test pass rate with coverage ≥ 90% (executed with ZERO physical model weights).

---

## 3. Verification Checklist

| Rule | Requirement | Validation Method |
|---|---|---|
| **Git Hygiene** | Model weights / checkpoints in `.gitignore`, never committed to GitHub | Git status / gitignore audit |
| **Zero Weight Dependency in Unit Tests** | Unit tests never require or download binary model weights | Pytest in clean offline environment |
| **Extracted Abstraction** | `ModelVerificationRunner` protocol decoupling runner from concrete tester | Static audit / Protocol check |
| **SOLID Compliance** | Single Responsibility, Dependency Injection | Code review |
| **MoE Scope Invariant** | Local model used for syntax only; no MoE specialization in production | Code review |
| **Strict OOP** | All code in classes, single class per file | Static audit |
| **Member Ordering** | Constants -> `__init__` -> Public -> Protected -> Private | Static audit |
| **Method Length** | Body strictly ≤ 20 logical lines (hard limit) | Line count audit |
| **File Length** | Total lines strictly ≤ 500 lines | `wc -l` audit |
| **No Magic Values** | Named constants with `_` prefix | Code review |
| **Issue #37 Prevention** | No `AttributeError: 'Linear' object has no attribute 'weight'` | Automated unit test with mock wrappers |
| **Local CI Verification** | Ruff + Mypy + Pytest all pass cleanly | Command execution |
