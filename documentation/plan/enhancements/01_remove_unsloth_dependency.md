# Enhancement Plan 01 — Remove Unsloth Dependency

> **Runs on:** Local (code changes) — **tested on Kaggle after deployment**
> **Depends on:** Epics 7 (SFT Training) and 8 (RL Training) already complete
> **Estimated effort:** ~1.5 hours
> **Goal:** Replace all `unsloth` usage with standard HuggingFace `transformers` + `peft` + `bitsandbytes`, enabling the training pipeline to run on Kaggle without internet access.

---

## Pre-Requisites

- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`
- Read design document: `documentation/design/enhancements/01_remove_unsloth_dependency.md`

---

## Design Reference

- `design/enhancements/01_remove_unsloth_dependency.md` — Full replacement design with before/after code

---

## Task 1.1: Update `pyproject.toml`

**File:** `pyproject.toml`

**Changes:**

1. In `[project.optional-dependencies] training`: remove `"unsloth"`, add `"bitsandbytes>=0.43"`
2. In `[[tool.mypy.overrides]] module`: remove `"unsloth.*"`, add `"bitsandbytes.*"`

**Verification:**
```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent
pip install -e ".[dev]"
```

---

## Task 1.2: Update `src/training/sft_trainer.py`

**Class:** `SFTTrainerPipeline`

**Design reference:** `design/enhancements/01_remove_unsloth_dependency.md` §3

### Change 1: Update class docstring (L19)

```python
# Before:
"""Orchestrates QLoRA supervised fine-tuning using Unsloth and TRL."""

# After:
"""Orchestrates QLoRA supervised fine-tuning using HuggingFace transformers and TRL."""
```

### Change 2: Replace `_load_model()` method (L127–139)

Replace the method body to use `AutoModelForCausalLM`, `AutoTokenizer`, and `BitsAndBytesConfig`
instead of `FastLanguageModel.from_pretrained()`.

New implementation:
```python
def _load_model(self, config: ModelConfig) -> tuple[object, object]:
    """Load base language model and tokenizer using HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=self._resolve_dtype(config.dtype),
        bnb_4bit_quant_type=self._BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        quantization_config=quantization_config,
        device_map=self._DEVICE_MAP,
        trust_remote_code=True,
        torch_dtype=self._resolve_dtype(config.dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True)
    return model, tokenizer
```

### Change 3: Replace `_apply_lora()` method (L68–87)

Replace the method body to use `peft.get_peft_model()` + `peft.LoraConfig` instead of
`FastLanguageModel.get_peft_model()`.

New implementation:
```python
def _apply_lora(self, model: object, config: LoRAConfig) -> object:
    """Apply QLoRA adapters targeting specified linear projection layers."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    lora_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )
    self._enable_input_grads(model)
    lora_model = get_peft_model(model, lora_config)
    self._enable_gradient_checkpointing(lora_model)
    self._log_trainable_params(lora_model)
    return lora_model
```

### Change 4: Add new private helper methods

Add three new methods to support the refactored `_load_model()` and `_apply_lora()`:

```python
def _resolve_dtype(self, dtype_str: str) -> object:
    """Convert string dtype identifier to torch dtype."""
    import torch
    dtype_map: dict[str, object] = {
        self._DTYPE_BFLOAT16: torch.bfloat16,
        self._DTYPE_FLOAT16: torch.float16,
        self._DTYPE_FLOAT32: torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def _enable_input_grads(self, model: object) -> None:
    """Enable input gradients required for LoRA training with quantised models."""
    if hasattr(model, "enable_input_require_grads"):
        getattr(model, "enable_input_require_grads")()

def _enable_gradient_checkpointing(self, model: object) -> None:
    """Enable gradient checkpointing with non-reentrant mode for memory efficiency."""
    if hasattr(model, "gradient_checkpointing_enable"):
        getattr(model, "gradient_checkpointing_enable")(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

def _log_trainable_params(self, model: object) -> None:
    """Log trainable vs total parameter counts from PEFT model."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = getattr(model, "get_nb_trainable_parameters")()
        self._telemetry.log_info(f"Trainable: {trainable:,} / {total:,}")
```

### Change 5: Add new class-level constants

```python
_BNB_QUANT_TYPE: str = "nf4"
_DEVICE_MAP: str = "auto"
_DTYPE_BFLOAT16: str = "bfloat16"
_DTYPE_FLOAT16: str = "float16"
_DTYPE_FLOAT32: str = "float32"
```

### Method Length Verification

| Method | Logical Lines | Status |
|--------|--------------|--------|
| `_load_model()` | 14 | ✅ ≤ 20 |
| `_apply_lora()` | 12 | ✅ ≤ 20 |
| `_resolve_dtype()` | 7 | ✅ ≤ 20 |
| `_enable_input_grads()` | 2 | ✅ ≤ 20 |
| `_enable_gradient_checkpointing()` | 3 | ✅ ≤ 20 |
| `_log_trainable_params()` | 3 | ✅ ≤ 20 |

---

## Task 1.3: Update `src/training/rl_trainer.py`

**Class:** `RLTrainerPipeline`

**Design reference:** `design/enhancements/01_remove_unsloth_dependency.md` §4

### Change 1: Replace `_load_sft_model()` method (L97–113)

Replace `FastLanguageModel.from_pretrained()` with `AutoModelForCausalLM.from_pretrained()` +
`AutoTokenizer.from_pretrained()` + `BitsAndBytesConfig`.

New implementation:
```python
def _load_sft_model(self, config: RLConfig) -> tuple[object, object]:
    """Load base pretrained language model and merge SFT adapter weights."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_compute_dtype=self._resolve_dtype(config.model.dtype),
        bnb_4bit_quant_type=self._BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=quantization_config,
        device_map=self._DEVICE_MAP,
        torch_dtype=self._resolve_dtype(config.model.dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = PeftModel.from_pretrained(model, str(config.adapter_path))
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    self._telemetry.log_info("SFT adapter merged into base model")
    return model, tokenizer
```

### Change 2: Replace `_merge_and_reapply_lora()` method (L115–128)

Replace `FastLanguageModel.get_peft_model()` with `peft.get_peft_model()` + `LoraConfig`.

New implementation:
```python
def _merge_and_reapply_lora(self, model: object, config: RLConfig) -> object:
    """Attach fresh LoRA adapter layers on top of merged SFT base model."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    lora_r = getattr(config.grpo, "lora_r", self._DEFAULT_LORA_R)
    lora_config = PeftLoraConfig(
        r=lora_r,
        lora_alpha=self._DEFAULT_LORA_ALPHA,
        lora_dropout=self._DEFAULT_LORA_DROPOUT,
        target_modules=list(self._TARGET_MODULES),
        task_type=TaskType.CAUSAL_LM,
    )
    self._enable_input_grads(model)
    lora_model = get_peft_model(model, lora_config)
    self._enable_gradient_checkpointing(lora_model)
    return lora_model
```

### Change 3: Add new private helper methods

Add the same helper methods as SFTTrainerPipeline (DRY note: these are kept in each class because
the coding standards require strict OOP with one class per file — no shared base class is warranted
by YAGNI since only two classes use them):

```python
def _resolve_dtype(self, dtype_str: str) -> object:
    """Convert string dtype identifier to torch dtype."""
    import torch
    dtype_map: dict[str, object] = {
        self._DTYPE_BFLOAT16: torch.bfloat16,
        self._DTYPE_FLOAT16: torch.float16,
        self._DTYPE_FLOAT32: torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def _enable_input_grads(self, model: object) -> None:
    """Enable input gradients required for LoRA training with quantised models."""
    if hasattr(model, "enable_input_require_grads"):
        getattr(model, "enable_input_require_grads")()

def _enable_gradient_checkpointing(self, model: object) -> None:
    """Enable gradient checkpointing with non-reentrant mode for memory efficiency."""
    if hasattr(model, "gradient_checkpointing_enable"):
        getattr(model, "gradient_checkpointing_enable")(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
```

### Change 4: Add new class-level constants

```python
_BNB_QUANT_TYPE: str = "nf4"
_DEVICE_MAP: str = "auto"
_DTYPE_BFLOAT16: str = "bfloat16"
_DTYPE_FLOAT16: str = "float16"
_DTYPE_FLOAT32: str = "float32"
```

### Method Length Verification

| Method | Logical Lines | Status |
|--------|--------------|--------|
| `_load_sft_model()` | 17 | ✅ ≤ 20 |
| `_merge_and_reapply_lora()` | 12 | ✅ ≤ 20 |
| `_resolve_dtype()` | 7 | ✅ ≤ 20 |
| `_enable_input_grads()` | 2 | ✅ ≤ 20 |
| `_enable_gradient_checkpointing()` | 3 | ✅ ≤ 20 |

---

## Task 1.4: Update `tests/unit/training/test_sft_trainer.py`

### Change 1: Remove `unsloth` module mock

In `mock_env` fixture, remove `monkeypatch.setitem(sys.modules, "unsloth", ...)`.

### Change 2: Update `_build_mock_objects()` method

Replace `mock_unsloth.FastLanguageModel.from_pretrained` with:
- `mock_transformers.AutoModelForCausalLM.from_pretrained`
- `mock_transformers.AutoTokenizer.from_pretrained`
- `mock_transformers.BitsAndBytesConfig`

Replace `mock_unsloth.FastLanguageModel.get_peft_model` with:
- `mock_peft.get_peft_model`
- `mock_peft.LoraConfig`
- `mock_peft.TaskType`

### Change 3: Update assertions

All assertions checking `mock_env["unsloth"].FastLanguageModel.*` become assertions against
`mock_env["peft"].get_peft_model` or `mock_env["transformers"].AutoModelForCausalLM.*`.

---

## Task 1.5: Update `tests/unit/training/test_rl_trainer.py`

### Change 1: Update `test_load_sft_model_and_reapply_lora`

Replace `mock_unsloth` setup with `mock_transformers` setup:
- `mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model`
- `mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer`
- `mock_transformers.BitsAndBytesConfig = MagicMock()`

Replace `mock_unsloth.FastLanguageModel.get_peft_model` with:
- `mock_peft.get_peft_model.return_value = mock_model`

Replace `patch.dict("sys.modules", {"unsloth": mock_unsloth, ...})` with:
- `patch.dict("sys.modules", {"transformers": mock_transformers, "peft": mock_peft})`

---

## Task 1.6: Run CI

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

# Stage 1: Lint
ruff check src/ tests/

# Stage 2: Type check
mypy src/

# Stage 3: Tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `unsloth` import removed from all source files (`sft_trainer.py`, `rl_trainer.py`)
- [ ] `pyproject.toml` no longer references `unsloth`; references `bitsandbytes` instead
- [ ] `SFTTrainerPipeline._load_model()` uses `AutoModelForCausalLM` + `BitsAndBytesConfig`
- [ ] `SFTTrainerPipeline._apply_lora()` uses `peft.get_peft_model()` + `LoraConfig`
- [ ] `RLTrainerPipeline._load_sft_model()` uses `AutoModelForCausalLM` + `BitsAndBytesConfig`
- [ ] `RLTrainerPipeline._merge_and_reapply_lora()` uses `peft.get_peft_model()` + `LoraConfig`
- [ ] All tests updated to mock `transformers` and `peft` instead of `unsloth`
- [ ] All methods ≤ 20 logical lines
- [ ] All files ≤ 500 lines
- [ ] `ruff check` clean, `mypy` clean
- [ ] Tests pass with ≥90% coverage

---

## Files Modified in This Enhancement

```
pyproject.toml
src/training/sft_trainer.py
src/training/rl_trainer.py
tests/unit/training/test_sft_trainer.py
tests/unit/training/test_rl_trainer.py
```
